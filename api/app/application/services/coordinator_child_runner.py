"""C2 v1 CoordinatorChildRunner — full worker contract (spec §8.3 + §8.5 + §14.3.1).

This is the sole owner of the coordinator-step child's lifecycle:

- Wires the cancel listener (subscribes per-root mailbox stream for
  CANCEL_REQUEST envelopes addressed to this child, calls request_stop on
  match — spec §8.5.3 race table accepts the pre-subscribe gap, mitigated
  by the listener's ready_event start gate).
- Invokes the inner runner (typically the AgentTaskRunner returned by
  ChildAgentTaskRunnerFactory.build) with the assembled child prompt.
- Catches CancelledByEventError / ChildScopeViolation / asyncio.TimeoutError /
  generic exception, and routes each to the correct finalizer.
- On natural completion, branches by work_unit.phase:
    write       → _finalize_success            (RESULT_READY + PatchManifest)
    exploration → _finalize_exploration_proposal
                  (RESULT_READY + NeedsAuthorizationDetails.proposed_write_plan)

The 7 finalizers each build their payload, publish via the envelope_factory
+ publisher pair (single source of truth for envelope construction), and
return. ``listener.shutdown(timeout=1.0)`` runs in ``finally`` so the
listener task is reaped on every exit path.

Spec-anchored invariants pinned in tests:
- request_stop(reason) is first-wins; cancel_event.set() is idempotent
  (TestRequestStopSemantics in test_coordinator_child_runner_skeleton.py).
- Exactly one envelope published per run_work_unit call
  (test_publisher_called_exactly_once_per_run).
- correlation_id == coordinator_run_id wire contract
  (test_envelope_factory_called_with_correlation_id).
- Listener.start() completes + ready_event set BEFORE inner_runner.invoke
  is called (test_run_work_unit_awaits_listener_ready_before_invoke).
- ScopeDecision → NeedsAuthorizationDetails.reason map is complete
  (test_scope_decision_to_reason_mapping_complete, 7 enum values).
- PARENT_CANCEL produces CANCEL_ACK NOT RESULT_READY
  (test_finalize_cancelled_on_parent_cancel, spec §8.5 r5 P0-2).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Mapping, Optional, TYPE_CHECKING

from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
    ProposedWritePlan,
)
from app.domain.models.patch_manifest import PatchManifest
from typing import Protocol, runtime_checkable

from app.application.services.coordinator_child_cancel_listener import (
    CoordinatorChildCancelListener,
)
from app.domain.services.graphs.react_graph import CancelledByEventError
from app.domain.services.permission.child_scope_violation import (
    ChildScopeViolation,
)
from app.domain.services.prompts.assembler import PromptAssembler

if TYPE_CHECKING:
    from app.domain.models.work_unit import WorkUnit


logger = logging.getLogger(__name__)


class StopReason(StrEnum):
    """C2 PR-3 §14.3.1 — sole authoritative stop classifier."""

    PARENT_CANCEL = "parent_cancel"
    TOKEN_BUDGET = "token_budget"
    WALLCLOCK_BUDGET = "wallclock_budget"


# spec §6.2 — ScopeDecision str values are the canonical
# NeedsAuthorizationDetails.reason values, so a direct passthrough is the
# right map. The explicit dict is kept (rather than ``reason=exc.decision.value``)
# so a future schema change in either enum surfaces as a test failure HERE
# (test_scope_decision_to_reason_mapping_complete) rather than silently.
@runtime_checkable
class CoordinatorChildInnerRunner(Protocol):
    """[C2 PR-4 r2 P0] Protocol for the runner passed to CoordinatorChildRunner.

    The live ``AgentTaskRunner`` (api/app/domain/services/agent_task_runner.py)
    does NOT currently satisfy this Protocol — it exposes ``invoke(task)`` not
    ``invoke_until_done(user_message)``. PR-5's runner_starter is the integration
    point that wraps AgentTaskRunner in an adapter conforming to this Protocol
    (the adapter assembles the Task object, calls invoke, awaits the inner
    completion, and returns a ``done_event`` that carries the final state).

    Until PR-5 lands the adapter, the only conforming caller is the unit-test
    suite (which mocks the inner runner directly). The runtime guard in
    ``run_work_unit`` fails closed with a clear ``TypeError`` instead of an
    obscure AttributeError on the wrong type.
    """

    async def invoke_until_done(self, *, user_message: str) -> "ChildRunResult":  # noqa: D401
        ...


@dataclass(frozen=True)
class ChildRunResult:
    """[C2 finish-core §5.1.1] Return value of
    ``CoordinatorChildInnerRunner.invoke_until_done``.

    ``done_event`` is the child's terminal output event (a ``DoneEvent`` on
    natural completion). ``tool_calls`` are the child's ``ToolEvent``s
    (status=CALLING) captured off the output stream — the data channel
    ``_extract_patch_files_from_history`` (§5.1.5) uses to find which paths
    the child wrote (``DoneEvent`` itself carries only ``metrics``).

    The in-repo consumer (``run_work_unit``) currently binds the whole result
    and passes it through unchanged; rewiring it to unpack ``.tool_calls`` /
    ``.done_event`` is staged for F2.3 (patch-extraction).
    """

    done_event: Any
    tool_calls: tuple[Any, ...] = ()


# [r7 P2#3] Narrow the value type to the closed NeedsAuthorizationDetails
# reason Literal so a typo would fail at type-check time instead of needing
# a runtime ``type: ignore`` at the consumer. ``exploration_proposal`` is in
# the reason Literal but NOT in this map — it's only reachable from natural
# ReAct done in the exploration finalizer, not from a ScopeDecision.
_NeedsAuthReason = Literal[
    "out_of_tool_allowlist", "out_of_path_lease", "op_mismatch",
    "hard_blocked", "budget_exhausted", "lease_expired", "revision_drift",
]
_SCOPE_DECISION_TO_REASON: Mapping[str, _NeedsAuthReason] = {
    "out_of_tool_allowlist": "out_of_tool_allowlist",
    "out_of_path_lease": "out_of_path_lease",
    "op_mismatch": "op_mismatch",
    "hard_blocked": "hard_blocked",
    "budget_exhausted": "budget_exhausted",
    "lease_expired": "lease_expired",
    "revision_drift": "revision_drift",
}


class CoordinatorChildRunner:
    """Owns the lifecycle of one coordinator-step child."""

    def __init__(
        self,
        *,
        cancel_event: asyncio.Event,
        inner_runner: Any = None,
        publisher: Any = None,
        parent_sandbox: Any = None,
        child_sandbox: Any = None,  # [finish-core §5.1.2] ParentSandboxPort over the child handle
        artifact_storage: Any = None,
        envelope_factory: Any = None,
        parent_session_id: str = "",
        coordinator_run_id: str = "",
        mailbox_subscriber: Any = None,
    ) -> None:
        self._cancel_event = cancel_event
        self._stop_reason: Optional[StopReason] = None
        self._inner_runner = inner_runner
        self._publisher = publisher
        self._parent_sandbox = parent_sandbox
        self._child_sandbox = child_sandbox  # seed-install (§5.1.4) + patch-extraction (§5.1.5)
        self._artifact_storage = artifact_storage
        # Lazy default for envelope_factory keeps the PR-3 skeleton ctor
        # signature (which allowed envelope_factory=None) working.
        if envelope_factory is None:
            from app.application.services.coordinator_envelope_factory import (
                CoordinatorEnvelopeFactory,
            )
            envelope_factory = CoordinatorEnvelopeFactory()
        self._envelope_factory = envelope_factory
        self._parent_session_id = parent_session_id
        self._coordinator_run_id = coordinator_run_id
        self._mailbox_subscriber = mailbox_subscriber

    def request_stop(self, reason: StopReason) -> None:
        """Sole entry — sets cancel_event + records first-wins stop reason.

        Idempotent: subsequent calls do not overwrite ``_stop_reason``.
        """
        if self._stop_reason is None:
            self._stop_reason = reason
        self._cancel_event.set()

    async def run_work_unit(
        self,
        *,
        coordinator_run_id: str,
        work_unit: "WorkUnit",
        child_session_id: str,
        spawn_manifest: Any,
        cancel_event: asyncio.Event,
        root_session_id: str = "",
    ) -> Any:
        """Run one work unit through the inner runner with cancel listener
        wired and finalizers routing every exit path."""
        # [C2 PR-4 r2 P0] Fail closed if the wired inner_runner doesn't satisfy
        # the Protocol (e.g. someone wires a raw AgentTaskRunner before the
        # PR-5 adapter lands). AttributeError mid-graph would surface as
        # _finalize_failed("AttributeError: ...") — useless for debugging.
        # TypeError here is loud and pre-graph, with a pointer to PR-5.
        if not hasattr(self._inner_runner, "invoke_until_done"):
            raise TypeError(
                f"inner_runner of type {type(self._inner_runner).__name__!r} "
                "does not implement CoordinatorChildInnerRunner protocol "
                "(missing invoke_until_done). PR-5's runner_starter adapter "
                "is the integration point that wraps AgentTaskRunner."
            )
        # Refresh coordinator_run_id at call time so envelope_factory.make_*
        # uses the correlation_id for THIS run (the ctor value may be a
        # placeholder when the runner is built once and reused for replays).
        self._coordinator_run_id = coordinator_run_id

        listener = CoordinatorChildCancelListener(
            subscriber=self._mailbox_subscriber,
            root_session_id=root_session_id,
            child_session_id=child_session_id,
            runner=self,
        )
        await listener.start()
        # spec §8.5.3 ready-before-run gate — guarantees the subscriber's
        # consumer group is registered before the inner runner can produce
        # tool events the parent might try to cancel mid-flight.
        await listener.ready_event.wait()

        # [r3 P1#2] §8.4 worker-start checkpoint (#1 in spec). If parent set
        # cancel_event BEFORE listener.start completed (the pre-subscribe race
        # window the listener.ready_event gate is supposed to close), we still
        # might find the event set here. Stamp PARENT_CANCEL and finalize
        # immediately instead of spending an LLM call on a child that's already
        # been cancelled.
        if cancel_event.is_set() and self._stop_reason is None:
            self._stop_reason = StopReason.PARENT_CANCEL
        if self._stop_reason is not None:
            main_result: Any
            main_exc: BaseException | None = None
            try:
                main_result = await self._finalize_by_stop_reason(
                    coordinator_run_id, work_unit, child_session_id,
                )
            except BaseException as exc:  # noqa: BLE001 — re-raised after shutdown
                main_exc = exc
            finally:
                await self._safe_listener_shutdown(listener)
            if main_exc is not None:
                raise main_exc
            return main_result

        try:
            try:
                # TODO(PR-9 wiring): wrap ``inner_runner.invoke_until_done`` in
                # ``start_wallclock_watchdog(runner=self,
                # max_wallclock_seconds=coordinator_limits.
                # max_wallclock_seconds_per_child)`` so spec §14.3 #5 (wall
                # clock budget for the child) actually trips. The
                # ``CoordinatorChildWallclockWatchdog`` is shipped in PR-6
                # but its task is not started anywhere yet — wiring lands
                # in a follow-up integration task (likely PR-9) once the
                # ``runner_starter`` adapter (PR-5) is extended to accept a
                # ``max_wallclock_seconds`` argument that this runner can
                # thread through. Until then, ``StopReason.WALLCLOCK_BUDGET``
                # only ever fires via the supervisor's 600s backstop, and
                # the §14.3 #5 internal cap is dead code.
                #
                # Similarly, ``BudgetEnforcementCallback`` (the token-cost
                # gate from §14.3 #1, also shipped in PR-6) needs to be
                # bound to the inner_runner's LLM callbacks list at
                # construction time. The runner_starter adapter is the
                # natural place to thread it. Until that lands, token-cost
                # checks happen only post-hoc via cost_summary aggregation,
                # not as an in-flight LLM-call guard.
                #
                # TODO(F2.3): result is a ChildRunResult; .tool_calls/.done_event
                # unpacking lands in F2.3 patch-extraction. Bound whole here
                # intentionally for now (passed through _finalize_success →
                # _extract_patch_files_from_history, which currently ignore it).
                done_event = await self._inner_runner.invoke_until_done(
                    user_message=self._build_child_prompt(work_unit, spawn_manifest),
                )
            except CancelledByEventError:
                return await self._finalize_by_stop_reason(
                    coordinator_run_id, work_unit, child_session_id,
                )
            except ChildScopeViolation as exc:
                return await self._finalize_needs_authorization_from_scope(
                    coordinator_run_id, work_unit, child_session_id, exc,
                )
            except asyncio.TimeoutError:
                return await self._finalize_timed_out(
                    coordinator_run_id, work_unit, child_session_id,
                )
            except Exception as exc:
                return await self._finalize_failed(
                    coordinator_run_id, work_unit, child_session_id, exc,
                )

            # natural ReAct done — branch by phase (spec §8.3 r13).
            if work_unit.phase == "exploration":
                return await self._finalize_exploration_proposal(
                    coordinator_run_id, work_unit, child_session_id, done_event,
                )
            return await self._finalize_success(
                coordinator_run_id, work_unit, child_session_id, done_event,
            )
        finally:
            # Run on EVERY exit path — success, failure, scope violation,
            # timeout, cancel — so the listener task is reaped instead of
            # leaking + processing further CANCEL_REQUEST envelopes that
            # nobody consumes.
            #
            # [r3 P1#3] Wrap to prevent shutdown errors from MASKING the
            # primary exception. Without the wrapper, a Python ``finally``
            # block that raises a NEW exception replaces the in-flight one
            # — so a noisy listener shutdown would obscure the real
            # finalizer-publish error or success return.
            #
            # [r3 P2#1] Note: shutdown timeout (1.0s) only runs AFTER
            # invoke_until_done returns/raises. If invoke_until_done hangs
            # forever, this timeout does NOT unwind the listener — the
            # wallclock budget watchdog (PR-6) is the upstream cancel that
            # forces invoke_until_done to terminate.
            await self._safe_listener_shutdown(listener)

    # ---- Finalizer matrix ------------------------------------------------ #

    async def _finalize_by_stop_reason(
        self, run_id: str, wu: "WorkUnit", child_id: str,
    ) -> Any:
        """[spec §14.3.1] Route to the correct finalizer based on
        ``_stop_reason``. Both budget reasons land in the budget finalizer;
        PARENT_CANCEL (or a defensive None fallback) lands in cancelled."""
        if self._stop_reason in (
            StopReason.TOKEN_BUDGET, StopReason.WALLCLOCK_BUDGET,
        ):
            return await self._finalize_needs_authorization_budget(
                run_id, wu, child_id,
            )
        if self._stop_reason is None:
            logger.warning(
                "CoordinatorChildRunner: cancel raised with stop_reason=None; "
                "defensive fallback to _finalize_cancelled (child_id=%s)",
                child_id,
            )
        return await self._finalize_cancelled(run_id, wu, child_id)

    async def _finalize_success(
        self, run_id: str, wu: "WorkUnit", child_id: str, done_event: Any,
    ) -> ResultReadyPayload:
        """[spec §8.3 write] Extract writes from inner runner history, build
        PatchManifest, publish RESULT_READY(SUCCESS)."""
        files = await self._extract_patch_files_from_history(run_id, wu, done_event)
        patch_manifest = PatchManifest(
            patch_id=f"{run_id}:{wu.work_unit_id}:p",
            coordinator_run_id=run_id,
            work_unit_id=wu.work_unit_id,
            files=tuple(files),
        )
        payload = ResultReadyPayload(
            summary=f"completed {wu.work_unit_id}",
            outcome=ResultReadyOutcome.SUCCESS,
            patch_manifest=patch_manifest,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_exploration_proposal(
        self, run_id: str, wu: "WorkUnit", child_id: str, done_event: Any,
    ) -> ResultReadyPayload:
        """[spec §8.3 r13] phase=exploration natural done → NEEDS_AUTHORIZATION
        with ProposedWritePlan."""
        proposal = self._extract_proposed_write_plan(done_event)
        details = NeedsAuthorizationDetails(
            reason="exploration_proposal",
            proposed_write_plan=proposal,
        )
        payload = ResultReadyPayload(
            summary=f"exploration proposal for {wu.work_unit_id}",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=details,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_failed(
        self, run_id: str, wu: "WorkUnit", child_id: str, exc: Exception,
    ) -> ResultReadyPayload:
        payload = ResultReadyPayload(
            summary=f"failed: {exc}",
            outcome=ResultReadyOutcome.FAILED,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_timed_out(
        self, run_id: str, wu: "WorkUnit", child_id: str,
    ) -> ResultReadyPayload:
        payload = ResultReadyPayload(
            summary=f"child timed out: {wu.work_unit_id}",
            outcome=ResultReadyOutcome.TIMED_OUT,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_needs_authorization_from_scope(
        self, run_id: str, wu: "WorkUnit", child_id: str,
        exc: ChildScopeViolation,
    ) -> ResultReadyPayload:
        # [r3 P2#2] Fail-closed fallback for unmapped ScopeDecision values
        # (e.g. a future enum addition that hasn't extended the map yet, or
        # a fabricated exception carrying ScopeDecision.IN_SCOPE). Default
        # to ``hard_blocked`` — the most restrictive interpretation — and
        # log so the missing mapping surfaces in audit.
        decision_value = exc.decision.value
        reason = _SCOPE_DECISION_TO_REASON.get(decision_value)
        if reason is None:
            logger.warning(
                "CoordinatorChildRunner: unmapped ScopeDecision=%s; "
                "falling back to hard_blocked",
                decision_value,
            )
            reason = "hard_blocked"
        details = NeedsAuthorizationDetails(
            reason=reason,
            requested_tool=exc.tool_name,
            requested_paths=(exc.target_path,) if exc.target_path else (),
        )
        payload = ResultReadyPayload(
            summary=f"out-of-scope: {exc.decision.value}",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=details,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_needs_authorization_budget(
        self, run_id: str, wu: "WorkUnit", child_id: str,
    ) -> ResultReadyPayload:
        details = NeedsAuthorizationDetails(
            reason="budget_exhausted",
            observed_evidence=self._build_budget_evidence(),
        )
        payload = ResultReadyPayload(
            summary="budget exhausted",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=details,
        )
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_cancelled(
        self, run_id: str, wu: "WorkUnit", child_id: str,
    ) -> CancelAckPayload:
        """[spec §8.5 r5 P0-2] PARENT_CANCEL produces CANCEL_ACK(cancelled),
        NOT RESULT_READY — RESULT_READY downstream consumers count it as a
        completion, but cancel is explicitly not a completion."""
        payload = CancelAckPayload(final_state="cancelled")
        await self._publish_cancel_ack(child_id, payload)
        return payload

    # ---- Listener shutdown safety wrapper ------------------------------- #

    async def _safe_listener_shutdown(self, listener: Any) -> None:
        """[r3 P1#3] Run listener.shutdown WITHOUT masking the in-flight
        exception/return from the main run_work_unit path.

        Python's ``finally`` semantics: a NEW exception raised in finally
        replaces any active exception, so a noisy shutdown error would mask
        the original failure (publisher error, finalizer KeyError, etc.) and
        production debugging would chase the shutdown trace instead of the
        root cause.

        [r5 P1#1] Catch ``Exception`` only — NOT BaseException — so cancel
        propagation (asyncio.CancelledError) + interpreter shutdown
        (SystemExit / KeyboardInterrupt) are NEVER suppressed. Swallowing
        CancelledError here would let the runner return as if the
        in-flight task wasn't cancelled, breaking task-tree cancel
        semantics under structured concurrency.

        Shutdown errors are logged at WARNING (with stack via exc_info) so
        they remain investigable, just not propagated."""
        try:
            await listener.shutdown(timeout=1.0)
        except Exception:  # noqa: BLE001 — CancelledError/SystemExit excluded by design
            logger.warning(
                "CoordinatorChildRunner: listener.shutdown raised — "
                "suppressed to avoid masking primary exception path",
                exc_info=True,
            )

    # ---- Publish helpers (single-source-of-truth wire boundary) --------- #

    # [r3 P1#4 DEFERRED to PR-7] terminal envelope_id is currently a fresh
    # UUID per finalizer invocation (CoordinatorEnvelopeFactory.make_*).
    # If PR-7 crash recovery re-runs the same work_unit's finalizer, the
    # second envelope would carry a different envelope_id and bypass
    # publisher dedup. PR-7's rehydrate path adds deterministic envelope_id
    # construction (e.g. ``f"{coordinator_run_id}:{work_unit_id}:{terminal_type}"``)
    # so retries are wire-level idempotent. For PR-4 (single-run scope),
    # the UUID is sufficient.
    async def _publish_result_ready(
        self, child_id: str, payload: ResultReadyPayload,
    ) -> None:
        envelope = self._envelope_factory.make_result_ready(
            parent_session_id=self._parent_session_id,
            child_session_id=child_id,
            correlation_id=self._coordinator_run_id,
            payload=payload,
        )
        await self._publisher.publish(envelope)

    async def _publish_cancel_ack(
        self, child_id: str, payload: CancelAckPayload,
    ) -> None:
        envelope = self._envelope_factory.make_cancel_ack(
            parent_session_id=self._parent_session_id,
            child_session_id=child_id,
            correlation_id=self._coordinator_run_id,
            payload=payload,
        )
        await self._publisher.publish(envelope)

    # ---- Stub helpers (PR-4 minimal; full impl in PR-5/PR-6) ------------ #

    def _build_child_prompt(self, wu: "WorkUnit", manifest: Any) -> str:
        """[spec §8.7] Compose the restricted child prompt. The PromptAssembler
        static helper (Task 4.8) handles section composition; here we only
        forward the work_unit's planner-visible fields."""
        return PromptAssembler.build_minimal_for_coordinator_child(
            objective=wu.objective,
            phase=wu.phase,
            allowed_paths=[lease.path for lease in wu.write_lease],
            expected_result_schema=wu.expected_result_schema,
        )

    async def _extract_patch_files_from_history(
        self, run_id: str, wu: "WorkUnit", done_event: Any,
    ) -> list[Any]:
        """[PR-4 minimal] Returns an empty list. PR-5 wires the real extraction:
        iterate inner_runner.tool_call_history, filter typed writes
        (file_write / file_str_replace), read final content from the child
        sandbox, SHA-256, upload via artifact_storage, build FilePatchEntry list.

        Returning [] here is acceptable for PR-4: the reducer (PR-5) treats
        an empty patch_manifest as "no writes" — same semantics as a child
        that genuinely produced no file changes. Tests verify the surrounding
        envelope structure; the file-extraction logic is the PR-5 deliverable.
        """
        return []

    def _extract_proposed_write_plan(self, done_event: Any) -> ProposedWritePlan:
        """[PR-4 minimal] Returns an empty proposal. PR-6 wires the real
        extraction (parse the LLM's final assistant message into a structured
        proposed_write_plan, optionally with a MinIO rationale_ref)."""
        return ProposedWritePlan(
            proposed_paths=(),
            proposed_tools=frozenset(),
        )

    def _build_budget_evidence(self) -> str:
        """[PR-4 minimal] Stamps the stop_reason so the parent can attribute
        budget exhaustion to the correct watchdog (token vs. wallclock).
        PR-6 expands with cumulative counters."""
        reason_str = self._stop_reason.value if self._stop_reason else "unknown"
        return f"stop_reason={reason_str}"
