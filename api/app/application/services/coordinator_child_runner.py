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
import hashlib
import logging
import time
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
from app.domain.models.path_validation import (
    CoordinatorPathContractError,
    to_workspace_relative as _domain_to_workspace_relative,
)
from typing import Protocol, runtime_checkable

from app.application.services.coordinator_child_cancel_listener import (
    CoordinatorChildCancelListener,
)
from app.application.services.coordinator_child_wallclock_watchdog import (
    start_wallclock_watchdog,
)
from app.domain.services.graphs.react_graph import CancelledByEventError
from app.domain.services.coordinator_shell_mode_flag import (
    is_coordinator_shell_mode_enabled,
)
from app.domain.services.permission.child_scope_violation import (
    ChildScopeViolation,
)
from app.domain.services.prompts.assembler import PromptAssembler

if TYPE_CHECKING:
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.permission.child_permission_context import (
        ChildBudget,
    )
    from app.domain.external.parent_sandbox import WorkspaceScan, WorkspaceScanEntry


logger = logging.getLogger(__name__)


# [S2 §3.2 C1] Inline-vs-ref ceiling for the SUCCESS PatchManifest. The
# terminal envelope store collapses a persisted payload >64KB to a
# {outcome,_truncated} marker (DbCoordinatorResultEnvelopeStoreRepository
# .persist_terminal, guarding _MAX_PAYLOAD_BYTES), silently dropping a large
# inline manifest → zero-apply. Below this
# ceiling we keep the manifest inline (cheap, no extra MinIO round-trip on
# resolve); above it (strictly > ceiling; exactly == ceiling stays inline, per
# the ``<=`` test below) we upload the manifest to MinIO and carry a tiny
# patch_manifest_ref instead. Margin under 64KB leaves room for the rest of the
# envelope (summary, cost_summary, etc.) that also counts toward the cap.
_MAX_INLINE_MANIFEST_BYTES = 48 * 1024


class _SeedInstallError(Exception):
    """[finish-core §5.1.4] Raised when child seed-install fails (fetch /
    write / digest mismatch). Routed to _finalize_failed (reason seed_*)."""


class _OutOfLeaseWriteError(Exception):
    """[finish-core §5.1.5] A child wrote a path outside its write_lease.

    [C2-full S2 §3.4] Carries a structured ``reason`` (one of the §3.4 reject
    codes) so _finalize_needs_authorization_out_of_lease can surface the SPECIFIC
    NEEDS_AUTHORIZATION reason on the wire instead of a hard-coded literal, plus
    an optional ``offending_path`` that feeds the bounded ``rejection_summary``
    on the envelope (§3.4). The defaults (``out_of_path_lease`` / ``None``)
    preserve every legacy bare-message raise
    (``_OutOfLeaseWriteError(str(...))``) — those keep the existing wire reason
    unchanged and contribute no summary entry."""

    def __init__(
        self,
        message: str = "",
        *,
        reason: str = "out_of_path_lease",
        offending_path: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.offending_path = offending_path


# [C2-full S2 §3.4] EXHAUSTIVE NEEDS_AUTHORIZATION reason vocabulary for
# snapshot-diff group zero-apply. Mirrored in the spec §3.4 reject table.
_SNAPSHOT_REJECT_REASONS: frozenset[str] = frozenset(
    {
        "out_of_path_lease",
        "out_of_tree_lease",
        "special_file",
        "symlink",
        "mode_only_change",
        "indeterminate_kind",
        "scan_truncated",
        "tree_add_target_exists",
        "parent_not_regular",
    }
)

# [C2-full S2 §3.4] bound the envelope's rejection_summary (first N offending
# paths). Group zero-apply raises on the FIRST violation, so in v1 the summary
# is at most one entry; the cap future-proofs a batch-collect variant.
_SNAPSHOT_SUMMARY_CAP: int = 20


def _snapshot_reject(reason: str, path: str) -> "_OutOfLeaseWriteError":
    """Build an _OutOfLeaseWriteError tagged with a §3.4 reason code + path.

    The reason is BOTH embedded in the message (so an aborted `pytest.raises(match=)`
    still reads it) AND carried as the structured ``reason`` attribute, which
    _finalize_needs_authorization_out_of_lease forwards to
    NeedsAuthorizationDetails.reason — i.e. the SPECIFIC §3.4 code reaches the wire,
    not just the exception message. The ``path`` is carried as ``offending_path``
    so the finalizer can populate the bounded ``rejection_summary`` (§3.4)."""
    assert reason in _SNAPSHOT_REJECT_REASONS, f"unknown reject reason {reason!r}"
    return _OutOfLeaseWriteError(
        f"[{reason}] {path}", reason=reason, offending_path=path,
    )


def _scans_semantically_equal(a: "WorkspaceScan", b: "WorkspaceScan") -> bool:
    """[§3.2 F23] Sort entries by rel_path and compare the
    (kind, sha256, size, mode, link_target) tuple per path — NOT a raw JSON/byte
    compare (which false-positives on dict ordering). truncated is part of the
    comparison so an aborted re-scan also counts as drift."""
    if a.truncated != b.truncated:
        return False
    ka, kb = sorted(a.entries), sorted(b.entries)
    if ka != kb:
        return False
    for k in ka:
        ea, eb = a.entries[k], b.entries[k]
        if (ea.kind, ea.sha256, ea.size, ea.mode, ea.link_target) != (
            eb.kind, eb.sha256, eb.size, eb.mode, eb.link_target
        ):
            return False
    return True


def _op_lease_compatible(diff_op: str, lease_op: str) -> bool:
    """A diff op is compatible only with an exactly-matching file-lease op."""
    return diff_op == lease_op


def _canon_tree_prefix(prefix: str) -> str:
    """Canonicalize a tree-lease prefix to its workspace-relative form for
    tree_contains (PR-3 validate_coordinator_tree_prefix already canonicalized
    at lease-construction; this re-applies to_workspace_relative defensively)."""
    return _to_workspace_relative(prefix)


def _to_workspace_relative(path: str) -> str:
    """Canonicalize a child write/lease path to its workspace-relative form.

    Delegates to the domain single source of truth
    (``path_validation.to_workspace_relative``, anchored at ``WORKSPACE_ROOT`` =
    ``/home/ubuntu``) so the abs->rel mapping cannot drift between the lease
    boundary (``_build_work_units_from_requests``) and patch extraction here:

    - already-relative paths pass through unchanged (existing convention);
    - absolute paths under the workspace root are stripped to the relative tail
      (``/home/ubuntu/sub/b.py`` -> ``sub/b.py``);
    - an absolute path OUTSIDE the workspace root (or the root itself) is a
      sandbox-escape attempt; the domain helper raises
      ``CoordinatorPathContractError``, which we translate to
      ``_OutOfLeaseWriteError`` (fail-closed) so the child finalizer routes the
      escape to NEEDS_AUTHORIZATION rather than a silent ValidationError
      swallowed into a FAILED envelope.

    Used for BOTH the lease keys and the child's written paths so the lease
    match is order-independent of the absolute/relative form each side used. The
    directory-component requirement (single-path contract) is enforced
    separately at the lease boundary + the ``FilePatchEntry`` wire schema — this
    helper only strips the workspace prefix.
    """
    try:
        return _domain_to_workspace_relative(path)
    except CoordinatorPathContractError as exc:
        raise _OutOfLeaseWriteError(str(exc)) from exc


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
    # [C2-full S2 §3.4] snapshot-diff group zero-apply reject codes.
    "out_of_tree_lease", "special_file", "symlink", "mode_only_change",
    "indeterminate_kind", "scan_truncated",
    "tree_add_target_exists", "parent_not_regular",
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
        # [C2b budget D2] Per-child caps. None tolerated (legacy/unit
        # constructions): no watchdog, no evidence enrichment — INV-B6.
        # Deliberately ChildBudget (not the whole CoordinatorLimits): the
        # runner must not hold global config.
        budget: "ChildBudget | None" = None,
        # [C2b budget D10] Optional CoordinatorMetrics for the budget
        # finalizer's best-effort exhaustion counter (INV-B9).
        coordinator_metrics: Any = None,
        # [S2 PR-4 §3.6] live snapshot caps; defaults to CoordinatorLimits()
        # for legacy/unit constructions.
        snapshot_limits: Any = None,
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
        self._budget = budget
        # [S2 PR-4] snapshot caps for the bounded finalizer; starter overwrites
        # with the live CoordinatorLimits. Default mirrors CoordinatorLimits.
        from app.domain.services.coordinator_limits import CoordinatorLimits
        self._snapshot_limits = snapshot_limits or CoordinatorLimits()
        # [S2 PR-4 §3.2] PRE-scan captured after seed-install; the bounded
        # finalizer diffs it against the quiesced POST scan. None until set.
        self._pre_scan: "Optional[WorkspaceScan]" = None
        self._coordinator_metrics = coordinator_metrics
        # [C2b budget D1/D5] Reference to the child's BudgetEnforcementCallback
        # (attach_budget_callback) so _build_budget_evidence can read
        # cumulative_usd. The LLM-side wiring goes through the
        # adapter→runner→flow setter chain, NOT through this reference.
        self._budget_callback: Any = None
        # [C2b budget D5] Monotonic stamp of inner-invoke start, for the
        # optional wallclock_elapsed_seconds evidence field.
        self._inner_invoke_started_monotonic: float | None = None

    def attach_budget_callback(self, cb: Any) -> None:
        """[C2b budget D1] Keep a reference to the child's
        BudgetEnforcementCallback for D5 evidence (cumulative_usd read in
        _build_budget_evidence). Separate from the setter chain that wires
        the callback into the child LLM callbacks list."""
        self._budget_callback = cb

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

        # [S2 §3.6] Two-level gate: shell-mode runs iff master flag ON AND
        # the unit carries shell_mode. Computed once; reused for the PRE scan
        # and the finalize routing so they can never diverge.
        shell_mode_active = (
            getattr(work_unit, "shell_mode", False)
            and is_coordinator_shell_mode_enabled()
        )

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
            # [finish-core §5.1.4] Seed-install BEFORE the ReAct loop. A
            # _SeedInstallError finalizes FAILED (reason seed_*) and NEVER
            # self-destroys (M1) — _finalize_failed only publishes
            # RESULT_READY(FAILED). INV-F1.4b. It short-circuits the run
            # (returns) before the inner invoke.
            #
            # [F2 P1] Placed INSIDE this outer try so the single
            # ``finally: _safe_listener_shutdown`` below reaps the cancel
            # listener on EVERY seed exit too — including an
            # ``asyncio.CancelledError`` raised mid-seed (parent cancels
            # during _install_seed's awaits), which is NOT a _SeedInstallError
            # and would otherwise escape with the listener still running. The
            # listener is now reaped EXACTLY ONCE on: _SeedInstallError,
            # CancelledError-during-seed, invoke failure, and natural success.
            try:
                await self._install_seed(work_unit)
                # [S2 PR-4 §3.2/§3.6] PRE scan AFTER seed, BEFORE the inner
                # invoke, only when shell-mode is ACTIVE (flag_on AND
                # wu.shell_mode — see shell_mode_active above). A scan RPC failure
                # here is a _SeedInstallError so it routes to _finalize_failed
                # (terminal envelope published — never escapes run_work_unit).
                #
                # [S2 §3.2 finalizer-budget invariant] The PRE scan await is
                # bounded by ``max_snapshot_seconds`` via asyncio.wait_for — EXACTLY
                # like the POST scan in _capture_shell_snapshot_diff (Task 4.6).
                # The sandbox-side ``max_seconds`` only bounds the WALK; a hung /
                # stalled snapshot RPC would otherwise block on the api side until
                # the DockerSandbox HTTP client timeout (~600s,
                # docker_sandbox.py:51), NOT the snapshot budget — violating §3.2's
                # "ALL finalizer snapshot awaits live inside the bounded budget"
                # rule. With the guard, a stalled PRE RPC raises
                # asyncio.TimeoutError, is wrapped into _SeedInstallError below, and
                # routes to _finalize_failed (terminal FAILED published) instead of
                # a ~600s hang.
                if shell_mode_active:
                    try:
                        self._pre_scan = await asyncio.wait_for(
                            self._child_sandbox.snapshot_workspace(
                                max_paths=self._snapshot_limits.max_snapshot_paths,
                                max_files=self._snapshot_limits.max_snapshot_files,
                                max_total_bytes=self._snapshot_limits.max_snapshot_total_bytes,
                                max_seconds=self._snapshot_limits.max_snapshot_seconds,
                            ),
                            timeout=self._snapshot_limits.max_snapshot_seconds,
                        )
                    except Exception as scan_exc:  # noqa: BLE001 — incl. asyncio.TimeoutError
                        raise _SeedInstallError(
                            f"pre_scan_failed: {scan_exc}"
                        ) from scan_exc
            except _SeedInstallError as exc:
                return await self._finalize_failed(
                    coordinator_run_id, work_unit, child_session_id, exc,
                )

            # [F2 P1] inner invoke try — nested in the same outer try whose
            # finally reaps the listener on every exit (success/failure/cancel).
            try:
                # [C2b budget D2/A1] Wallclock watchdog wraps ONLY the inner
                # invoke. The inner ``finally: wd.cancel()`` runs BEFORE the
                # except arms below (Python semantics) — so by the time ANY
                # finalizer executes, the watchdog can no longer trip and
                # dirty ``_stop_reason`` mid-publish (INV-B2). Seed-install
                # and finalizer windows are deliberately uncovered (spec
                # §6-L9 / F8).
                wd: asyncio.Task | None = None
                if self._budget is not None:
                    _max_wc = self._budget.max_wallclock_seconds
                    if _max_wc > 0:
                        wd = start_wallclock_watchdog(
                            runner=self, max_wallclock_seconds=_max_wc,
                        )
                    else:
                        # [spec L8] Non-positive cap: production-unreachable
                        # (env loader rejects), but direct construction can
                        # hit it — honest DISABLED + WARNING beats a
                        # trips-on-first-call cap=0 watchdog.
                        logger.warning(
                            "CoordinatorChildRunner: max_wallclock_seconds=%s "
                            "non-positive — wallclock watchdog DISABLED for "
                            "this child",
                            _max_wc,
                        )
                # [C2b budget D5] Stamp inner-invoke start for the optional
                # wallclock_elapsed_seconds evidence field.
                self._inner_invoke_started_monotonic = time.monotonic()
                # TODO(F2.3): result is a ChildRunResult; .tool_calls/.done_event
                # unpacking lands in F2.3 patch-extraction. Bound whole here
                # intentionally for now (passed through _finalize_success →
                # _extract_patch_files_from_history, which currently ignore it).
                try:
                    done_event = await self._inner_runner.invoke_until_done(
                        user_message=self._build_child_prompt(work_unit, spawn_manifest),
                    )
                finally:
                    # [INV-B2] Cancel synchronously, exception-free, BEFORE the
                    # outer except arms (and therefore before every finalizer).
                    if wd is not None:
                        wd.cancel()
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

            # [INV-B1 终线确定性 — impl-audit R2#1] A trip (budget/watchdog/
            # parent cancel) can land while the adapter is ALREADY blocked
            # inside output_stream.get() awaiting the final DoneEvent: the
            # trip's request_stop() sets cancel_event + _stop_reason, then the
            # DoneEvent enqueues, and get() returns it — the adapter's
            # loop-top cancel check has already passed for that iteration, so
            # invoke_until_done returns "done" without raising. Re-assert
            # determinism at the runner's single exit point: any _stop_reason
            # RECORDED BY HERE wins over natural success (spec :51's "strictly
            # before enqueue" argument only orders the events; it does not
            # guarantee the adapter re-checks between them). This fully covers
            # the budget invariant (INV-B1): a budget trip can only be set
            # DURING the inner invoke (watchdog cancelled at the inner finally
            # above; no LLM calls after invoke returns), so it is always
            # recorded by the time control reaches here.
            #
            # [impl-audit R3#1 scope] This does NOT cover a PARENT_CANCEL that
            # arrives LATER, during _finalize_success's own awaits (the cancel
            # listener stays live until the outer finally). That is the spec
            # §6-L9 RESULT_READY-vs-CANCEL_REQUEST interleaving, frozen as
            # benign: the child publishes a terminal SUCCESS for work it
            # actually completed, the supervisor observes it as terminal, and
            # any redundant parent force-terminate is harmless (row already
            # terminal, reaper mapping consistent). Closing that window would
            # need a listener freeze/terminal latch — a C2 cancel-correctness
            # change out of this PR's scope, and contrary to §6-L9's accepted
            # design.
            if self._stop_reason is not None:
                return await self._finalize_by_stop_reason(
                    coordinator_run_id, work_unit, child_session_id,
                )

            # natural ReAct done — branch by phase (spec §8.3 r13).
            if work_unit.phase == "exploration":
                return await self._finalize_exploration_proposal(
                    coordinator_run_id, work_unit, child_session_id, done_event,
                )
            # [S2 §3.6] Shell-finalize only when the gate is ACTIVE (flag_on AND
            # wu.shell_mode). Flag-off ⇒ typed path even if wu.shell_mode leaked.
            if shell_mode_active:
                return await self._finalize_success_shell(
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
            # wallclock budget watchdog (LIVE since C2b budget wiring; D2
            # inner-finally above) is the upstream cancel that forces
            # invoke_until_done to terminate.
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

    async def _build_manifest_payload_inline_or_ref(
        self,
        run_id: str,
        wu: "WorkUnit",
        patch_manifest: PatchManifest,
        *,
        summary: str,
    ) -> ResultReadyPayload:
        """[S2 §3.2 C1] BUILD (do NOT publish) a SUCCESS RESULT_READY payload
        carrying the manifest either INLINE (small) or BY MinIO REF (large), and
        RETURN it. The terminal store truncates a persisted payload >64KB to a
        marker (dropping a large inline manifest → silent zero-apply); uploading
        the manifest and carrying a tiny ``patch_manifest_ref`` instead lets a
        large write set survive persist + rehydrate. Inline and ref are mutually
        exclusive (ResultReadyPayload validator, Task 2.1).

        [defect-fix R2 P1] BUILD-ONLY by design: the ``put_content_addressed_bytes``
        upload is a NEW failure source, so the caller invokes this INSIDE its
        protected (extraction) region — an upload failure then degrades to a
        terminal FAILED/NEEDS_AUTHORIZATION envelope rather than escaping with no
        terminal envelope and stranding the parent waiter. The caller does the
        single unprotected publish OUTSIDE that region (F2 P0). Reused by the
        PR-4 snapshot-capture finalizer (same name/signature; it builds inside
        its own protected region and publishes outside)."""
        manifest_json = patch_manifest.model_dump_json()
        if len(manifest_json.encode("utf-8")) <= _MAX_INLINE_MANIFEST_BYTES:
            return ResultReadyPayload(
                summary=summary,
                outcome=ResultReadyOutcome.SUCCESS,
                patch_manifest=patch_manifest,
            )
        ref = await self._artifact_storage.put_content_addressed_bytes(
            prefix=f"coordinator/{run_id}/{wu.work_unit_id}/manifest/",
            content=manifest_json.encode("utf-8"),
        )
        logger.info(
            "CoordinatorChildRunner: manifest for wu=%s exceeds inline ceiling "
            "(%d bytes > %d); carrying by ref=%s",
            wu.work_unit_id,
            len(manifest_json.encode("utf-8")),
            _MAX_INLINE_MANIFEST_BYTES,
            ref,
        )
        return ResultReadyPayload(
            summary=summary,
            outcome=ResultReadyOutcome.SUCCESS,
            patch_manifest_ref=ref,
        )

    async def _finalize_success(
        self, run_id: str, wu: "WorkUnit", child_id: str, done_event: Any,
    ) -> ResultReadyPayload:
        """[spec §8.3 write] Extract writes from inner runner history, build
        PatchManifest, publish RESULT_READY(SUCCESS).

        [F2 P0] Extraction (``_extract_patch_files_from_history``) can raise
        ``_OutOfLeaseWriteError`` (child wrote outside its lease) and
        PatchManifest/FilePatchEntry construction can raise
        ``pydantic.ValidationError`` (e.g. a path that violates the wire
        schema) or sandbox/artifact errors. These run inside ``run_work_unit``'s
        outer ``try`` which only has a ``finally`` (no ``except``), so an
        escape here would leave NO terminal envelope published — the child
        would crash in the starter's done-callback and the parent waiter would
        strand. Wrap ONLY the extraction + manifest build (NOT the final
        ``_publish_result_ready``, to avoid swallowing a publish failure and
        re-publishing) so a write-phase failure degrades to a terminal
        NEEDS_AUTHORIZATION / FAILED envelope instead."""
        try:
            files = await self._extract_patch_files_from_history(run_id, wu, done_event)
            patch_manifest = PatchManifest(
                patch_id=f"{run_id}:{wu.work_unit_id}:p",
                coordinator_run_id=run_id,
                work_unit_id=wu.work_unit_id,
                files=tuple(files),
            )
            # [S2 §3.2 C1 + defect-fix R2 P1] BUILD the payload (inline or, for
            # an oversized manifest, by MinIO ref) INSIDE this try. The by-ref
            # upload is a NEW failure source: keeping it here means an upload
            # failure degrades to a terminal FAILED envelope via the generic
            # ``except`` below — exactly like an extraction failure — instead of
            # escaping run_work_unit's finally-only outer wrapper with NO
            # terminal envelope (which would strand the parent waiter).
            payload = await self._build_manifest_payload_inline_or_ref(
                run_id, wu, patch_manifest,
                summary=f"completed {wu.work_unit_id}",
            )
        except _OutOfLeaseWriteError as exc:
            return await self._finalize_needs_authorization_out_of_lease(
                run_id, wu, child_id, exc,
            )
        except Exception as exc:  # noqa: BLE001 — ValidationError + sandbox/artifact/upload errors
            return await self._finalize_failed(run_id, wu, child_id, exc)
        # [F2 P0] The single _publish_result_ready intentionally OUTSIDE the try
        # above so a publish failure is not swallowed + re-published as FAILED.
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_success_shell(
        self, run_id: str, wu: "WorkUnit", child_id: str, done_event: Any,
    ) -> ResultReadyPayload:
        """[S2 PR-4 §3.2] Shell-mode write finalize: capture the PRE/POST
        snapshot diff (quiesce + stability + lease revalidation), BUILD the
        SUCCESS RESULT_READY payload (inline-or-by-ref) and publish it. Capture
        AND build exceptions degrade to NEEDS_AUTHORIZATION
        (_OutOfLeaseWriteError) or FAILED (scan/quiesce/MinIO-upload), exactly
        like _finalize_success — NEVER escaping run_work_unit (the outer wrapper
        is finally-only)."""
        try:
            files = await self._capture_shell_snapshot_diff(run_id, wu)
            patch_manifest = PatchManifest(
                patch_id=f"{run_id}:{wu.work_unit_id}:p",
                coordinator_run_id=run_id,
                work_unit_id=wu.work_unit_id,
                files=tuple(files),
            )
            # [S2 §3.2 C1] BUILD the payload via the PR-2 build-only producer
            # (NOT an inline ResultReadyPayload(patch_manifest=...)): a shell-diff
            # manifest can be large, and publishing it inline would lose it to the
            # 64KB terminal-store truncation → silent zero-apply on rehydrate. The
            # helper returns an inline payload when small, else UPLOADS the
            # manifest to MinIO and returns a payload carrying patch_manifest_ref.
            # It is INSIDE this try because the MinIO upload is a NEW failure
            # source — an upload failure must route to _finalize_failed (a terminal
            # envelope), never escape run_work_unit's finally-only wrapper.
            payload = await self._build_manifest_payload_inline_or_ref(
                run_id, wu, patch_manifest,
                summary=f"completed {wu.work_unit_id}",
            )
        except _OutOfLeaseWriteError as exc:
            return await self._finalize_needs_authorization_out_of_lease(
                run_id, wu, child_id, exc,
            )
        except Exception as exc:  # noqa: BLE001 — scan/quiesce/upload → FAILED
            return await self._finalize_failed(run_id, wu, child_id, exc)
        # [F2 P0] The single publish is the UNPROTECTED final step — OUTSIDE the
        # try above so a publish failure is not swallowed + re-published as FAILED
        # (exactly one publish, never double-handled). The build (incl. upload) is
        # already done and protected; only the publish remains.
        await self._publish_result_ready(child_id, payload)
        return payload

    async def _finalize_needs_authorization_out_of_lease(
        self, run_id: str, wu: "WorkUnit", child_id: str,
        exc: "_OutOfLeaseWriteError",
    ) -> ResultReadyPayload:
        """[F2 P0] A write-phase child wrote a path outside its write_lease
        (defensive lease enforcement caught it during patch extraction).
        Publish a terminal RESULT_READY(NEEDS_AUTHORIZATION) carrying the
        SPECIFIC reject reason (``exc.reason``; default ``out_of_path_lease`` for
        legacy bare raises) — the §3.4 vocabulary the ChildScopeGate would emit
        for the runtime equivalent (see _SCOPE_DECISION_TO_REASON) — plus a
        bounded ``rejection_summary`` (first offending path, §3.4) so the
        zero-apply is diagnosable on the envelope. The reducer/orchestrator
        treats it like any other out-of-lease grievance instead of seeing the
        child crash with no envelope."""
        summary: tuple[str, ...] = (
            (exc.offending_path,) if exc.offending_path else ()
        )
        details = NeedsAuthorizationDetails(
            reason=exc.reason,
            observed_evidence=str(exc),
            rejection_summary=summary[:_SNAPSHOT_SUMMARY_CAP],
        )
        payload = ResultReadyPayload(
            summary=f"out-of-lease write: {wu.work_unit_id}",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=details,
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

    async def _install_seed(self, wu: "WorkUnit") -> None:
        """[finish-core §5.1.4 G1d] Install parent base bytes into the child
        sandbox before the ReAct loop so the child reads real content, and
        verify the installed digest matches the lease's base_digest. op=add
        leases have no seed (seed_content_ref is None by invariant)."""
        if self._child_sandbox is None:
            return  # legacy/test path with no child sandbox
        for lease in wu.write_lease:
            if lease.op not in ("modify", "delete") or lease.seed_content_ref is None:
                continue
            try:
                seed_bytes = await self._artifact_storage.get_bytes(
                    lease.seed_content_ref
                )
                await self._child_sandbox.atomic_write_file(lease.path, seed_bytes)
                observed = await self._child_sandbox.compute_digest(lease.path)
            except Exception as exc:  # noqa: BLE001
                raise _SeedInstallError(
                    f"seed_install_failed: lease={lease.path}: {exc}"
                ) from exc
            # Fail closed: a seeded lease (op modify/delete with a
            # seed_content_ref) MUST carry a base_digest to verify against.
            # Without this guard, ``observed != None`` could silently pass
            # (e.g. compute_digest also returning None), skipping verification.
            # Dispatch always fills base_digest for modify, so valid leases are
            # unaffected — only a malformed lease now fails closed here.
            if lease.base_digest is None:
                raise _SeedInstallError(
                    f"seed_missing_base_digest: lease={lease.path} "
                    "(cannot verify seed)"
                )
            if observed != lease.base_digest:
                raise _SeedInstallError(
                    f"seed_digest_mismatch: lease={lease.path} "
                    f"expected={lease.base_digest} observed={observed}"
                )

    async def _finalize_failed(
        self, run_id: str, wu: "WorkUnit", child_id: str, exc: Exception,
    ) -> ResultReadyPayload:
        # The failing exception was previously swallowed into ``summary`` only,
        # which is NOT persisted to coordinator_result_envelope_store and is
        # consumed off the Redis stream by the subscriber — leaving live
        # worker failures undiagnosable. Log it loudly (with traceback) so the
        # root cause of a FAILED worker is visible in the api logs.
        logger.warning(
            "coordinator child %s (wu=%s, run=%s) FAILED: %s",
            child_id,
            getattr(wu, "work_unit_id", "?"),
            run_id,
            exc,
            exc_info=exc,
        )
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
        # [C2b budget D10 / INV-B9] Best-effort exhaustion counter — single
        # aggregation point for BOTH budget reasons. The try/except wraps the
        # metric emit ONLY: a telemetry failure logs + continues, while a
        # _publish_result_ready failure below propagates untouched (the
        # terminal envelope is load-bearing, the metric is not).
        if self._coordinator_metrics is not None:
            try:
                self._coordinator_metrics.budget_exhaustion.add(
                    1,
                    attributes={
                        "stop_reason": (
                            self._stop_reason.value
                            if self._stop_reason else "unknown"
                        ),
                        "coordinator_run_id": run_id,
                        "work_unit_id": wu.work_unit_id,
                    },
                )
            except Exception:  # noqa: BLE001 — INV-B9 best-effort
                logger.warning(
                    "budget_exhaustion metric emit failed (best-effort; "
                    "envelope publish unaffected)",
                    exc_info=True,
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
            work_unit_id=wu.work_unit_id,
            expected_result_schema=wu.expected_result_schema,
        )

    async def _extract_patch_files_from_history(
        self, run_id: str, wu: "WorkUnit", done_event: Any,
    ) -> list[Any]:
        """[finish-core §5.1.5 G1e] Build FilePatchEntry list from the child's
        typed-write tool calls (ChildRunResult.tool_calls) + final bytes read
        from the child sandbox. Out-of-lease writes are rejected (defensive
        lease enforcement; runtime PE gate is deferred — §5.1.7)."""
        from app.domain.models.patch_manifest import FilePatchEntry

        result = done_event  # ChildRunResult (§5.1.1)
        tool_calls = getattr(result, "tool_calls", ()) or ()
        # Canonicalize lease keys to workspace-relative so the match below is
        # independent of whether the planner leased an absolute or relative
        # path. A lease that escapes the workspace root can never be satisfied
        # by a workspace write — drop it (the write side fails closed instead).
        lease_by_path: dict[str, Any] = {}
        for lease in wu.write_lease:
            try:
                lease_by_path[_to_workspace_relative(lease.path)] = lease
            except _OutOfLeaseWriteError:
                continue

        # Last-write-wins, preserve first-seen order.
        written_paths: list[str] = []
        seen: set[str] = set()
        for ev in tool_calls:
            if getattr(ev, "function_name", None) not in ("file_write", "file_str_replace"):
                continue
            args = getattr(ev, "function_args", {}) or {}
            # [F2 P1] Mirror ChildScopeGate.extract_target_path EXACTLY
            # (value-based ``filepath`` precedence, NOT key-presence): a None
            # ``filepath`` *value* falls back to ``path``, same as the gate's
            # ``path = args.get("filepath"); if path is None: path = args.get("path")``.
            # An empty-string ``filepath`` is NOT None, so it does NOT fall
            # back to ``path`` and is then dropped by the ``not path`` guard
            # below — matching the gate's "" -> missing-target behavior. A
            # key-presence check (``if "filepath" in args``) diverged from the
            # gate for ``{"filepath": None, "path": "x"}``: the gate authorizes
            # the write against "x" while key-presence would skip it entirely.
            path = args.get("filepath")
            if path is None:
                path = args.get("path")
            if not isinstance(path, str) or not path:
                continue
            if path not in seen:
                seen.add(path)
                written_paths.append(path)

        files: list[FilePatchEntry] = []
        for path in written_paths:
            # ``path`` is the raw path the child wrote (absolute or relative);
            # ``canon`` is its workspace-relative form for lease matching + the
            # PatchManifest. An absolute write outside the workspace root raises
            # _OutOfLeaseWriteError here (→ NEEDS_AUTHORIZATION, fail-closed).
            canon = _to_workspace_relative(path)
            lease = lease_by_path.get(canon)
            if lease is None:
                raise _OutOfLeaseWriteError(
                    f"child wrote out-of-lease path: {path!r} "
                    f"(allowed: {sorted(lease_by_path)})"
                )
            if lease.op not in ("add", "modify"):
                # R1 P1: typed writes (file_write/file_str_replace) only ever
                # produce add/modify. A write to a delete-leased path is a
                # contract violation AND FilePatchEntry rejects
                # new_digest/content_ref/content_size for op=delete
                # (patch_manifest.py:106-112).
                raise _OutOfLeaseWriteError(
                    f"child wrote to a non-writable lease (op={lease.op}): {path!r}"
                )
            # Read the bytes back using the RAW path the child wrote (the child
            # sandbox accepts whatever form the child used).
            content = await self._child_sandbox.read_file(path)
            new_digest = hashlib.sha256(content).hexdigest()
            content_ref = await self._artifact_storage.put_content_addressed_bytes(
                prefix=f"coordinator/{run_id}/{wu.work_unit_id}/patch/",
                content=content,
            )
            files.append(FilePatchEntry(
                # Workspace-relative — the PatchApplier re-anchors under the
                # parent sandbox root (validate_relative_path_strict contract).
                path=canon,
                op=lease.op,
                base_digest=lease.base_digest if lease.op == "modify" else None,
                new_digest=new_digest,
                content_ref=content_ref,
                content_size=len(content),
            ))
        return files

    async def _extract_patch_files_from_snapshot(
        self, run_id: str, wu: "WorkUnit",
        pre_scan: "WorkspaceScan", post_scan: "WorkspaceScan",
    ) -> list[Any]:
        """[C2-full S2 §3.2/§3.4] Build FilePatchEntry list from a PRE/POST
        workspace snapshot diff (shell-mode capture). Diff identity tuple =
        (kind, sha256, size, mode, link_target). ADD-only tree leases; exact
        file-lease precedence; base_digest = PRE sha256 for modify/delete;
        parent-side kind-invariant precheck via check_path. ANY violation →
        _OutOfLeaseWriteError (group zero-apply → NEEDS_AUTHORIZATION)."""
        from app.domain.models.patch_manifest import FilePatchEntry
        from app.domain.models.path_validation import (
            CoordinatorPathContractError,
            tree_contains,
            validate_directory_qualified_relative_path,
        )

        # scan.truncated ⇒ fail-CLOSED (cannot trust an aborted walk).
        if pre_scan.truncated or post_scan.truncated:
            raise _snapshot_reject("scan_truncated", "<workspace>")

        # Canonicalize lease keys to workspace-relative for exact match.
        file_lease_by_path: dict[str, Any] = {}
        for lease in wu.write_lease:
            try:
                file_lease_by_path[_to_workspace_relative(lease.path)] = lease
            except _OutOfLeaseWriteError:
                continue

        def _tuple(e: "WorkspaceScanEntry") -> tuple:
            return (e.kind, e.sha256, e.size, e.mode, e.link_target)

        # Determine diff op per rel_path (union of PRE/POST keys).
        all_paths = set(pre_scan.entries) | set(post_scan.entries)
        diffs: list[tuple[str, str, Any, Any]] = []  # (op, rel, pre_entry, post_entry)
        for rel in sorted(all_paths):
            pre_e = pre_scan.entries.get(rel)
            post_e = post_scan.entries.get(rel)
            if pre_e is None and post_e is not None:
                diffs.append(("add", rel, None, post_e))
            elif pre_e is not None and post_e is None:
                diffs.append(("delete", rel, pre_e, None))
            elif pre_e is not None and post_e is not None:
                if _tuple(pre_e) == _tuple(post_e):
                    continue  # F15 content-identical / no-op
                # mode-only change (F11): same sha256+kind+size+link, diff mode.
                if (
                    pre_e.kind == post_e.kind == "regular"
                    and pre_e.sha256 == post_e.sha256
                    and pre_e.size == post_e.size
                    and pre_e.link_target == post_e.link_target
                    and pre_e.mode != post_e.mode
                ):
                    raise _snapshot_reject("mode_only_change", rel)
                # kind change (F12). Distinguish symlink (spec §3.4) from the
                # genuine special-file kinds: a regular→symlink replacement must
                # report `symlink`, NOT `special_file` — check symlink FIRST so
                # the symlink-specific branch is reachable before the special
                # fall-through.
                if pre_e.kind != post_e.kind:
                    if "symlink" in {pre_e.kind, post_e.kind}:
                        raise _snapshot_reject("symlink", rel)
                    # [codex PR-4 R1 P1] An "other" (indeterminate) inode on
                    # either side reports the §3.4 `indeterminate_kind` code, NOT
                    # the special-file fall-through — match the build loop's
                    # per-entry vocabulary so the wire reason is correct.
                    if "other" in {pre_e.kind, post_e.kind}:
                        raise _snapshot_reject("indeterminate_kind", rel)
                    raise _snapshot_reject("special_file", rel)
                diffs.append(("modify", rel, pre_e, post_e))
            # both None impossible (rel came from the union).

        files: list[Any] = []
        for op, rel, pre_e, post_e in diffs:
            # Non-regular fail-closed (F8 symlink / F9 special / F10 other).
            for e in (pre_e, post_e):
                if e is None:
                    continue
                if e.kind == "symlink":
                    raise _snapshot_reject("symlink", rel)
                if e.kind in ("fifo", "socket", "block", "char"):
                    raise _snapshot_reject("special_file", rel)
                if e.kind == "other":
                    raise _snapshot_reject("indeterminate_kind", rel)
                if e.kind != "regular":
                    raise _snapshot_reject("indeterminate_kind", rel)

            # Path validation (F18 bare top-level).
            try:
                canon = validate_directory_qualified_relative_path(
                    _to_workspace_relative(rel)
                )
            except (ValueError, CoordinatorPathContractError):
                raise _snapshot_reject("out_of_path_lease", rel)

            # Lease matching with precedence: exact file lease wins.
            exact = file_lease_by_path.get(canon)
            if exact is not None:
                if not _op_lease_compatible(op, exact.op):
                    raise _snapshot_reject("out_of_path_lease", canon)
                governing_op = exact.op
            else:
                # Tree lease only covers op=add.
                covered = any(
                    tree_contains(_canon_tree_prefix(tl.prefix), canon)
                    and "add" in tl.ops
                    for tl in wu.write_tree_lease
                )
                if op == "add" and covered:
                    governing_op = "add"
                elif op in ("modify", "delete") and any(
                    tree_contains(_canon_tree_prefix(tl.prefix), canon)
                    for tl in wu.write_tree_lease
                ):
                    raise _snapshot_reject("out_of_tree_lease", canon)
                else:
                    raise _snapshot_reject("out_of_path_lease", canon)

            # Parent-side kind invariant (F22/F25/F26) via check_path (NOT exists).
            check = await self._parent_sandbox.check_path(canon)
            if governing_op == "add":
                if check.kind == "regular":
                    raise _snapshot_reject("tree_add_target_exists", canon)
                if check.kind != "missing":
                    raise _snapshot_reject("parent_not_regular", canon)
            else:  # modify / delete
                if check.kind != "regular":
                    raise _snapshot_reject("parent_not_regular", canon)

            # Build the entry.
            if governing_op == "delete":
                files.append(FilePatchEntry(
                    path=canon, op="delete", base_digest=pre_e.sha256,
                ))
                continue
            content = await self._child_sandbox.read_file(rel)
            new_digest = hashlib.sha256(content).hexdigest()
            # [codex PR-4 R1 P1] Bind the captured bytes to the proven-stable POST
            # scan. quiesce + the double-scan stability check (Task 4.6) prove the
            # workspace was quiescent AT SCAN TIME, but a detached writer not
            # reaped by quiesce could rewrite a leased file in the window BETWEEN
            # the second POST scan and this read — slipping bytes into the manifest
            # that no stable scan ever witnessed. The scan's sha256/size is that
            # witness; a mismatch means the workspace was NOT quiescent → raise
            # RuntimeError (→ _finalize_failed → FAILED), never a silent capture.
            if post_e is not None and (
                new_digest != post_e.sha256 or len(content) != post_e.size
            ):
                raise RuntimeError(
                    "workspace not quiescent: "
                    f"{rel!r} changed between the stable POST scan and the "
                    f"content read (scan={post_e.sha256}/{post_e.size}, "
                    f"read={new_digest}/{len(content)})"
                )
            content_ref = await self._artifact_storage.put_content_addressed_bytes(
                prefix=f"coordinator/{run_id}/{wu.work_unit_id}/patch/",
                content=content,
            )
            files.append(FilePatchEntry(
                path=canon,
                op=governing_op,
                base_digest=pre_e.sha256 if governing_op == "modify" else None,
                new_digest=new_digest,
                content_ref=content_ref,
                content_size=len(content),
            ))
        return files

    async def _capture_shell_snapshot_diff(
        self, run_id: str, wu: "WorkUnit",
    ) -> list[Any]:
        """[C2-full S2 §3.2] Quiesce shell, take the POST scan + a stability
        re-scan, confirm semantic equality, then diff PRE vs POST. Bounded by
        max_snapshot_seconds. Raises _OutOfLeaseWriteError (lease/cap → NEEDS_AUTH)
        or RuntimeError/Exception (quiesce/scan failure → FAILED). Must run inside
        the finalize try/except so an escape still publishes a terminal envelope."""
        limits = self._snapshot_limits
        scan_kwargs = dict(
            max_paths=limits.max_snapshot_paths,
            max_files=limits.max_snapshot_files,
            max_total_bytes=limits.max_snapshot_total_bytes,
            max_seconds=limits.max_snapshot_seconds,
        )

        async def _bounded() -> list[Any]:
            # (a) quiesce: kill every tracked shell process-group.
            await self._child_sandbox.kill_all_shell_sessions()
            # (b) POST scan + (c) stability re-scan.
            post_a = await self._child_sandbox.snapshot_workspace(**scan_kwargs)
            post_b = await self._child_sandbox.snapshot_workspace(**scan_kwargs)
            if not _scans_semantically_equal(post_a, post_b):
                raise RuntimeError(
                    "workspace not quiescent: live writer detected on re-scan"
                )
            pre = self._pre_scan
            if pre is None:
                raise RuntimeError("PRE scan missing; cannot diff shell workspace")
            return await self._extract_patch_files_from_snapshot(
                run_id, wu, pre, post_a,
            )

        return await asyncio.wait_for(
            _bounded(), timeout=limits.max_snapshot_seconds,
        )

    def _extract_proposed_write_plan(self, done_event: Any) -> ProposedWritePlan:
        """[PR-4 minimal] Returns an empty proposal. PR-6 wires the real
        extraction (parse the LLM's final assistant message into a structured
        proposed_write_plan, optionally with a MinIO rationale_ref)."""
        return ProposedWritePlan(
            proposed_paths=(),
            proposed_tools=frozenset(),
        )

    def _build_budget_evidence(self) -> str:
        """[C2b budget D5] Three-state contract (spec D5 / R7#1):

        - ``budget is None`` (legacy/unit constructions): EXACT old format
          ``stop_reason=<v>`` — INV-B6 clause (iv).
        - ``budget`` set, ``_budget_callback is None`` (unpriced fail-soft —
          wallclock-only enforcement): caps + elapsed emitted,
          ``token_cost_usd_observed`` OMITTED.
        - ``budget`` + callback attached: full fields including the observed
          cumulative USD (read from the callback's read-only property).

        Free-form space-separated ``key=value`` string — the wire field
        (NeedsAuthorizationDetails.observed_evidence) is a plain str.
        """
        reason_str = self._stop_reason.value if self._stop_reason else "unknown"
        if self._budget is None:
            return f"stop_reason={reason_str}"
        parts = [f"stop_reason={reason_str}"]
        if self._budget_callback is not None:
            parts.append(
                f"token_cost_usd_observed="
                f"{float(self._budget_callback.cumulative_usd):.6f}"
            )
        parts.append(f"token_cap_usd={float(self._budget.max_token_cost_usd):.6f}")
        parts.append(
            f"wallclock_cap_seconds={self._budget.max_wallclock_seconds}"
        )
        if self._inner_invoke_started_monotonic is not None:
            elapsed = time.monotonic() - self._inner_invoke_started_monotonic
            parts.append(f"wallclock_elapsed_seconds={elapsed:.3f}")
        return " ".join(parts)
