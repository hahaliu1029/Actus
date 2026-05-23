"""Phase 1 minimal subagent research orchestration.

Canonical try/finally guarantees:
1. probe quota slot acquired before any work
2. children created serially (UoW race fix)
3. child events streamed via asyncio.as_completed (incremental yield)
4. summary join validated deterministically with 1× retry on fail
5. finally: sandbox suspend per child + quota release + jsonl metric

CancelledError propagates → all running children cancelled via
ExecutionSupervisor.request_cancel before re-raise (parent SSE
disconnect path).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from langchain_core.language_models import BaseChatModel

from app.application.services.agent_service import AgentService
from app.application.services.session_service import SessionService
from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.external.supervisor_registry import SupervisorRegistryPort
from app.domain.models.event import BaseEvent
from app.domain.models.session import Session
from app.domain.services.execution_supervisor import ExecutionSupervisor
from app.domain.services.mailbox_skip_helper import _should_skip_mailbox_lifecycle
from app.domain.services.graphs.token_estimator import TokenEstimator
from app.domain.services.prompts.subagent_summary_join import (
    build_summary_prompt,
    validate_joined_summary,
)
from app.domain.services.subagent_research_classifier import (
    SubagentResearchClassifier,
)
from app.domain.services.tool_filter_presets import (
    SUBAGENT_RESEARCH_ALLOWED_TOOLS,
)
from app.infrastructure.cache.probe_quota import ProbeQuotaService
from app.interfaces.schemas.subagent import (
    ChildDoneEvent,
    ChildOutcome,
    ChildStartedEvent,
    DroppedChild,
    JoinedSummaryEvent,
)

logger = logging.getLogger(__name__)


# T12: ``Session.tool_filter_preset`` value persisted on every child created
# by ``run_research``. ``AgentService._create_task`` reads it on resume /
# FINISHING / orphan paths and re-derives the allowlist via ``resolve_preset``
# so a pod restart doesn't silently drop the tool restriction (F8).
_SUBAGENT_RESEARCH_PRESET = "subagent_research"


METRIC_LOG_PATH = Path.home() / ".gstack" / "metrics" / "actus-multiagent-probe.jsonl"


@dataclass(frozen=True)
class ChildResult:
    """Aggregate outcome per child session for join + metric."""
    child_id: str
    prompt: str
    outcome: ChildOutcome
    final_answer: Optional[str]
    transcript_tokens: int
    error_summary: Optional[str]


class SubagentResearchService:
    """Orchestrates a research probe: 1-3 children, summary join, metric."""

    def __init__(
        self,
        session_service: SessionService,
        agent_service: AgentService,
        execution_supervisor: ExecutionSupervisor,
        token_estimator: TokenEstimator,
        summary_llm: BaseChatModel,
        classifier: SubagentResearchClassifier,
        sandbox_lifecycle_service: Any,
        quota_service: ProbeQuotaService,
        *,
        supervisor_registry: Optional[SupervisorRegistryPort] = None,
        mailbox_publisher: Optional[MailboxPublisher] = None,
    ) -> None:
        self._session_service = session_service
        self._agent_service = agent_service
        self._supervisor = execution_supervisor
        self._token_estimator = token_estimator
        self._summary_llm = summary_llm
        self._classifier = classifier
        self._sandbox_lifecycle_service = sandbox_lifecycle_service
        self._quota_service = quota_service
        # codex r3 [R3-3, HIGH ARCH] — set of child IDs whose
        # SPAWN_REQUEST publish actually succeeded during the most
        # recent ``_ensure_supervisor_and_publish_spawns`` invocation.
        # ``run_research`` resets this per invocation; the finally-block
        # consults it before skipping legacy suspend.
        self._handoff_published_child_ids: set[str] = set()
        # codex r11 [R11-2, HIGH ARCH] — children whose rollback to
        # ``legacy`` FAILED. Caller must skip ``_consume_child`` /
        # task creation for these to avoid the parallel-writer race
        # (DB still says mailbox so runner publishes terminal
        # envelopes, but parent has no supervisor handoff and would
        # otherwise legacy-suspend in parallel).
        self._failed_rollback_child_ids: set[str] = set()
        # C3 PR-4.5 — mailbox-plane wiring (None when flag is off or for
        # tests that bypass DI). ``run_research`` uses ``supervisor_registry``
        # to ensure the per-root supervisor exists before publishing the
        # first SPAWN_REQUEST (spec §11.2 R1 P0.2 race fix) and
        # ``mailbox_publisher`` to actually XADD the envelopes.
        self._supervisor_registry = supervisor_registry
        self._mailbox_publisher = mailbox_publisher

    async def _consume_child(
        self, child_session_id: str, user_id: str, prompt: str,
    ) -> ChildResult:
        """Consume a single child's chat() stream → ChildResult.

        Reads events until DoneEvent or ErrorEvent. Extracts final assistant
        message + computes transcript token count. Defensive check on session
        status for WAITING/TIMED_OUT terminal states.
        """
        from app.domain.models.event import (
            DoneEvent,
            ErrorEvent,
            MessageEvent,
        )

        transcript_text_parts: list[str] = []
        final_answer: Optional[str] = None

        try:
            async for event in self._agent_service.chat(
                session_id=child_session_id,
                user_id=user_id,
                is_admin=False,
                message=prompt,
                tool_filter=SUBAGENT_RESEARCH_ALLOWED_TOOLS,
            ):
                if hasattr(event, "message") and event.message:
                    transcript_text_parts.append(str(event.message))
                elif hasattr(event, "content"):
                    transcript_text_parts.append(str(event.content))

                if isinstance(event, MessageEvent) and getattr(event, "role", None) == "assistant":
                    final_answer = event.message

                if isinstance(event, ErrorEvent):
                    error_summary = getattr(event, "error", str(event))
                    return ChildResult(
                        child_id=child_session_id,
                        prompt=prompt,
                        outcome=ChildOutcome.FAILED,
                        final_answer=None,
                        transcript_tokens=self._estimate_tokens(transcript_text_parts),
                        error_summary=error_summary,
                    )

                if isinstance(event, DoneEvent):
                    break

            try:
                session = await self._session_service.get_session(
                    child_session_id, user_id, is_admin=False
                )
                outcome = self._map_session_status_to_outcome(session)
            except Exception:
                outcome = ChildOutcome.COMPLETED

            return ChildResult(
                child_id=child_session_id,
                prompt=prompt,
                outcome=outcome,
                final_answer=final_answer,
                transcript_tokens=self._estimate_tokens(transcript_text_parts),
                error_summary=None,
            )

        except asyncio.CancelledError:
            return ChildResult(
                child_id=child_session_id,
                prompt=prompt,
                outcome=ChildOutcome.CANCELLED,
                final_answer=None,
                transcript_tokens=self._estimate_tokens(transcript_text_parts),
                error_summary="parent disconnect or explicit cancel",
            )
        except Exception as e:
            logger.warning("child %s consume failed: %s", child_session_id, e)
            return ChildResult(
                child_id=child_session_id,
                prompt=prompt,
                outcome=ChildOutcome.FAILED,
                final_answer=None,
                transcript_tokens=self._estimate_tokens(transcript_text_parts),
                error_summary=str(e),
            )

    def _estimate_tokens(self, text_parts: list[str]) -> int:
        full = "\n".join(text_parts)
        try:
            return int(self._token_estimator.count(full))
        except Exception:
            return 0

    def _map_session_status_to_outcome(self, session: Session) -> ChildOutcome:
        """Map terminal session.status → ChildOutcome enum value."""
        status_str = getattr(session.status, "value", str(session.status)).lower()
        if "waiting" in status_str:
            return ChildOutcome.WAITING
        if "timed_out" in status_str or "timeout" in status_str:
            return ChildOutcome.TIMED_OUT
        if "failed" in status_str or "error" in status_str:
            return ChildOutcome.FAILED
        return ChildOutcome.COMPLETED

    # codex r2 [R2-6, MEDIUM CONTRACT] — applied to ``task_prompt`` before
    # it leaves the application layer for the Redis Stream. Caps payload
    # size for redaction-friendly audit and keeps Redis Stream entries
    # bounded (the stream is XADD-capped at ``maxlen``; individual
    # entries should not balloon either).
    _SPAWN_REQUEST_PROMPT_MAX_BYTES: int = 1024

    @classmethod
    def _cap_task_prompt(cls, prompt: str) -> str:
        """codex r6 [R6-4, MEDIUM CONTRACT] — reserve the ellipsis
        suffix's byte budget BEFORE truncation so the final wire-size
        respects ``_SPAWN_REQUEST_PROMPT_MAX_BYTES``. Earlier rounds
        appended ``"…"`` (3 bytes UTF-8) after a hard truncate to
        ``max_bytes``, producing entries up to ``max_bytes + 3``.

        codex r17 [R17-2, MEDIUM SEC] — note on redaction: the prompt
        flows from the same user request that creates the child
        session row (which ALSO stores the prompt for the runner to
        consume via ``agent_service.chat``). The Redis Stream is
        per-root and accessed only by the supervisor consumer
        (same trust boundary as the session DB). PR-4.5 carries the
        capped prompt for audit/traceability without additional
        redaction; a dedicated redaction policy (PII scrubber etc.)
        is deferred to the audit-policy work tracked outside C3.
        """
        encoded = (prompt or "").encode("utf-8")
        if len(encoded) <= cls._SPAWN_REQUEST_PROMPT_MAX_BYTES:
            return prompt
        ellipsis = "…"
        budget = cls._SPAWN_REQUEST_PROMPT_MAX_BYTES - len(ellipsis.encode("utf-8"))
        # Truncate on byte boundary then re-decode tolerantly so we
        # don't split a multi-byte char in the middle.
        return encoded[:budget].decode("utf-8", errors="ignore") + ellipsis

    async def _rollback_child_to_legacy(
        self,
        child_id: str,
        *,
        expected_parent_id: str | None = None,
    ) -> bool:
        """codex r6 [R6-1, CRITICAL ARCH] / r11 [R11-2] / r12 [R12-1] /
        r20 [R20-3, MEDIUM SEC] — rollback of a child session row to
        ``subagent_control_plane='legacy'`` after a SPAWN_REQUEST
        publish (or spawn) failure.

        Returns ``True`` if the DB UPDATE committed, ``False``
        otherwise. Callers MUST treat a ``False`` return as "do not
        start this child runner": the stale ``mailbox`` row would
        otherwise let the runner publish SPAWN_ACK/RESULT_READY in
        parallel with the parent's legacy suspend → re-opens the M1
        race.

        codex r20 [R20-3, MEDIUM SEC] / r22 [R22-2, HIGH ARCH] —
        when ``expected_parent_id`` is provided, the UPDATE is
        scoped to rows whose ``parent_session_id`` matches; a
        cross-root row (upstream bug routing a foreign child into
        ``children``) is therefore not silently rewritten. The
        WHERE-failure path (rowcount 0) returns ``False`` so the
        caller marks the child do-not-start — a foreign-root or
        deleted row should never be started under any plane.

        Bypasses the SessionRepository protocol because this is an
        operational rollback narrower than the normal lifecycle
        methods; uses the SessionService's uow factory so the
        connection comes from the same pool as the create path.
        """
        try:
            from sqlalchemy import update
            from app.infrastructure.models.session import SessionModel
            uow_factory = self._session_service._uow_factory
            async with uow_factory() as uow:
                stmt = update(SessionModel).where(SessionModel.id == child_id)
                if expected_parent_id is not None:
                    stmt = stmt.where(
                        SessionModel.parent_session_id == expected_parent_id
                    )
                stmt = stmt.values(subagent_control_plane="legacy")
                result = await uow.db_session.execute(stmt)
                await uow.db_session.commit()
            # codex r22 [R22-2, HIGH ARCH] — distinguish "rolled back"
            # (rowcount ≥ 1) from "no row matched" (rowcount 0, e.g.
            # cross-root parent_id mismatch). Returning False on
            # rowcount 0 forces the caller to mark the child
            # do-not-start, which is the correct behavior: a row that
            # didn't match our parent guard is either a foreign-root
            # orphan or a deleted row, and we shouldn't start its
            # runner under any plane.
            rowcount = getattr(result, "rowcount", None)
            if rowcount == 0:
                logger.warning(
                    "rollback child=%s matched 0 rows (likely cross-root or "
                    "deleted) — treating as failed rollback to force caller "
                    "to skip starting this child runner",
                    child_id,
                )
                return False
            return True
        except Exception:
            logger.error(
                "rollback child=%s to control_plane=legacy failed — "
                "caller MUST skip starting this child runner to avoid "
                "the parallel-writer race (operator: re-run the UPDATE "
                "manually OR delete the orphan row + container)",
                child_id,
                exc_info=True,
            )
            return False

    async def _ensure_supervisor_and_publish_spawns(
        self,
        *,
        parent_id: str,
        children_with_prompts: list[tuple[Any, str]],
    ) -> None:
        """C3 PR-4.5 — ensure supervisor + publish SPAWN_REQUEST for each
        mailbox-plane child (spec §11.2 + R1 P0.2 race fix).

        Best-effort: a registry/publisher failure is logged and SWALLOWED.
        Pre-PR-5 the flag is off so this whole path is a no-op (every
        child has ``subagent_control_plane='legacy'``). Post-PR-5, if
        publish fails the SubagentResearchService's existing legacy
        fallback handles the child (the suspend at line ~466).
        """
        registry = self._supervisor_registry
        publisher = self._mailbox_publisher
        if registry is None and publisher is None:
            return
        mailbox_children = [
            (c, p) for c, p in children_with_prompts
            if getattr(c, "subagent_control_plane", None) == "mailbox"
        ]
        if not mailbox_children:
            return

        # codex r3 [R3-3, HIGH ARCH] / r11 [R11-2] — reset the
        # per-invocation handoff sets so the finally-block sees only
        # what happened THIS run.
        self._handoff_published_child_ids = set()
        self._failed_rollback_child_ids = set()

        # codex r1 [R1-7, MEDIUM SEC] — defense in depth: only publish
        # for children whose ``parent_session_id`` actually equals
        # ``parent_id`` AND whose ``worker_type == 'subagent'``. The
        # supervisor trusts publisher-side child→root mapping; a
        # programming mistake elsewhere that hands us a foreign child
        # would otherwise let us publish destroy authority to the wrong
        # root's stream. Drop any mismatch with a warning so the bug is
        # visible without crashing the request.
        verified_children: list[tuple[Any, str]] = []
        for c, p in mailbox_children:
            if (
                getattr(c, "worker_type", None) == "subagent"
                and getattr(c, "parent_session_id", None) == parent_id
            ):
                verified_children.append((c, p))
            else:
                logger.warning(
                    "subagent_research: refusing to publish SPAWN_REQUEST for "
                    "child=%s — worker_type=%s parent_session_id=%s (expected "
                    "parent=%s, worker_type=subagent). Likely upstream bug.",
                    getattr(c, "id", "?"),
                    getattr(c, "worker_type", "?"),
                    getattr(c, "parent_session_id", "?"),
                    parent_id,
                )
                # codex r19 [R19-4, MEDIUM SEC] — verified-out
                # children still have ``subagent_control_plane='mailbox'``
                # on their DB row, so their runner would publish into
                # the wrong root's stream. Roll them back to legacy
                # so the runner reads legacy; if rollback fails,
                # mark do-not-start so the parent skips task creation.
                # codex r20 [R20-3, MEDIUM SEC] — scope the rollback
                # SQL by ``expected_parent_id`` so a foreign-root
                # child (upstream routing bug) isn't silently
                # rewritten. We deliberately use ``parent_id`` (the
                # request's parent) not ``c.parent_session_id`` so a
                # mismatch results in a no-op UPDATE (rowcount 0)
                # rather than touching the wrong root's row.
                if not await self._rollback_child_to_legacy(
                    c.id, expected_parent_id=parent_id
                ):
                    self._failed_rollback_child_ids.add(c.id)
        if not verified_children:
            return
        mailbox_children = verified_children

        # 1. Ensure the supervisor for this root exists. ``spawn`` is
        # idempotent — it's a no-op when the slot already holds an alive
        # supervisor.
        #
        # codex r4 [R4-1, HIGH ARCH] — if spawn fails AND we cannot
        # otherwise prove a supervisor exists, we MUST fall back to
        # legacy suspend in the finally block (do NOT mark handoff
        # successful). Track spawn outcome and require both spawn-ok +
        # publish-ok before adding to ``_handoff_published_child_ids``.
        supervisor_confirmed = False
        if registry is not None:
            try:
                await registry.spawn(parent_id)
                supervisor_confirmed = True
            except Exception:
                logger.exception(
                    "supervisor_registry.spawn(%s) failed in subagent research "
                    "— skipping publish and falling back to legacy suspend "
                    "for the affected children",
                    parent_id,
                )

        if publisher is None or not supervisor_confirmed:
            # codex r7 [R7-1, HIGH ARCH] / r11 [R11-2] — no consumer /
            # no publisher means the supervisor will never see these
            # children. The child rows are still
            # ``subagent_control_plane='mailbox'`` though, so the
            # child runner would read mailbox plane and publish
            # SPAWN_ACK/RESULT_READY anyway — re-opening the M1 race
            # the finally fallback is supposed to close. Roll each
            # affected child to legacy; if rollback fails, mark the
            # child as do-not-start so the caller skips task creation.
            for c, _ in mailbox_children:
                rolled_back = await self._rollback_child_to_legacy(
                    c.id, expected_parent_id=parent_id
                )
                if not rolled_back:
                    self._failed_rollback_child_ids.add(c.id)
            return

        # 2. Publish a SPAWN_REQUEST per mailbox-plane child.
        from datetime import datetime, timezone
        from app.domain.models.mailbox_envelope import (
            MailboxEnvelope,
            MailboxEnvelopeType,
            ProducerRole,
            SpawnRequestPayload,
        )
        now = datetime.now(tz=timezone.utc)
        for child, prompt in mailbox_children:
            try:
                await publisher.publish(MailboxEnvelope(
                    envelope_id=f"spawn-req:{child.id}",
                    type=MailboxEnvelopeType.SPAWN_REQUEST,
                    parent_session_id=parent_id,
                    child_session_id=child.id,
                    correlation_id=f"spawn:{child.id}",
                    emitted_at=now,
                    producer_role=ProducerRole.EXTERNAL_PUBLISHER,
                    payload=SpawnRequestPayload(
                        agent_kind="research",
                        # codex r2 [R2-6] — populate task_prompt with the
                        # actual prompt capped to a fixed byte budget so
                        # the wire payload is honest without ballooning
                        # the Redis Stream entry size.
                        task_prompt=self._cap_task_prompt(prompt),
                    ).model_dump(mode="json"),
                ))
                # codex r3 [R3-3, HIGH ARCH] — record successful publish
                # so the caller can decide whether to fall back to legacy
                # suspend for children whose SPAWN_REQUEST failed.
                self._handoff_published_child_ids.add(child.id)
            except Exception:
                logger.exception(
                    "SPAWN_REQUEST publish failed for child %s — rolling "
                    "back DB row to control_plane=legacy and falling back "
                    "to legacy suspend",
                    child.id,
                )
                # codex r6 [R6-1, CRITICAL ARCH] / r11 [R11-2] —
                # persist ``subagent_control_plane='legacy'`` on the
                # child row NOW so the child's AgentTaskRunner reads
                # ``legacy`` before invoke()'s spawn check fires.
                # Otherwise the runner sees stale ``mailbox`` plane,
                # publishes SPAWN_ACK/RESULT_READY into a stream with
                # no consumer, and the parent's finally fallback runs
                # legacy suspend in parallel → re-opens the M1
                # single-writer race. If rollback fails, mark the
                # child as do-not-start.
                rolled_back = await self._rollback_child_to_legacy(
                    child.id, expected_parent_id=parent_id
                )
                if not rolled_back:
                    self._failed_rollback_child_ids.add(child.id)

    async def _cancel_pending_children(
        self,
        tasks: list[asyncio.Task[ChildResult]],
        children: list[tuple[Any, str]],
        user_id: str,
    ) -> None:
        """Cancel all still-running child tasks and notify supervisor.

        Called from BOTH the inner `except CancelledError` (outer task-level
        cancel) AND `finally` (GeneratorExit from aclose). All awaits are
        shielded so cleanup completes even under cancel storms; tasks are then
        drained via gather so we don't leak pending tasks past return.
        """
        pending: list[tuple[asyncio.Task[ChildResult], Any]] = []
        for t, (child, _) in zip(tasks, children):
            if not t.done():
                t.cancel()
                pending.append((t, child))

        for _, child in pending:
            try:
                await asyncio.shield(
                    self._supervisor.request_cancel(
                        session_id=child.id,
                        user_id=user_id,
                        reason="probe_parent_disconnected",
                        stop_session=self._agent_service.stop_session,
                    )
                )
            except asyncio.CancelledError:
                # Outer await re-cancelled; inner request_cancel still running
                # in background per asyncio.shield semantics. Continue to next.
                logger.warning(
                    "child %s cancel re-entered while shielded; inner "
                    "request_cancel still running in background",
                    child.id,
                )
            except Exception as e:
                logger.warning(
                    "child %s cancel failed: %s", child.id, e
                )

        if pending:
            # Drain cancelled tasks so they don't outlive this coroutine.
            # return_exceptions=True swallows CancelledError + any inner
            # exceptions; we just want them done before quota/sandbox release.
            try:
                await asyncio.shield(
                    asyncio.gather(
                        *(t for t, _ in pending), return_exceptions=True
                    )
                )
            except asyncio.CancelledError:
                logger.warning(
                    "drain of cancelled children re-entered while shielded"
                )

    async def _do_summary_join_with_retry(
        self,
        prompts: list[str],
        completed: list[ChildResult],
        dropped: list[dict],
        max_retries: int = 1,
    ) -> tuple[str, list[str]]:
        """Invoke summary_llm + validate. On fail, 1× retry with error feedback.

        Returns (summary_text, validation_warnings). warnings empty on success.
        """
        base_prompt = build_summary_prompt(prompts, completed, dropped)
        summary = ""
        last_errors: list[str] = []

        for attempt in range(max_retries + 1):
            try:
                response = await self._summary_llm.ainvoke(base_prompt)
                summary = getattr(response, "content", str(response))
            except Exception as e:
                logger.warning("summary_llm error on attempt %d: %s", attempt, e)
                return ("(summary unavailable due to LLM error)", [str(e)])

            ok, errors = validate_joined_summary(summary, completed, dropped)
            if ok:
                return (summary, [])

            last_errors = errors
            if attempt < max_retries:
                base_prompt = (
                    f"{base_prompt}\n\n# Previous attempt failed validation:\n"
                    + "\n".join(f"- {e}" for e in errors)
                    + "\n# Fix the above and retry:\n"
                )

        return (summary, last_errors)

    async def run_research(
        self,
        user_id: str,
        prompts: list[str],
        *,
        parent_session_id: str,
        max_children: int = 3,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run a research probe. See module docstring for invariants.

        Yields ChildStartedEvent × N → ChildDoneEvent × N (interleaved by
        as_completed order) → JoinedSummaryEvent (final).
        """
        from app.application.errors.exceptions import (
            BadRequestError, ConflictError, NotFoundError,
        )

        parent_id = parent_session_id
        if parent_id is None:
            raise ValueError("parent_session_id is required")

        probe_run_id = uuid.uuid4().hex
        start_ts = time.time()

        parent = await self._session_service.get_session(
            parent_id, user_id, is_admin=False
        )
        if parent is None:
            raise NotFoundError(
                f"Session {parent_id} not found or not accessible"
            )

        classifier_results = await self._classifier.classify_batch(prompts)
        rejected = [
            (i, r) for i, r in enumerate(classifier_results) if not r.approved
        ]
        if rejected:
            msg = "; ".join(f"prompt {i+1}: {r.reason}" for i, r in rejected)
            raise BadRequestError(f"Preflight rejected: {msg}")

        if not await self._quota_service.acquire(user_id, probe_run_id):
            raise ConflictError(
                f"Active probe quota exceeded for user {user_id}"
            )

        children: list[tuple[Any, str]] = []
        completed_results: list[ChildResult] = []
        # Pre-declare so finally can distinguish "never reached metric compute"
        # (None) from "reached compute but happened to be 0 / empty" (0 / {}).
        summary_tokens: int | None = None
        metrics: dict | None = None
        # Pre-declare so finally can cancel still-running children even when
        # the generator is closed via aclose() (which raises GeneratorExit at
        # the yield point, NOT CancelledError — the inner except branch
        # therefore never fires on client disconnect).
        tasks: list[asyncio.Task[ChildResult]] = []
        # codex r13 [R13-1, HIGH ARCH] — pre-declare so the cancel paths
        # can reference the filtered list (which is what ``tasks`` was
        # actually created from). Passing the unfiltered ``children`` to
        # ``_cancel_pending_children`` paired wrong indices via zip when
        # any rollback-failed child was skipped.
        startable_children: list[tuple[Any, str]] = []

        try:
            for prompt in prompts[:max_children]:
                child = await self._session_service.create_session_with_parent(
                    user_id=user_id,
                    parent_session_id=parent_id,
                    tool_filter_preset=_SUBAGENT_RESEARCH_PRESET,
                )
                # NOTE: spec calls `session_service.update_title(...)` here for
                # UX (prefixing child titles with "[probe]"), but that method
                # does not exist on SessionService. Skipping — title is
                # cosmetic; correctness does not depend on it.
                children.append((child, prompt))

            # C3 PR-4.5 — R1 P0.2 fix: ensure the per-root MailboxSupervisor
            # exists BEFORE publishing the first SPAWN_REQUEST. The supervisor
            # is the stream consumer; publishing into an empty group would
            # leak envelopes until pod-restart reconcile sweeps them.
            # ``SupervisorRegistry.spawn`` is idempotent (no-op when slot
            # exists), so this is safe to call on every research run.
            #
            # codex r2 [R2-6, MEDIUM CONTRACT] — thread the per-child
            # prompt through so SPAWN_REQUEST.task_prompt is honest (was
            # empty in r1, contradicting the wire schema). The publisher
            # applies a length cap for safety.
            await self._ensure_supervisor_and_publish_spawns(
                parent_id=parent_id,
                children_with_prompts=children,
            )

            # codex r11 [R11-2, HIGH ARCH] — filter out children whose
            # mailbox→legacy rollback FAILED. Starting their runner
            # would race the parent's finally-suspend (DB still says
            # mailbox so runner would publish terminal envelopes
            # against a non-existent supervisor handoff). Yield a
            # ChildDoneEvent with the failure outcome so the caller
            # sees the dropped child and the metric is accurate.
            for child, prompt in children:
                if child.id in self._failed_rollback_child_ids:
                    logger.error(
                        "subagent_research: skipping start of child=%s — "
                        "control_plane rollback failed; child runner not "
                        "started to prevent the parallel-writer race",
                        child.id,
                    )
                    completed_results.append(
                        ChildResult(
                            child_id=child.id,
                            prompt=prompt,
                            outcome=ChildOutcome.FAILED,
                            final_answer=None,
                            transcript_tokens=0,
                            error_summary=(
                                "internal: mailbox→legacy rollback failed; "
                                "child runner skipped to avoid race"
                            ),
                        )
                    )
                    yield ChildDoneEvent(
                        id=f"done-{child.id}",
                        probe_run_id=probe_run_id,
                        child_session_id=child.id,
                        outcome=ChildOutcome.FAILED,
                        final_answer=None,
                        transcript_tokens=0,
                        error_summary=(
                            "internal: mailbox→legacy rollback failed; "
                            "child runner skipped"
                        ),
                    )
                    continue
                startable_children.append((child, prompt))

            for child, prompt in startable_children:
                yield ChildStartedEvent(
                    id=f"started-{child.id}",
                    probe_run_id=probe_run_id,
                    child_session_id=child.id,
                    prompt=prompt,
                )

            async def _wrapped(child_id: str, prompt: str) -> ChildResult:
                return await self._consume_child(child_id, user_id, prompt)

            tasks.extend(
                asyncio.create_task(_wrapped(child.id, prompt))
                for child, prompt in startable_children
            )
            # codex r25 [R25-1, HIGH ARCH] — prune the handoff set so
            # it only contains children whose runner TASK actually
            # exists. ``_ensure_supervisor_and_publish_spawns`` adds
            # to ``_handoff_published_child_ids`` immediately on
            # publish success, but the window between publish and
            # task creation can be interrupted (rollback failure,
            # filter logic, generator cancel mid-loop). If a child
            # is in the handoff set but has no created task, the
            # finally would skip suspend while no runner exists to
            # fire RESULT_READY → orphan sandbox. Restrict to
            # ``startable_children`` ids so the finally only honors
            # handoff for children with a real task.
            _started_child_ids = {child.id for child, _ in startable_children}
            self._handoff_published_child_ids &= _started_child_ids

            try:
                for done_task in asyncio.as_completed(tasks):
                    result = await done_task
                    completed_results.append(result)
                    yield ChildDoneEvent(
                        id=f"done-{result.child_id}",
                        probe_run_id=probe_run_id,
                        child_session_id=result.child_id,
                        outcome=result.outcome,
                        final_answer=result.final_answer,
                        transcript_tokens=result.transcript_tokens,
                        error_summary=result.error_summary,
                    )
            except asyncio.CancelledError:
                # Outer cancel of the generator's underlying task. Cancel
                # pending children, then propagate.
                # codex r13 [R13-1] — pass startable_children so the
                # zip with ``tasks`` pairs the right (task, child) for
                # each running runner.
                await self._cancel_pending_children(
                    tasks=tasks, children=startable_children, user_id=user_id,
                )
                raise

            completed = [
                r for r in completed_results
                if r.outcome == ChildOutcome.COMPLETED
            ]
            dropped = [
                DroppedChild(
                    child_id=r.child_id,
                    outcome=r.outcome.value,
                    error_summary=r.error_summary,
                ).model_dump()
                for r in completed_results
                if r.outcome != ChildOutcome.COMPLETED
            ]

            prompts_for_completed = [r.prompt for r in completed]
            summary, warnings = await self._do_summary_join_with_retry(
                prompts=prompts_for_completed,
                completed=completed,
                dropped=dropped,
            )

            summary_tokens = self._estimate_tokens([summary])
            end_ts = time.time()

            metrics = self._compute_metrics(
                prompts=prompts,
                completed_results=completed_results,
                summary_tokens=summary_tokens,
                start_ts=start_ts,
                end_ts=end_ts,
            )

            yield JoinedSummaryEvent(
                id=f"summary-{probe_run_id}",
                probe_run_id=probe_run_id,
                summary=summary,
                summary_tokens=summary_tokens,
                completed_children=[r.child_id for r in completed],
                dropped_children=[DroppedChild(**d) for d in dropped],
                metrics=metrics,
                validation_warnings=warnings,
            )

        finally:
            # Critical: aclose() raises GeneratorExit (not CancelledError) at
            # the most recent yield point, so the inner `except CancelledError`
            # above does NOT fire on client SSE disconnect mid-fanout. We MUST
            # check for pending children here and notify them — otherwise they
            # keep running after the quota slot is released. Codex R3 verified
            # this with a live repro.
            if any(not t.done() for t in tasks):
                try:
                    # codex r13 [R13-1, HIGH ARCH] — same fix as the
                    # inner cancel: pass startable_children so the zip
                    # alignment matches what ``tasks`` was built from.
                    await self._cancel_pending_children(
                        tasks=tasks, children=startable_children, user_id=user_id,
                    )
                except Exception as e:
                    logger.warning("pending children cancel in finally failed: %s", e)
            # Shield sandbox suspend per child + quota release from outer cancel.
            # GeneratorExit also lands here; await on a shielded coroutine is
            # allowed in finally.
            #
            # C3 PR-4.5 (codex r1 [R1-1, CRITICAL ARCH]) — mailbox-plane
            # children must SKIP the legacy suspend here: MailboxSupervisor
            # owns destroy on the RESULT_READY / CANCEL_ACK path (M1
            # single-writer). Running suspend here in parallel races the
            # supervisor's destroy() and corrupts the SandboxBinding state
            # machine. Pre-PR-5 the feature flag defaults False so every
            # child is ``subagent_control_plane='legacy'`` and the legacy
            # branch runs unchanged. The literal
            # ``_should_skip_mailbox_lifecycle`` reference also satisfies
            # the AST CI gate (§13.3) for this function body.
            for child, _ in children:
                # codex r5 [R5-3, HIGH CONTRACT] — re-read the current
                # session row BEFORE deciding to skip suspend. §11.6
                # rollback SQL can flip the row from mailbox to legacy
                # mid-run; if we trust the stale in-memory child we'd
                # skip both supervisor destroy AND legacy suspend,
                # leaking the sandbox. Best-effort: a DB lookup failure
                # falls through to legacy suspend (safer than skip).
                refreshed = None
                try:
                    refreshed = await self._session_service.get_session(
                        child.id, user_id, is_admin=True
                    )
                except Exception:
                    logger.debug(
                        "finally re-read child=%s failed — falling back to legacy suspend",
                        child.id,
                        exc_info=True,
                    )

                # codex r3 [R3-3, HIGH ARCH] — only skip suspend when
                # BOTH (a) the CURRENT row is still mailbox-plane AND
                # (b) the SPAWN_REQUEST publish actually succeeded
                # (handoff established). If publish failed OR the row
                # was rolled back, fall through to legacy suspend so
                # the sandbox isn't orphaned. ``_should_skip_mailbox_lifecycle``
                # is the AST-gate-recognized predicate (spec §13.3).
                if (
                    refreshed is not None
                    and _should_skip_mailbox_lifecycle(refreshed)
                    and child.id in self._handoff_published_child_ids
                ):
                    logger.debug(
                        "subagent_research: skip suspend child=%s — mailbox plane "
                        "+ handoff established (current row still mailbox)",
                        child.id,
                    )
                    continue
                try:
                    await asyncio.shield(
                        self._sandbox_lifecycle_service.suspend(child.id)
                    )
                except asyncio.CancelledError:
                    logger.warning(
                        "child %s sandbox suspend shielded but re-cancelled",
                        child.id,
                    )
                except Exception as e:
                    logger.warning(
                        "child %s sandbox suspend failed: %s", child.id, e
                    )
            try:
                await asyncio.shield(
                    self._quota_service.release(user_id, probe_run_id)
                )
            except asyncio.CancelledError:
                logger.warning("quota release shielded but re-cancelled")
            except Exception as e:
                logger.warning("quota release failed: %s", e)
            # Metric write moved into finally so client disconnect after summary
            # still records the run. Strict gate: only write the canonical
            # metric schema when the full pipeline reached _compute_metrics
            # (metrics is not None). Partial / cancelled runs are intentionally
            # NOT recorded with summary_tokens=0/metrics={} placeholders, which
            # would corrupt downstream aggregations.
            if metrics is not None:
                try:
                    await asyncio.shield(
                        self._write_metric_line(
                            probe_run_id=probe_run_id,
                            parent_session_id=parent_id,
                            completed_results=completed_results,
                            summary_tokens=summary_tokens or 0,
                            metrics=metrics,
                        )
                    )
                except asyncio.CancelledError:
                    logger.warning("metric write shielded but re-cancelled")
                except Exception as e:
                    logger.warning("metric write failed: %s", e)

    def _compute_metrics(
        self,
        prompts: list[str],
        completed_results: list[ChildResult],
        summary_tokens: int,
        start_ts: float,
        end_ts: float,
    ) -> dict:
        """Compute multi-metric: parent_context_savings + ratio + latency + success_rate."""
        HISTORICAL_EXPANSION_FACTOR = 5.0

        total_child_transcript = sum(r.transcript_tokens for r in completed_results)
        baseline_estimate = int(
            sum(self._estimate_tokens([p]) for p in prompts)
            * HISTORICAL_EXPANSION_FACTOR
        )
        completed_count = sum(
            1 for r in completed_results if r.outcome == ChildOutcome.COMPLETED
        )
        success_rate = (
            completed_count / len(completed_results) if completed_results else 0.0
        )
        compression_ratio = (
            summary_tokens / total_child_transcript
            if total_child_transcript > 0 else 0.0
        )

        return {
            "parent_context_savings_estimator": max(
                0, baseline_estimate - summary_tokens
            ),
            "compression_ratio": compression_ratio,
            "end_to_end_latency_seconds": int(end_ts - start_ts),
            "child_success_rate": success_rate,
            "completed_count": completed_count,
            "total_count": len(completed_results),
            "total_child_transcript_tokens": total_child_transcript,
            "summary_tokens": summary_tokens,
        }

    async def _write_metric_line(
        self,
        probe_run_id: str,
        completed_results: list[ChildResult],
        summary_tokens: int,
        metrics: dict,
        *,
        parent_session_id: str,
    ) -> None:
        """Append a jsonl metric line to ~/.gstack/metrics/.

        File I/O runs in a thread so the async path stays non-blocking.
        """
        parent_id = parent_session_id
        if parent_id is None:
            raise ValueError("parent_session_id is required")
        line = {
            "ts": time.time(),
            "probe_run_id": probe_run_id,
            "parent_session_id": parent_id,
            "child_session_ids": [r.child_id for r in completed_results],
            "child_outcomes": [r.outcome.value for r in completed_results],
            "child_transcript_tokens": [
                r.transcript_tokens for r in completed_results
            ],
            "summary_tokens": summary_tokens,
            "metrics": metrics,
            "user_quality_rating": None,
        }

        def _write_sync() -> None:
            METRIC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(METRIC_LOG_PATH, "a") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        try:
            await asyncio.to_thread(_write_sync)
        except Exception as e:
            logger.warning("metric jsonl write failed: %s", e)
