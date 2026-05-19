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
from app.domain.models.event import BaseEvent
from app.domain.models.session import Session
from app.domain.services.execution_supervisor import ExecutionSupervisor
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
    ) -> None:
        self._session_service = session_service
        self._agent_service = agent_service
        self._supervisor = execution_supervisor
        self._token_estimator = token_estimator
        self._summary_llm = summary_llm
        self._classifier = classifier
        self._sandbox_lifecycle_service = sandbox_lifecycle_service
        self._quota_service = quota_service

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
        sample_session_id: str,
        user_id: str,
        prompts: list[str],
        max_children: int = 3,
    ) -> AsyncGenerator[BaseEvent, None]:
        """Run a research probe. See module docstring for invariants.

        Yields ChildStartedEvent × N → ChildDoneEvent × N (interleaved by
        as_completed order) → JoinedSummaryEvent (final).
        """
        from app.application.errors.exceptions import (
            BadRequestError, ConflictError, NotFoundError,
        )

        probe_run_id = uuid.uuid4().hex
        start_ts = time.time()

        parent = await self._session_service.get_session(
            sample_session_id, user_id, is_admin=False
        )
        if parent is None:
            raise NotFoundError(
                f"Session {sample_session_id} not found or not accessible"
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

        try:
            for prompt in prompts[:max_children]:
                child = await self._session_service.create_session_with_parent(
                    user_id=user_id,
                    sample_session_id=sample_session_id,
                    tool_filter_preset=_SUBAGENT_RESEARCH_PRESET,
                )
                # NOTE: spec calls `session_service.update_title(...)` here for
                # UX (prefixing child titles with "[probe]"), but that method
                # does not exist on SessionService. Skipping — title is
                # cosmetic; correctness does not depend on it.
                children.append((child, prompt))

            for child, prompt in children:
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
                for child, prompt in children
            )

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
                await self._cancel_pending_children(
                    tasks=tasks, children=children, user_id=user_id,
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
                    await self._cancel_pending_children(
                        tasks=tasks, children=children, user_id=user_id,
                    )
                except Exception as e:
                    logger.warning("pending children cancel in finally failed: %s", e)
            # Shield sandbox suspend per child + quota release from outer cancel.
            # GeneratorExit also lands here; await on a shielded coroutine is
            # allowed in finally.
            for child, _ in children:
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
                            sample_session_id=sample_session_id,
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
        sample_session_id: str,
        completed_results: list[ChildResult],
        summary_tokens: int,
        metrics: dict,
    ) -> None:
        """Append a jsonl metric line to ~/.gstack/metrics/.

        File I/O runs in a thread so the async path stays non-blocking.
        """
        line = {
            "ts": time.time(),
            "probe_run_id": probe_run_id,
            "sample_session_id": sample_session_id,
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
