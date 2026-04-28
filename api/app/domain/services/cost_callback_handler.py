"""B4 M0 Phase E: CostCallbackHandler — turns LangChain callbacks into CostRecords.

One handler is attached per session. It:

- On ``on_chat_model_start``: caches the call's metadata (``langgraph_node``,
  ``langgraph_step``) + invocation params (``model``) under ``run_id`` in a
  bounded LRU (Issue 1C: maxsize=10_000 so missing-``on_llm_end`` paths can't
  OOM the process over days of uptime).
- On ``on_llm_end``: pops the cached entry, extracts ``usage_metadata`` off
  the returned ``AIMessage``, computes ``total_usd`` via ``compute_cost``,
  builds a ``CostRecord``, and schedules persistence via
  ``asyncio.create_task`` so the callback returns in sync-path p99 < 2ms.

Design notes:
- The persister is injected so tests use a simple list-append and prod uses
  a fresh ``DBUnitOfWork`` per call (cheap + avoids session leaks).
- Persister exceptions are swallowed — a failing DB must not take down the
  LLM call path. The circuit breaker (future extension) keeps the write
  path short-circuited while DB is down.
- ``flush_pending`` is exposed for tests and for FINISHING-drain safety.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Awaitable, Callable, List, Mapping, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.pricing.static_pricing import (
    PRICING_VERSION,
    compute_cost,
    get_price,
)

logger = logging.getLogger(__name__)


Persister = Callable[[CostRecord], Awaitable[None]]

# UUID5 namespace for session-level degraded sentinel rows. Constant on
# purpose — implementation detail of write_session_degraded_marker, but
# the deterministic key relies on this namespace being stable across
# deploys so ON CONFLICT DO NOTHING dedups idempotent writes.
# Generated 2026-04-27; do not change without a migration plan.
_DRAIN_DEGRADED_NAMESPACE: uuid.UUID = uuid.UUID(
    "d73a3bc4-deed-49bb-99c9-c4fbd4caaeae"
)

# Soft upper bound on marker write latency. Caller (terminal drain) cannot
# afford to re-wedge for the full DB pool wait if the persister is the
# source of degradation. ``write_session_degraded_marker`` uses
# ``asyncio.create_task + asyncio.wait`` (NOT ``wait_for``) so this is a
# bound on time-to-RETURN-False, not a hard upper bound on persister wall
# time — the abandoned task continues in the background.
_MARKER_WRITE_TIMEOUT_SECONDS: float = 1.0

# Schema bounds from cost_records (cost_record_orm.py:78-79 + migration
# b4m0_add_cost_records.py:58-59). Without clamping in _build_record, an
# overlong model/provider from a misbehaving adapter would (a) blow the
# main INSERT with string-data-right-truncation, then (b) blow the
# per-call degraded-marker insert too, because ``_persist_safely`` builds
# the marker via ``replace(record, ...)`` which preserves the original
# overlong values — leaving the ledger entirely missing that LLM call.
# Clamping at ``_build_record`` is the single read-point that covers both.
_MAX_MODEL_LEN: int = 128
_MAX_PROVIDER_LEN: int = 64


@dataclass
class _PendingEntry:
    run_id: UUID
    metadata: dict
    model: str
    provider: str
    attempt_ix: int = 0


@dataclass(frozen=True)
class FlushResult:
    """Outcome of one ``flush_pending`` call.

    - ``drained``: True iff every persist task pending at entry settled
      within the timeout window.
    - ``pending_count``: tasks still in flight at return (0 iff drained).
    - ``persist_failures``: delta of ``_persist_failure_count`` observed
      during this flush window only. Failures predating the call are NOT
      counted; failures occurring AFTER return continue to be covered by
      ``_persist_safely``'s per-task degraded-row path.
    """

    drained: bool
    pending_count: int
    persist_failures: int


# Translate internal LangGraph node names to stable product-level names.
# by_node shows up in the UI, so we don't want a refactor that renames
# ``planner_node`` → ``planner_v2`` to break downstream dashboards.
# Unknown nodes pass through verbatim and get caught by the coverage test.
_LANGGRAPH_NODE_MAP: dict[str, str] = {
    # main_graph.py
    "planner_node": "planner",
    "executor_node": "executor",
    "updater_node": "updater",
    "summarizer_node": "summarizer",
    "interrupt_node": "interrupt",
    # react_graph.py
    "pre_llm_node": "react_pre_llm",
    "llm_node": "react_llm",
    "tool_node": "react_tool",
    "interrupt_helper": "interrupt",
    # graph-external calls (set explicitly via metadata by the caller).
    # Every string here matches a literal ``metadata.langgraph_node``
    # value emitted from somewhere in the codebase — keeping the entry
    # makes the bucket name stable across LangGraph internal renames
    # and refactors. The node-mapping test enumerates these as the
    # authoritative graph-external-bucket allowlist.
    "background_summary": "background_summary",
    "conversation_summary": "conversation_summary",
    "memory_gate": "memory_gate",
    "context_compaction": "context_compaction",
    "continuation_classifier": "continuation_classifier",
    # fallback attribution — LLM call outside any known node
    "out_of_graph": "out_of_graph",
    # internal marker written by _persist_safely's degraded retry
    "persist_degraded": "persist_degraded",
}


def _map_node_name(raw: str | None) -> str:
    if not raw:
        return "out_of_graph"
    return _LANGGRAPH_NODE_MAP.get(raw, raw)


# Module-level GC anchor + observability for marker tasks abandoned by the
# soft-bound timeout in write_session_degraded_marker. Without a hard
# reference, asyncio could collect the task before it logs a late
# completion / failure.
_PENDING_MARKER_TASKS: set[asyncio.Task] = set()


def _on_marker_task_done(task: asyncio.Task) -> None:
    _PENDING_MARKER_TASKS.discard(task)
    if task.cancelled():
        # We never cancel marker tasks. Cancellation here means process
        # shutdown caught the abandoned task — log at INFO, not ERROR.
        logger.info(
            "abandoned marker task %s cancelled at shutdown",
            task.get_name(),
        )
        return
    exc = task.exception()
    if exc is not None:
        logger.warning(
            "abandoned marker task %s eventually raised: %s",
            task.get_name(),
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )


def _infer_provider(model: str) -> str:
    """Best-effort provider tag from model name. Rough classifier; M0 only."""
    m = (model or "").lower()
    if m.startswith(("gpt-", "o1")):
        return "openai"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("claude"):
        return "anthropic"
    if m.startswith("gemini"):
        return "google"
    if m.startswith(("kimi", "moonshot")):
        return "moonshot"
    if m.startswith("glm"):
        return "zhipu"
    if m.startswith(("qwen", "qwen-vl")):
        return "dashscope"
    return "unknown"


class CostCallbackHandler(AsyncCallbackHandler):
    """LangChain async handler that records one CostRecord per LLM invocation."""

    def __init__(
        self,
        session_id: str,
        user_id: str,
        persister: Persister,
        *,
        max_pending: int = 10_000,
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.user_id = user_id
        self._persister = persister
        self._pending: "OrderedDict[UUID, _PendingEntry]" = OrderedDict()
        self._max_pending = max_pending
        self._active_tasks: set[asyncio.Task] = set()
        # Count real-persist failures; each failure also triggers a
        # best-effort degraded-marker insert so aggregation surfaces
        # ``partial`` instead of silently reporting ``actual`` with missing
        # rows. Consumers can also read this directly from memory (e.g.
        # terminal-status code) to add a final session-level signal.
        self._persist_failure_count: int = 0
        # Dedup the WARNING fired by ``_clamp`` so a single misbehaving
        # adapter doesn't flood logs once per LLM call. Per (session, kind)
        # at most — ops only need to know "this session had an overlong
        # value at least once for this kind", not the count.
        self._seen_overlong: set[str] = set()

    # ---- LangChain callback surface ------------------------------------- #

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        invocation_params = kwargs.get("invocation_params") or {}
        model = str(invocation_params.get("model") or "unknown")
        # Prefer the authoritative provider_id plumbed from ProviderProfile
        # (via adapter ``_identifying_params``); only fall back to the name-
        # prefix heuristic when the adapter hasn't declared one.
        provider_id = invocation_params.get("provider_id")
        provider = (
            str(provider_id)
            if provider_id and provider_id != "unknown"
            else _infer_provider(model)
        )
        entry = _PendingEntry(
            run_id=run_id,
            metadata=dict(metadata or {}),
            model=model,
            provider=provider,
        )

        # Bounded LRU — Issue 1C. Evict oldest BEFORE insert so the invariant
        # `len(pending) <= max_pending` is preserved post-insert.
        while len(self._pending) >= self._max_pending:
            evicted_key, _ = self._pending.popitem(last=False)
            logger.warning(
                "CostCallbackHandler: LRU evicted pending run_id=%s "
                "(session_id=%s, cap=%d) — missing on_llm_end path",
                evicted_key,
                self.session_id,
                self._max_pending,
            )

        self._pending[run_id] = entry

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle an errored/cancelled LLM run.

        LangChain fires ``on_llm_error`` instead of ``on_llm_end`` on errors
        and cancellations, and for streaming mid-flight failures it still
        hands us the already-merged partial ``LLMResult`` via
        ``kwargs["response"]`` (see langchain_core chat_models.py lines
        ~712-722). Mid-flight failures matter for the ledger because the
        provider may well have already billed tokens that streamed before
        the error: user disconnected an SSE, a retryable upstream 5xx
        arrived after half the tokens, etc.

        Behavior:
        - Early failure (no partial response) → just pop the pending entry;
          nothing to bill.
        - Partial response with usage_metadata → write a CostRecord with
          ``cost_status=UNKNOWN`` + tokens/cost derived from the usage
          (UNKNOWN signals "mid-flight error"; mixing with a downstream
          successful row lets aggregation surface ``partial``).
        - Partial response with content but no usage → write an UNKNOWN
          record with zero tokens so the gap is visible in the ledger.
        """
        entry = self._pending.pop(run_id, None)
        if entry is None:
            logger.debug(
                "CostCallbackHandler: on_llm_error without matching start "
                "(session_id=%s run_id=%s)",
                self.session_id, run_id,
            )
            return

        response = kwargs.get("response")
        if response is None:
            logger.debug(
                "CostCallbackHandler: LLM run errored before any response, "
                "dropped pending run_id=%s (session_id=%s, error=%s)",
                run_id, self.session_id, type(error).__name__,
            )
            return

        usage_metadata = self._extract_usage_metadata(response)
        if usage_metadata is None and not self._response_has_partial_payload(response):
            # Empty shell — early failure dressed up in an LLMResult wrapper.
            logger.debug(
                "CostCallbackHandler: errored LLM run had no partial data, "
                "dropped pending run_id=%s (session_id=%s, error=%s)",
                run_id, self.session_id, type(error).__name__,
            )
            return

        # Partial-stream fallback tags may also be present (streaming
        # fallback could error after the stamp landed on the first chunk).
        self._apply_stream_fallback_tags(response, entry)

        # Build the record with normal pricing logic, then force
        # ``cost_status=UNKNOWN`` so the aggregate flags this session as
        # partial. Token counts + total_usd still reflect provider-reported
        # usage so ops can see what the error actually cost.
        record = self._build_record(entry, usage_metadata)
        record = replace(record, cost_status=CostStatus.UNKNOWN)

        task = asyncio.create_task(self._persist_safely(record))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    @staticmethod
    def _response_has_partial_payload(response: LLMResult) -> bool:
        """Did anything that could cost tokens actually stream before the error?

        Streaming providers bill for tool-call deltas and reasoning content
        even when ``msg.content`` stays empty. If we only checked ``content``,
        a run that emitted ``AIMessage(content="", tool_calls=[...])`` before
        an upstream disconnect would drop silently out of the ledger.

        Accept as "partial payload":
        - any non-empty ``content``
        - any ``tool_calls`` or ``tool_call_chunks`` (streaming tool_call)
        - any ``additional_kwargs`` — Actus adapters stash
          ``reasoning_content`` / ``reasoning_signature`` there for
          thinking-capable providers (see provider_profiles/_parse.py).
        """
        try:
            gen = response.generations[0][0]
        except (IndexError, AttributeError):
            return False
        msg = getattr(gen, "message", None)
        if msg is None:
            return False
        if getattr(msg, "content", None):
            return True
        if getattr(msg, "tool_calls", None):
            return True
        if getattr(msg, "tool_call_chunks", None):
            return True
        if getattr(msg, "additional_kwargs", None):
            return True
        return False

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        entry = self._pending.pop(run_id, None)
        if entry is None:
            logger.debug(
                "CostCallbackHandler: on_llm_end without matching start "
                "(session_id=%s run_id=%s)",
                self.session_id,
                run_id,
            )
            return

        # Streaming fallback path stamps the merged message's
        # ``response_metadata`` with ``actus_fallback_*`` fields because
        # ``_notify_fallback_escalation`` can't reach handlers during
        # streaming (LangChain doesn't forward run_manager into subclass
        # ``_astream``). Apply those tags to the pending entry here.
        self._apply_stream_fallback_tags(response, entry)

        usage_metadata = self._extract_usage_metadata(response)
        record = self._build_record(entry, usage_metadata)

        task = asyncio.create_task(self._persist_safely(record))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

    # ---- Public helpers -------------------------------------------------- #

    async def flush_pending(
        self, timeout: float | None = None
    ) -> FlushResult:
        """Wait for in-flight persist tasks to settle.

        Used in tests and in the FINISHING drain to ensure cost records are
        flushed before a session transitions to COMPLETED.

        ``timeout`` caps the wait. On timeout, the still-pending tasks are
        left running — they'll resolve later in the background; per-task
        failures are already logged by ``_persist_safely``. ``None`` =
        block indefinitely (test-only path).

        Returns FlushResult so callers can dispatch a session-level degraded
        marker on ``drained=False`` or ``persist_failures>0`` (Issue 1D).

        Uses ``asyncio.wait`` (not ``asyncio.wait_for(gather(...))``) on
        purpose: ``wait_for`` cancels its awaitable on timeout, which would
        cancel the inner persist tasks and drop cost rows on the floor.
        ``wait`` lets survivors keep running after we return.
        """
        failures_at_entry = self._persist_failure_count
        if not self._active_tasks:
            return FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )
        # Snapshot the set so concurrent task-done callbacks mutating
        # ``_active_tasks`` don't race with ``asyncio.wait``'s iteration.
        pending = set(self._active_tasks)
        if timeout is None:
            await asyncio.gather(*pending, return_exceptions=True)
            return FlushResult(
                drained=True,
                pending_count=0,
                persist_failures=self._persist_failure_count - failures_at_entry,
            )
        _, not_done = await asyncio.wait(pending, timeout=timeout)
        if not_done:
            logger.warning(
                "CostCallbackHandler.flush_pending timeout %.1fs — "
                "%d task(s) still in flight for session_id=%s; leaving "
                "them to run in background.",
                timeout,
                len(not_done),
                self.session_id,
            )
        return FlushResult(
            drained=not bool(not_done),
            pending_count=len(not_done),
            persist_failures=self._persist_failure_count - failures_at_entry,
        )

    def pending_keys(self) -> frozenset[UUID]:
        """Snapshot of pending run_ids — exposed for tests + diagnostics."""
        return frozenset(self._pending.keys())

    def mark_fallback_escalation(
        self,
        run_id: UUID,
        *,
        attempt_ix: int = 1,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Record that the fallback wrapper escalated this run.

        Called by ``ActusFallbackChatModel`` when the primary adapter raises
        and fallback takes over. Updates the pending entry's ``attempt_ix``,
        and — critically — its ``model`` / ``provider`` so the CostRecord
        reflects the adapter that actually billed, not the primary whose
        ``_identifying_params`` was captured at ``on_chat_model_start`` time.

        The method is sync (pure in-memory dict update) so the fallback
        wrapper can call it without awaiting. It's idempotent — a second
        call with a higher attempt_ix wins; lower ones are ignored.
        """
        entry = self._pending.get(run_id)
        if entry is None:
            logger.debug(
                "mark_fallback_escalation for unknown run_id=%s (session_id=%s)",
                run_id, self.session_id,
            )
            return
        if attempt_ix > entry.attempt_ix:
            entry.attempt_ix = attempt_ix
        if model:
            entry.model = str(model)
        if provider:
            entry.provider = str(provider)

    # ---- Internals ------------------------------------------------------- #

    @staticmethod
    def _apply_stream_fallback_tags(
        response: LLMResult, entry: _PendingEntry
    ) -> None:
        """Mirror ``mark_fallback_escalation`` for the streaming fallback path.

        Reads ``actus_fallback_attempt_ix`` / ``actus_fallback_model`` /
        ``actus_fallback_provider`` off the merged message's
        ``response_metadata``. No-op when those keys are absent (primary
        succeeded, or non-fallback adapter).
        """
        try:
            msg = response.generations[0][0].message
        except (IndexError, AttributeError):
            return
        rm = getattr(msg, "response_metadata", None) or {}
        if not rm:
            return
        fb_attempt = rm.get("actus_fallback_attempt_ix")
        if fb_attempt is not None:
            try:
                fb_attempt_int = int(fb_attempt)
            except (TypeError, ValueError):
                fb_attempt_int = 0
            if fb_attempt_int > entry.attempt_ix:
                entry.attempt_ix = fb_attempt_int
        fb_model = rm.get("actus_fallback_model")
        if fb_model:
            entry.model = str(fb_model)
        fb_provider = rm.get("actus_fallback_provider")
        if fb_provider:
            entry.provider = str(fb_provider)

    @staticmethod
    def _extract_usage_metadata(response: LLMResult) -> Optional[Mapping[str, Any]]:
        try:
            gen = response.generations[0][0]
        except (IndexError, AttributeError):
            return None
        msg = getattr(gen, "message", None)
        if msg is None:
            return None
        return getattr(msg, "usage_metadata", None)

    def _clamp(self, value: str, max_len: int, kind: str) -> str:
        """Truncate ``value`` to ``max_len``; log WARNING on first hit per kind.

        ``kind`` is the field label ("model" / "provider") used both for
        the dedup key and the log message. Dedup is per-handler-instance
        (handler is per session) so the same overlong value seen on every
        LLM call in one session logs once, not N times. Different sessions
        get fresh handlers and log independently.
        """
        if len(value) <= max_len:
            return value
        if kind not in self._seen_overlong:
            self._seen_overlong.add(kind)
            logger.warning(
                "CostCallbackHandler: %s exceeds %d chars for session_id=%s "
                "(len=%d) — clamping to fit cost_records.%s",
                kind,
                max_len,
                self.session_id,
                len(value),
                kind,
            )
        return value[:max_len]

    def _build_record(
        self,
        entry: _PendingEntry,
        usage_metadata: Optional[Mapping[str, Any]],
    ) -> CostRecord:
        metadata = entry.metadata
        node_name = _map_node_name(metadata.get("langgraph_node"))
        step_ix = int(metadata.get("langgraph_step") or 0)

        # Pricing lookup uses the original (unclamped) names so an unknown
        # adapter that just happens to use a >128-char model name doesn't
        # flip from priced→unpriced just because we truncated the suffix.
        # The truncated values land on the persisted CostRecord only.
        price = get_price(entry.model, entry.provider)

        input_tokens = int((usage_metadata or {}).get("input_tokens", 0) or 0)
        output_tokens = int((usage_metadata or {}).get("output_tokens", 0) or 0)
        input_details = dict((usage_metadata or {}).get("input_token_details") or {})
        output_details = dict((usage_metadata or {}).get("output_token_details") or {})
        cache_read = int(input_details.get("cache_read", 0) or 0)
        cache_write = int(
            input_details.get("cache_creation", 0)
            or input_details.get("cache_write", 0)
            or 0
        )
        reasoning = int(output_details.get("reasoning", 0) or 0)

        if usage_metadata is None:
            # We do not compute a char-count estimate in M0, so calling this
            # ``estimated`` would lie about the signal. ``unknown`` is the
            # honest label: no usage observed, no cost derived. A future
            # milestone may reintroduce ``ESTIMATED`` once real estimation
            # lands — the enum value is kept in the domain for that.
            status = CostStatus.UNKNOWN
            total_usd = Decimal(0)
        elif price is None:
            status = CostStatus.UNKNOWN
            total_usd = Decimal(0)
        else:
            status = CostStatus.ACTUAL
            total_usd = compute_cost(usage_metadata, price) or Decimal(0)

        return CostRecord(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            user_id=self.user_id,
            run_id=str(entry.run_id),
            node_name=node_name,
            step_ix=step_ix,
            attempt_ix=entry.attempt_ix,
            model=self._clamp(entry.model, _MAX_MODEL_LEN, "model"),
            provider=self._clamp(entry.provider, _MAX_PROVIDER_LEN, "provider"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            reasoning_tokens=reasoning,
            total_usd=total_usd,
            pricing_version=PRICING_VERSION,
            cost_status=status,
            created_at=datetime.now(timezone.utc),
        )

    def _build_session_degraded_record(self, reason: str) -> CostRecord:
        """Build a session-level degraded sentinel CostRecord (Issue 1D).

        Uses a deterministic uuid5 key over (session_id, reason) so the
        DB's ON CONFLICT DO NOTHING dedups repeated writes for the same
        (session, reason) pair. Different reasons collapse to the same
        aggregation outcome but produce distinct rows for operator triage.
        """
        run_id = str(uuid.uuid5(
            _DRAIN_DEGRADED_NAMESPACE,
            f"{self.session_id}:terminal-drain:{reason}",
        ))
        return CostRecord(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            user_id=self.user_id,
            run_id=run_id,
            node_name="persist_degraded",
            step_ix=0,
            attempt_ix=0,
            # NOT empty strings: the aggregation service unconditionally
            # rolls every row into by_model[r.model] / by_provider[r.provider]
            # (see cost_aggregation_service.py). Empty strings would surface
            # as ``{"": "0"}`` in GET /cost and leak through to the UI.
            # Use stable internal sentinels — recognizable in API response
            # as not-a-real-model and not-a-real-provider.
            model="session_degraded_marker",
            provider="internal",
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            total_usd=Decimal(0),
            pricing_version=PRICING_VERSION,
            cost_status=CostStatus.UNKNOWN,
            created_at=datetime.now(timezone.utc),
        )

    async def write_session_degraded_marker(self, reason: str) -> bool:
        """Best-effort session-level degraded sentinel (Issue 1D §3.2).

        Writes one CostRecord with node_name="persist_degraded" so the
        existing aggregation override forces cost_status=partial. The
        run_id is a stable uuid5 over (session_id, "terminal-drain", reason).

        Calls self._persister DIRECTLY (NOT _persist_safely) to avoid
        marker-of-marker recursion.

        Uses ``asyncio.create_task + asyncio.wait`` (NOT ``wait_for``):
        ``wait_for`` would cancel the inner task on timeout AND await the
        cancellation; if the DB driver is non-cooperative the outer caller
        could re-wedge past the nominal bound. ``wait`` instead does NOT
        touch the inner task on timeout — we abandon observation, park
        the task in ``_PENDING_MARKER_TASKS`` for late-completion logging,
        and return ``False``.

        Returns True iff the persister completed within the soft bound
        without raising; False on bound exceeded, persister exception, or
        any other error.
        """
        marker = self._build_session_degraded_record(reason)
        task = asyncio.create_task(
            self._persister(marker),
            name=f"cost-marker-{self.session_id}-{reason}",
        )
        done, _ = await asyncio.wait(
            [task], timeout=_MARKER_WRITE_TIMEOUT_SECONDS
        )
        if not done:
            # Abandon observation; park task for GC + late-completion logging.
            _PENDING_MARKER_TASKS.add(task)
            task.add_done_callback(_on_marker_task_done)
            logger.warning(
                "marker write soft-bound exceeded for session=%s reason=%s; "
                "task abandoned to background",
                self.session_id,
                reason,
            )
            return False
        # Cancellation guard: ``task.exception()`` re-raises CancelledError
        # on a cancelled task instead of returning it. Without this check,
        # a marker task that ends up cancelled (e.g., the persister itself
        # raised CancelledError, or the underlying DB driver propagated
        # cancellation) would crash out of write_session_degraded_marker
        # via CancelledError — violating the documented "False on any
        # other error" contract AND skipping the subsequent status write
        # in ``_set_terminal_status._terminal_op``.
        if task.cancelled():
            logger.warning(
                "marker write task cancelled for session=%s reason=%s",
                self.session_id,
                reason,
            )
            return False
        exc = task.exception()
        if exc is not None:
            logger.warning(
                "marker write failed for session=%s reason=%s: %s",
                self.session_id,
                reason,
                exc,
            )
            return False
        return True

    @property
    def persist_failure_count(self) -> int:
        """Count of real-persist failures seen by this handler.

        Used by aggregation + callers that need a session-local signal to
        distinguish "complete ledger" from "ledger known to be incomplete".
        """
        return self._persist_failure_count

    async def _persist_safely(self, record: CostRecord) -> None:
        try:
            await self._persister(record)
            return
        except Exception as exc:  # noqa: BLE001 — fire-and-forget by design
            self._persist_failure_count += 1
            logger.warning(
                "CostCallbackHandler: persister failed for session_id=%s "
                "run_id=%s: %s",
                self.session_id,
                record.run_id,
                exc,
            )

        # Best-effort degraded-marker insert. We write a CostRecord with
        # the same ``run_id`` so the DB's unique index dedups the *original*
        # row if it managed to partially land. Fields are zeroed out and
        # ``cost_status`` is ``UNKNOWN`` so aggregation mixes with surviving
        # ``actual`` rows → ``partial``. A second failure is logged but
        # swallowed (degraded fallback is best-effort; the design doc
        # acknowledges a total-DB-outage case cannot be covered without
        # adding a side-channel).
        try:
            marker = replace(
                record,
                cost_status=CostStatus.UNKNOWN,
                total_usd=Decimal(0),
                input_tokens=0,
                output_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                reasoning_tokens=0,
                node_name="persist_degraded",
            )
            await self._persister(marker)
        except Exception as marker_exc:  # noqa: BLE001
            logger.warning(
                "CostCallbackHandler: degraded marker insert also failed "
                "for session_id=%s run_id=%s: %s",
                self.session_id,
                record.run_id,
                marker_exc,
            )
