"""B5 PR-S2-3: OTel LLM metrics callback.

Emits two OTel meter signals per LLM invocation, alongside the
existing ``CostCallbackHandler`` (which writes a ``CostRecord`` row
to Postgres for finance / per-session aggregation). The two surfaces
are intentionally orthogonal:

- DB ``CostRecord`` — durable per-call ledger, queryable by session,
  used for billing / quotas / audit.
- OTel meter — short-window histogram + counter for dashboards
  (Phoenix / Prometheus / Grafana), used for SLO / latency / error
  rate / cost-per-second alarms.

Two instruments
---------------
- ``llm.latency_ms`` — histogram, attributes
  ``model`` / ``llm_provider`` / ``graph_node``. Records the wall
  clock latency between ``on_chat_model_start`` and ``on_llm_end``
  for each invocation.
- ``cost_usd_micro`` — counter, attributes
  ``model`` / ``llm_provider`` / ``graph_node``. Each call adds the
  computed USD cost ×1_000_000 (so the bucket stays integer-typed).
  Counter (not histogram) so dashboards can compute "USD per minute"
  via OTel's standard rate aggregation.

FOLLOW-11 anchor (TODO2.md #13 Sprint 2 PR-S2-3): "MeterPort 接
record_llm_invocation 扩 latency_ms / model / llm_provider + ...
CostCallbackHandler 写 cost_usd_micro". This callback is the
"meter side"; the existing ``CostCallbackHandler`` keeps the DB
ledger semantics unchanged.

Privacy
-------
Like ``OtelToolSpanCallback``, this handler MUST NOT surface raw
prompt content / response content / tool args. Only the call's
metadata (model, provider, graph_node, latency, cost) is recorded.
``llm.latency_ms`` and ``cost_usd_micro`` are inherently
non-sensitive aggregates.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Mapping, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from app.domain.external.observability import MeterPort
from app.domain.services.pricing.static_pricing import compute_cost, get_price


logger = logging.getLogger(__name__)


@dataclass
class _PendingLLMCall:
    """In-flight LLM call: start time + per-call attributes.

    Stashed under ``run_id`` between ``on_chat_model_start`` and
    ``on_llm_end`` so the handler can compute latency without
    holding a reference to the LLM instance. ``attempt_ix`` is
    bumped to ``1+`` by ``mark_fallback_escalation`` (or by the
    streaming-path tag reader in ``on_llm_end``) when the fallback
    adapter takes over — so the recorded cost / latency attribute
    to the adapter that actually billed.
    """

    started_at: float  # time.monotonic() value
    model: str
    llm_provider: str
    graph_node: str
    attempt_ix: int = 0


def _infer_provider(model: str) -> str:
    """Best-effort provider tag from model name.

    Mirror of ``cost_callback_handler._infer_provider`` so the two
    callbacks emit the same ``llm_provider`` attribute value for the
    same model — joining DB ledger and OTel meter on
    ``(model, llm_provider)`` is unambiguous.
    """
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


def _resolve_model(
    serialized: dict[str, Any] | None,
    invocation_params: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> str:
    """Pull the model name from LangChain's start-event payload.

    Search order:
    1. ``invocation_params["model"]`` (LangChain Chat model standard).
    2. ``serialized["kwargs"]["model"]``.
    3. ``metadata["ls_model_name"]`` (LangSmith convention).
    4. ``"unknown"`` fallback.
    """
    if invocation_params:
        m = invocation_params.get("model")
        if isinstance(m, str) and m:
            return m
    if serialized:
        kwargs = serialized.get("kwargs") or {}
        m = kwargs.get("model")
        if isinstance(m, str) and m:
            return m
    if metadata:
        m = metadata.get("ls_model_name")
        if isinstance(m, str) and m:
            return m
    return "unknown"


def _resolve_provider(
    invocation_params: dict[str, Any] | None, model: str
) -> str:
    """Resolve ``llm_provider`` in DB-ledger order (reviewer P2 fix).

    The cost ledger uses ``provider_id`` as the canonical key — pricing
    tables key on it (``static_pricing.PRICING_TABLE``) and
    ``CostCallbackHandler.on_chat_model_start`` reads the same field.
    Joining DB ledger + OTel meter on ``(model, llm_provider)``
    requires the same value: e.g. ``gpt-4o`` belongs to
    ``"openai_official"`` (canonical), not the heuristic ``"openai"``.

    Search order:

    1. ``invocation_params["provider_id"]`` (authoritative; plumbed
       from ``ProviderProfile`` via adapter ``_identifying_params``).
    2. Heuristic ``_infer_provider(model)`` — fallback for adapters
       that don't declare a profile yet.
    """
    if invocation_params:
        pid = invocation_params.get("provider_id")
        if isinstance(pid, str) and pid and pid != "unknown":
            return pid
    return _infer_provider(model)


def _resolve_graph_node(metadata: dict[str, Any] | None) -> str:
    """Pull ``graph_node`` from LangGraph metadata.

    LangGraph writes ``langgraph_node`` (the registered node name)
    onto the callback metadata. We surface it as ``graph_node`` to
    match the canonical attribute contract — same value the
    ``traced_node`` decorator binds onto the contextvar.
    """
    if not metadata:
        return "out_of_graph"
    raw = metadata.get("langgraph_node")
    if isinstance(raw, str) and raw:
        return raw
    return "out_of_graph"


def _extract_usage_metadata(
    response: LLMResult,
) -> Optional[Mapping[str, Any]]:
    """Read the LangChain-standard ``usage_metadata`` off the AI message.

    Production adapters (``ActusChatModel`` / ``ActusResponsesModel``)
    stamp **token counters** here — ``input_tokens`` / ``output_tokens``
    plus optional ``input_token_details`` / ``output_token_details``
    sub-dicts for cache reads / cache writes / reasoning tokens. They
    do NOT stamp a precomputed ``total_usd`` (reviewer P1 finding).
    The cost ledger uses the same field via the canonical
    ``compute_cost(usage_metadata, get_price(model, provider))`` path
    in ``static_pricing`` — we mirror it here so OTel meter + DB
    ledger derive cost from the same source of truth.
    """
    try:
        gen = response.generations[0][0]
    except (IndexError, AttributeError):
        return None
    msg = getattr(gen, "message", None)
    if msg is None:
        return None
    return getattr(msg, "usage_metadata", None)


def _response_metadata(response: LLMResult) -> dict[str, Any]:
    """Best-effort fetch of the AI message's ``response_metadata`` dict.

    Used by the streaming-path fallback escalation reader. Returns an
    empty dict if the structure is unexpected.
    """
    try:
        msg = response.generations[0][0].message
    except (IndexError, AttributeError):
        return {}
    rm = getattr(msg, "response_metadata", None) or {}
    if not isinstance(rm, dict):
        return {}
    return rm


class OtelLLMMetricsCallback(AsyncCallbackHandler):
    """LangChain callback that records LLM latency + cost via OTel."""

    def __init__(self, meter: MeterPort) -> None:
        self._latency_hist = meter.create_histogram(
            "llm.latency_ms",
            unit="ms",
            description="LLM call latency from start to end (wall clock)",
        )
        self._cost_counter = meter.create_counter(
            "cost_usd_micro",
            unit="usd_micro",
            description=(
                "LLM call cost in micro-USD (USD * 1_000_000) — counter "
                "so dashboards aggregate via standard rate aggregation"
            ),
        )
        # ``run_id -> _PendingLLMCall``. Bounded by LangChain's own
        # life-cycle: every ``on_chat_model_start`` is paired with
        # exactly one ``on_llm_end`` / ``on_llm_error``. Cleared on
        # both terminal events.
        self._pending: dict[UUID, _PendingLLMCall] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        invocation_params = kwargs.get("invocation_params") or {}
        model = _resolve_model(serialized, invocation_params, metadata)
        self._pending[run_id] = _PendingLLMCall(
            started_at=time.monotonic(),
            model=model,
            llm_provider=_resolve_provider(invocation_params, model),
            graph_node=_resolve_graph_node(metadata),
        )

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Some LangChain LLM types fire ``on_llm_start`` instead of
        # ``on_chat_model_start``. Mirror the same stash logic.
        invocation_params = kwargs.get("invocation_params") or {}
        model = _resolve_model(serialized, invocation_params, metadata)
        self._pending[run_id] = _PendingLLMCall(
            started_at=time.monotonic(),
            model=model,
            llm_provider=_resolve_provider(invocation_params, model),
            graph_node=_resolve_graph_node(metadata),
        )

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        pending = self._pending.pop(run_id, None)
        if pending is None:
            return

        # Reviewer P2 (fallback) fix: when streaming-path fallback
        # engages, the wrapper stamps ``actus_fallback_*`` keys on
        # the merged message's ``response_metadata``. Sync-path
        # fallback calls ``mark_fallback_escalation`` directly. We
        # support BOTH paths so latency / cost attribute to the
        # adapter that actually billed.
        self._apply_stream_fallback_tags(response, pending)

        latency_ms = (time.monotonic() - pending.started_at) * 1000.0
        attrs = {
            "model": pending.model,
            "llm_provider": pending.llm_provider,
            "graph_node": pending.graph_node,
            # B5 PR-S2-3 reviewer round-2 P2 fix: emit ``attempt_ix``
            # so dashboards can distinguish primary success
            # (``attempt_ix == 0``) from fallback success
            # (``attempt_ix >= 1``). Critical for the common
            # ``ActusFallbackChatModel`` ``api_type="auto"`` case where
            # primary (Chat) and fallback (Responses) share the SAME
            # ``model`` + ``llm_provider`` (same ProviderProfile, just
            # different transport API), so without this attr fallback
            # is invisible on the metric surface even though the
            # latency / cost it records IS the fallback's.
            # ``attempt_ix`` is part of the canonical attribute
            # contract (``CANONICAL_ATTRIBUTES`` in
            # ``domain/external/observability.py``).
            "attempt_ix": pending.attempt_ix,
        }
        try:
            self._latency_hist.record(latency_ms, attributes=attrs)
        except Exception:  # pragma: no cover  defensive
            logger.debug(
                "OtelLLMMetricsCallback: latency record failed",
                exc_info=True,
            )

        # Reviewer P1 fix: derive USD from the canonical pricing-table
        # path (``compute_cost(usage_metadata, get_price(model, provider))``)
        # so we don't depend on adapters stamping a precomputed
        # ``total_usd`` field — production adapters never do. Same
        # source of truth as the DB ledger; OTel meter + DB stay in
        # lockstep on cost.
        usage_metadata = _extract_usage_metadata(response)
        total_usd = self._compute_total_usd(
            pending.model, pending.llm_provider, usage_metadata
        )
        if total_usd is not None and total_usd > 0:
            micro = int(round(float(total_usd) * 1_000_000))
            if micro > 0:
                try:
                    self._cost_counter.add(micro, attributes=attrs)
                except Exception:  # pragma: no cover  defensive
                    logger.debug(
                        "OtelLLMMetricsCallback: cost add failed",
                        exc_info=True,
                    )

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        # Discard the pending entry — error path doesn't have usage
        # metadata + the latency would be misleading (timeouts /
        # retries). Drop without recording so dashboards see a clean
        # "successful call latency" distribution.
        self._pending.pop(run_id, None)

    # ---- Fallback escalation hooks -------------------------------------- #

    def mark_fallback_escalation(
        self,
        run_id: UUID,
        *,
        attempt_ix: int = 1,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        """Duck-typed hook called by ``ActusFallbackChatModel`` (sync path).

        Mirrors ``CostCallbackHandler.mark_fallback_escalation``: when
        the primary adapter raises and the fallback takes over, this
        updates the pending entry's ``model`` / ``llm_provider`` so the
        eventual ``on_llm_end`` records latency / cost against the
        fallback adapter's identity (the one that actually billed).

        Sync (pure dict update) so the wrapper can call it without
        awaiting. Idempotent — higher ``attempt_ix`` wins.
        """
        pending = self._pending.get(run_id)
        if pending is None:
            logger.debug(
                "mark_fallback_escalation for unknown run_id=%s", run_id
            )
            return
        if attempt_ix > pending.attempt_ix:
            pending.attempt_ix = attempt_ix
        if model:
            pending.model = str(model)
        if provider:
            pending.llm_provider = str(provider)

    @staticmethod
    def _apply_stream_fallback_tags(
        response: LLMResult, pending: _PendingLLMCall
    ) -> None:
        """Streaming-path mirror of ``mark_fallback_escalation``.

        Reads ``actus_fallback_attempt_ix`` / ``actus_fallback_model``
        / ``actus_fallback_provider`` off the merged AI message's
        ``response_metadata``. No-op when those keys are absent
        (primary succeeded, or non-fallback adapter).

        This is necessary because LangChain's ``BaseChatModel.astream``
        does NOT forward ``run_manager`` into subclass ``_astream``;
        the wrapper can't reach handlers via ``_notify_fallback_escalation``
        on the streaming path. Tagging response_metadata is the
        out-of-band signal the wrapper uses instead.
        """
        rm = _response_metadata(response)
        if not rm:
            return
        fb_attempt = rm.get("actus_fallback_attempt_ix")
        if fb_attempt is not None:
            try:
                fb_attempt_int = int(fb_attempt)
            except (TypeError, ValueError):
                fb_attempt_int = 0
            if fb_attempt_int > pending.attempt_ix:
                pending.attempt_ix = fb_attempt_int
        fb_model = rm.get("actus_fallback_model")
        if fb_model:
            pending.model = str(fb_model)
        fb_provider = rm.get("actus_fallback_provider")
        if fb_provider:
            pending.llm_provider = str(fb_provider)

    # ---- Cost compute ---------------------------------------------------- #

    @staticmethod
    def _compute_total_usd(
        model: str,
        provider: str,
        usage_metadata: Optional[Mapping[str, Any]],
    ) -> Optional[Decimal]:
        """Compute USD via the canonical pricing-table path.

        Returns ``None`` when no usage metadata is available or no
        price entry matches ``(provider, model)`` — caller skips the
        cost emit in either case (mirrors the DB ledger's
        ``cost_status=UNKNOWN`` semantics: no data → no rate).
        """
        if usage_metadata is None:
            return None
        price = get_price(model, provider)
        if price is None:
            return None
        return compute_cost(usage_metadata, price)
