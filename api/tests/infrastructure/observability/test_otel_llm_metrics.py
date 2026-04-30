"""B5 PR-S2-3: ``OtelLLMMetricsCallback`` emits llm.latency_ms +
cost_usd_micro per LLM invocation.

Locks (after reviewer round):

- ``on_chat_model_start`` → ``on_llm_end`` records latency (>=0)
  with ``model`` / ``llm_provider`` / ``graph_node`` attrs.
- **P1 (reviewer)**: cost is derived via the canonical
  ``compute_cost(usage_metadata, get_price(model, provider))`` path
  — same source of truth as the DB ledger. Production adapters
  stamp **token counters** in ``usage_metadata`` (no ``total_usd``);
  the meter computes USD from those token counts + the pricing table.
- **P2 (reviewer)**: ``llm_provider`` is sourced from
  ``invocation_params["provider_id"]`` first (canonical, matches DB
  ledger); only when missing/``"unknown"`` do we fall back to
  ``_infer_provider(model)``. This keeps OTel meter and DB ledger
  joinable on ``(model, llm_provider)``.
- **P2 (fallback)**: when ``ActusFallbackChatModel`` escalates from
  primary to fallback, both the sync hook
  (``mark_fallback_escalation``) and the streaming-path
  ``response_metadata.actus_fallback_*`` keys are honoured —
  latency / cost attribute to the adapter that actually billed.
- ``on_llm_error`` drops the pending entry (no latency / cost
  recorded for failed calls — keeps the success-latency
  distribution clean).
- ``_infer_provider`` mirrors ``cost_callback_handler._infer_provider``
  for adapters without a declared ``provider_id``.
"""
from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.infrastructure.observability.otel_llm_metrics import (
    OtelLLMMetricsCallback,
    _infer_provider,
)
from app.infrastructure.observability.otel_meter import OtelMeter


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def callback_and_reader() -> tuple[OtelLLMMetricsCallback, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = OtelMeter(provider.get_meter("actus-test"))
    return OtelLLMMetricsCallback(meter), reader


def _flatten_metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    metrics_data = reader.get_metrics_data()
    out: dict[str, Any] = {}
    if metrics_data is None:
        return out
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                out[metric.name] = metric
    return out


def _make_llm_result(
    *,
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_read: int = 0,
    response_metadata: dict[str, Any] | None = None,
    include_usage: bool = True,
) -> LLMResult:
    """Build a minimal LLMResult mirroring real adapter output shape.

    Production ``ActusChatModel`` / ``ActusResponsesModel`` stamp
    LangChain-standard ``UsageMetadata`` token counters (NOT a
    precomputed ``total_usd``). The meter computes USD from these
    via the pricing table — same path as the DB ledger.
    """
    usage_meta: dict[str, Any] | None = None
    if include_usage:
        usage_meta = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if cache_read:
            usage_meta["input_token_details"] = {"cache_read": cache_read}
    msg = AIMessage(
        content="ok",
        usage_metadata=usage_meta,
        response_metadata=response_metadata or {},
    )
    gen = ChatGeneration(message=msg)
    return LLMResult(generations=[[gen]])


@pytest.mark.anyio
async def test_records_latency_with_canonical_attrs(callback_and_reader):
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    metrics = _flatten_metrics(reader)
    assert "llm.latency_ms" in metrics
    point = list(metrics["llm.latency_ms"].data.data_points)[0]
    assert point.count == 1
    assert point.sum >= 0  # wall clock latency, can be ~0 in fast tests
    assert point.attributes.get("model") == "gpt-4o"
    assert point.attributes.get("llm_provider") == "openai_official"
    assert point.attributes.get("graph_node") == "executor_node"


@pytest.mark.anyio
async def test_records_cost_via_pricing_table_from_token_counters(
    callback_and_reader,
):
    """Reviewer P1: cost is computed via the canonical pricing-table
    path (``compute_cost(usage_metadata, get_price(model, provider))``)
    — production adapters stamp **token counters** in
    ``usage_metadata``, NEVER a precomputed ``total_usd``.

    Sanity model: ``gpt-4o`` under ``openai_official`` (real OpenAI
    pricing). Input 1_000_000 tokens / output 1_000_000 tokens →
    `2.5 + 10.0` = `12.5 USD = 12_500_000 micro`. We assert the
    canonical compute path produced the SAME number the meter
    counter recorded — locking the source of truth.
    """
    from decimal import Decimal

    from app.domain.services.pricing.static_pricing import (
        compute_cost,
        get_price,
    )

    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(
        response=_make_llm_result(
            input_tokens=1_000_000, output_tokens=1_000_000
        ),
        run_id=run_id,
    )

    # Re-derive expected cost from the same canonical path.
    expected_usd: Decimal = compute_cost(
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        get_price("gpt-4o", "openai_official"),
    ) or Decimal(0)
    expected_micro = int(round(float(expected_usd) * 1_000_000))
    assert expected_micro > 0  # sanity — pricing table populated

    metrics = _flatten_metrics(reader)
    point = list(metrics["cost_usd_micro"].data.data_points)[0]
    assert point.value == expected_micro
    assert point.attributes.get("model") == "gpt-4o"
    assert point.attributes.get("llm_provider") == "openai_official"


@pytest.mark.anyio
async def test_provider_id_from_invocation_params_overrides_heuristic(
    callback_and_reader,
):
    """Reviewer P2: ``invocation_params['provider_id']`` is the
    canonical key (matches ``ProviderProfile.provider_id`` and
    ``CostCallbackHandler``); the heuristic ``_infer_provider`` is
    only a fallback. ``gpt-4o`` would heuristic to ``"openai"``, but
    the pricing table keys on ``"openai_official"`` — meter MUST
    pick the canonical value so DB ledger + OTel meter join cleanly.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    # Canonical wins, NOT the heuristic ``"openai"``.
    assert point.attributes.get("llm_provider") == "openai_official"


@pytest.mark.anyio
async def test_provider_id_unknown_falls_back_to_heuristic(
    callback_and_reader,
):
    """``provider_id`` absent or literally ``"unknown"`` → heuristic
    ``_infer_provider(model)`` is used so adapters that don't yet
    declare a profile still get a sensible ``llm_provider`` attr.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "deepseek-chat"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={"model": "deepseek-chat"},  # no provider_id
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("llm_provider") == "deepseek"


@pytest.mark.anyio
async def test_no_cost_emit_when_pricing_table_misses(callback_and_reader):
    """No price entry for ``(model, provider)`` → cost counter stays
    empty (mirrors DB ledger ``cost_status=UNKNOWN``); latency still
    recorded.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "exotic-unpriced-model"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node"},
        invocation_params={
            "model": "exotic-unpriced-model",
            "provider_id": "no_such_provider",
        },
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    metrics = _flatten_metrics(reader)
    assert "llm.latency_ms" in metrics
    cost_metric = metrics.get("cost_usd_micro")
    if cost_metric is not None:
        assert len(list(cost_metric.data.data_points)) == 0


@pytest.mark.anyio
async def test_no_cost_emit_when_usage_metadata_missing(callback_and_reader):
    """No ``usage_metadata`` (early failure / non-streaming partial)
    → no cost emit; latency still recorded.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "planner_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(
        response=_make_llm_result(include_usage=False),
        run_id=run_id,
    )

    metrics = _flatten_metrics(reader)
    assert "llm.latency_ms" in metrics
    cost_metric = metrics.get("cost_usd_micro")
    if cost_metric is not None:
        assert len(list(cost_metric.data.data_points)) == 0


@pytest.mark.anyio
async def test_mark_fallback_escalation_switches_attribution(
    callback_and_reader,
):
    """Reviewer P2 fallback: sync-path fallback (``mark_fallback_escalation``)
    swaps the pending entry's ``model`` / ``llm_provider`` so the
    eventual ``on_llm_end`` records latency / cost against the
    fallback adapter's identity.

    Mirrors ``CostCallbackHandler.mark_fallback_escalation``;
    ``ActusFallbackChatModel._notify_fallback_escalation`` calls it
    via duck-typing on any handler that exposes the method.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    # Primary adapter: gpt-4o
    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    # Fallback fires → claude-haiku-4-5 / anthropic_official.
    handler.mark_fallback_escalation(
        run_id,
        attempt_ix=1,
        model="claude-haiku-4-5",
        provider="anthropic_official",
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("model") == "claude-haiku-4-5"
    assert point.attributes.get("llm_provider") == "anthropic_official"


@pytest.mark.anyio
async def test_streaming_fallback_response_metadata_switches_attribution(
    callback_and_reader,
):
    """Reviewer P2 fallback: streaming-path fallback stamps
    ``actus_fallback_*`` keys on the merged AIMessage's
    ``response_metadata`` (see ``ActusFallbackChatModel._stamp_fallback_escalation``).
    The metrics callback reads these in ``on_llm_end`` and updates
    attribution before recording — covering the streaming case where
    ``mark_fallback_escalation`` can't fire (LangChain's
    ``BaseChatModel.astream`` doesn't forward ``run_manager``).
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(
        response=_make_llm_result(
            response_metadata={
                "actus_fallback_attempt_ix": 1,
                "actus_fallback_model": "claude-haiku-4-5",
                "actus_fallback_provider": "anthropic_official",
            }
        ),
        run_id=run_id,
    )

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("model") == "claude-haiku-4-5"
    assert point.attributes.get("llm_provider") == "anthropic_official"


@pytest.mark.anyio
async def test_mark_fallback_escalation_unknown_run_id_is_noop(
    callback_and_reader,
):
    """Late escalation hook (we never saw the start) must not crash."""
    handler, _ = callback_and_reader
    handler.mark_fallback_escalation(
        uuid4(),
        attempt_ix=1,
        model="claude-haiku-4-5",
        provider="anthropic_official",
    )


@pytest.mark.anyio
async def test_attempt_ix_zero_for_primary_success(callback_and_reader):
    """Reviewer round-2 P2 baseline: primary success → ``attempt_ix=0``.

    Establishes the dashboard pivot: ``attempt_ix == 0`` is the
    primary-success bucket; ``attempt_ix >= 1`` is the fallback
    bucket. Without this baseline the next test (same-model/provider
    fallback) couldn't be the canary signal.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("attempt_ix") == 0


@pytest.mark.anyio
async def test_attempt_ix_emitted_for_same_model_provider_fallback(
    callback_and_reader,
):
    """Reviewer round-2 P2: in the common ``api_type='auto'`` Chat→
    Responses fallback (``ActusFallbackChatModel``), primary and
    fallback adapters share the SAME ``model`` + ``provider_id``
    (same ``ProviderProfile``, just different transport API). The
    only meter signal that fallback fired is ``attempt_ix``. This
    test simulates that exact case and asserts the metric point
    carries ``attempt_ix == 1`` so dashboards can:

    - Filter ``attempt_ix > 0`` to see fallback rate per provider
    - Compute ``fallback_rate = sum(attempt_ix>0) / sum(*)``
    - Alert when same-provider transport fallback spikes

    Without this fix the metric would be indistinguishable from a
    primary success — fallback completely invisible on the meter
    surface even though latency / cost reflect the fallback path.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    # Primary: gpt-4o / openai_official (Chat API).
    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    # ``ActusFallbackChatModel`` notifies escalation with the
    # SAME model + provider_id (Responses API uses the same
    # ProviderProfile). Only ``attempt_ix`` differs from primary.
    handler.mark_fallback_escalation(
        run_id,
        attempt_ix=1,
        model="gpt-4o",
        provider="openai_official",
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    # Same model + provider as primary — only attempt_ix tells us
    # fallback engaged.
    assert point.attributes.get("model") == "gpt-4o"
    assert point.attributes.get("llm_provider") == "openai_official"
    assert point.attributes.get("attempt_ix") == 1, (
        f"same-model/provider fallback must surface as attempt_ix>=1; "
        f"got {dict(point.attributes)!r}"
    )


@pytest.mark.anyio
async def test_attempt_ix_emitted_via_streaming_response_metadata(
    callback_and_reader,
):
    """Symmetric round-2 P2 lock for the streaming path: when fallback
    is signalled via ``response_metadata.actus_fallback_attempt_ix``
    (LangChain ``BaseChatModel.astream`` doesn't forward
    ``run_manager`` so the sync hook can't fire), the metric attrs
    must STILL carry the elevated ``attempt_ix``.
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(
        response=_make_llm_result(
            response_metadata={
                "actus_fallback_attempt_ix": 1,
                # Same-provider/same-model — only attempt_ix is the signal.
                "actus_fallback_model": "gpt-4o",
                "actus_fallback_provider": "openai_official",
            }
        ),
        run_id=run_id,
    )

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("attempt_ix") == 1


@pytest.mark.anyio
async def test_cost_counter_also_carries_attempt_ix(callback_and_reader):
    """Round-2 P2: ``cost_usd_micro`` must also carry ``attempt_ix`` so
    fallback cost can be sliced separately from primary cost on the
    same dashboard. (Both meters share the same ``attrs`` dict — this
    test pins the shared-attrs invariant so a future refactor that
    forks attrs between the two emits is caught.)
    """
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    handler.mark_fallback_escalation(
        run_id,
        attempt_ix=2,  # second-tier fallback
        model="gpt-4o",
        provider="openai_official",
    )
    await handler.on_llm_end(
        response=_make_llm_result(input_tokens=1000, output_tokens=500),
        run_id=run_id,
    )

    cost_point = list(
        _flatten_metrics(reader)["cost_usd_micro"].data.data_points
    )[0]
    assert cost_point.attributes.get("attempt_ix") == 2
    assert cost_point.value > 0  # priced via openai_official/gpt-4o


@pytest.mark.anyio
async def test_error_path_drops_pending_no_record(callback_and_reader):
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata={"langgraph_node": "executor_node"},
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_error(
        error=RuntimeError("timeout"), run_id=run_id
    )

    metrics = _flatten_metrics(reader)
    latency = metrics.get("llm.latency_ms")
    if latency is not None:
        assert len(list(latency.data.data_points)) == 0


@pytest.mark.anyio
async def test_on_llm_start_path_also_works(callback_and_reader):
    """Some LLM types fire ``on_llm_start`` (not on_chat_model_start)."""
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_llm_start(
        serialized={"kwargs": {"model": "deepseek-chat"}},
        prompts=["hi"],
        run_id=run_id,
        metadata={"langgraph_node": "summarizer_node"},
        invocation_params={
            "model": "deepseek-chat",
            "provider_id": "deepseek_chat",
        },
    )
    await handler.on_llm_end(
        response=_make_llm_result(input_tokens=10_000, output_tokens=5_000),
        run_id=run_id,
    )

    metrics = _flatten_metrics(reader)
    cost_point = list(metrics["cost_usd_micro"].data.data_points)[0]
    assert cost_point.value > 0  # priced via deepseek_chat / deepseek-chat
    assert cost_point.attributes.get("llm_provider") == "deepseek_chat"


@pytest.mark.anyio
async def test_unknown_run_id_on_end_is_noop(callback_and_reader):
    """``on_llm_end`` for a run we never saw must not crash."""
    handler, _ = callback_and_reader
    await handler.on_llm_end(response=_make_llm_result(), run_id=uuid4())


@pytest.mark.anyio
async def test_graph_node_falls_back_to_out_of_graph(callback_and_reader):
    """No metadata → ``graph_node="out_of_graph"`` (canonical bucket)."""
    handler, reader = callback_and_reader
    run_id = uuid4()

    await handler.on_chat_model_start(
        serialized={"kwargs": {"model": "gpt-4o"}},
        messages=[],
        run_id=run_id,
        metadata=None,
        invocation_params={
            "model": "gpt-4o",
            "provider_id": "openai_official",
        },
    )
    await handler.on_llm_end(response=_make_llm_result(), run_id=run_id)

    point = list(
        _flatten_metrics(reader)["llm.latency_ms"].data.data_points
    )[0]
    assert point.attributes.get("graph_node") == "out_of_graph"


def test_provider_inference_matches_cost_callback():
    """Heuristic mirror of ``cost_callback_handler._infer_provider``.

    Used as a fallback when ``provider_id`` is absent — production
    canonical path goes through ``invocation_params['provider_id']``
    (see ``test_provider_id_from_invocation_params_overrides_heuristic``).
    """
    from app.domain.services.cost_callback_handler import (
        _infer_provider as cost_infer,
    )

    for model in [
        "gpt-4",
        "o1-mini",
        "deepseek-chat",
        "claude-3.5",
        "gemini-pro",
        "kimi-128k",
        "moonshot-v1",
        "glm-4",
        "qwen-max",
        "qwen-vl-plus",
        "unknown-model",
    ]:
        assert _infer_provider(model) == cost_infer(model), (
            f"provider mismatch for {model!r}: "
            f"meter={_infer_provider(model)!r} vs "
            f"cost={cost_infer(model)!r}"
        )
