"""B5 Sprint 2 Day 1-3 PoC — three-way context bridging probe.

Probe code, NOT a production deliverable. Spec line 728 marks this PR as
"Code 形式探针，可不合并主线" (probe form code, may not merge to main).

Verifies three independent context propagation paths converge on the same
``trace_id``:

1. **FastAPI middleware contextvars** (``_trace_context_var``) — the
   Sprint 1 carrier set by ``ObservabilityMiddleware`` per request.
2. **OTel** ``opentelemetry.context.Context`` — the SDK's own ContextVar.
   Bridge model: middleware starts an OTel span, takes its trace_id (16
   random bytes → 32 hex), uses that as the Actus ``TraceContext.trace_id``.
3. **LangChain / LangGraph** ``RunnableConfig.callbacks`` — the canonical
   LangGraph emission path. Callback handlers must observe the SAME OTel
   trace_id and the SAME Actus contextvar bound by the request.
4. ``asyncio.create_task`` children — both contextvars and OTel context
   auto-snapshot at task creation, so backgrounded emits stay in-trace.

If this test passes, **Plan A is unblocked**: PR-S2-1 (LoggerPort + SDK
init), PR-S2-2 (TracerPort / @traced_node), PR-S2-3 (MeterPort + cost
bridge) proceed on the OTel-spined backbone.

If this test fails persistently across the Day 1-3 window, **activate
Plan B** (spec line 410): drop OTel SDK, ship LoggerPort/TracerPort/
MeterPort as self-built wrappers writing JSONL traces; downgrade Sprint
3 to Loki + Prometheus instead of Phoenix/OTLP.

Acceptance (spec line 408):

    单个端到端 trace（FastAPI request → planner → executor → tool →
    llm.call）的所有 span 共享同一个 trace_id 且 attributes 完整通过
    validate_attributes.

To run::

    cd api && uv run pytest tests/poc/test_b5_obs_s2_three_way_bridging.py -v
"""
from __future__ import annotations

import asyncio
import uuid

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.runnables import RunnableConfig, RunnableLambda
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.domain.external.observability import (
    TraceContext,
    build_canonical_attributes,
    validate_attributes,
)
from app.infrastructure.observability.context import (
    get_trace_context,
    reset_trace_context,
    set_trace_context,
)

pytestmark = pytest.mark.anyio


def _otel_trace_id_hex(span: trace.Span) -> str:
    return format(span.get_span_context().trace_id, "032x")


@pytest.fixture(scope="module")
def _module_otel_setup():
    """Install ``TracerProvider`` once per module.

    OTel's global ``set_tracer_provider`` rejects later overrides with a
    warning; subsequent calls would silently leave the first provider in
    place, so per-test installs are unsafe. Instead we install once here
    and let each test reset the shared exporter via ``otel_exporter``.
    """
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.fixture
def otel_exporter(_module_otel_setup) -> InMemorySpanExporter:
    """Return the shared ``InMemorySpanExporter`` cleared for this test."""
    exporter = _module_otel_setup
    exporter.clear()
    yield exporter
    exporter.clear()


class _OtelObservingCallback(AsyncCallbackHandler):
    """Records the OTel trace_id and Actus contextvar at callback fire time.

    Stand-in for the future PR-S2-2 ``LangGraphOTelCallbackHandler``. The
    PoC question is: when LangChain's runnable machinery invokes
    ``on_chain_start`` from inside the request task, does the callback
    body see the same OTel current span and the same Actus ContextVar
    binding? If yes, three-way bridging Just Works™ and the production
    handler can emit spans / read attributes without explicit context
    plumbing.
    """

    def __init__(self) -> None:
        self.observed_trace_ids: list[str] = []
        self.observed_actus_ctx: list[TraceContext | None] = []

    async def on_chain_start(self, serialized, inputs, **kwargs) -> None:  # type: ignore[override]
        current = trace.get_current_span()
        ctx = current.get_span_context()
        self.observed_trace_ids.append(format(ctx.trace_id, "032x"))
        self.observed_actus_ctx.append(get_trace_context())


async def test_request_to_llm_chain_shares_trace_id(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """End-to-end probe: HTTP request → planner → executor → tool → llm.call.

    Asserts (spec acceptance):

    - Every ``build_canonical_attributes`` snapshot carries the same
      ``trace_id`` (the OTel root span's id).
    - Every snapshot passes ``validate_attributes`` (v1 contract).
    - LangChain ``RunnableConfig.callbacks`` callback fires inside the
      request's OTel + Actus context (same trace_id, non-None ctx).
    - ``asyncio.create_task`` child sees the same trace_id.
    - Every OTel exported span has the same trace_id; canonical span
      names are present.
    """
    tracer = trace.get_tracer(__name__)
    captured_attrs: list[dict] = []
    callback = _OtelObservingCallback()

    # 1) Simulate ObservabilityMiddleware: start root span + bind Actus ctx.
    with tracer.start_as_current_span("http.request") as root_span:
        request_trace_id = _otel_trace_id_hex(root_span)
        actus_ctx = TraceContext(
            trace_id=request_trace_id,
            request_id=str(uuid.uuid4()),
            event_id=str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
        )
        token = set_trace_context(actus_ctx)
        try:
            captured_attrs.append(build_canonical_attributes())

            # 2) Planner node (LangGraph @traced_node candidate)
            with tracer.start_as_current_span("graph.node.planner"):
                captured_attrs.append(
                    build_canonical_attributes(graph_node="planner")
                )

                # 3) Executor node + nested tool / llm spans
                with tracer.start_as_current_span("graph.node.executor"):
                    captured_attrs.append(
                        build_canonical_attributes(graph_node="executor")
                    )

                    # 4) Tool span — sub-span of executor
                    with tracer.start_as_current_span("tool.shell_execute"):
                        captured_attrs.append(
                            build_canonical_attributes(
                                graph_node="executor",
                                tool_name="shell_execute",
                                tool_call_id=str(uuid.uuid4()),
                            )
                        )

                    # 5) LLM call span + RunnableConfig.callbacks bridging
                    runnable = RunnableLambda(lambda x: x).with_config(
                        run_name="llm.call_chain"
                    )
                    with tracer.start_as_current_span("llm.call"):
                        captured_attrs.append(
                            build_canonical_attributes(
                                graph_node="executor",
                                llm_provider="openai",
                                model="gpt-4o-mini",
                            )
                        )
                        config: RunnableConfig = {"callbacks": [callback]}
                        await runnable.ainvoke({"input": "ping"}, config=config)

                    # 6) Background asyncio.create_task — context auto-snapshot
                    async def background_emit() -> None:
                        captured_attrs.append(
                            build_canonical_attributes(
                                graph_node="background.flush"
                            )
                        )
                        with tracer.start_as_current_span("background.task"):
                            captured_attrs.append(
                                build_canonical_attributes(
                                    graph_node="background.flush"
                                )
                            )

                    bg = asyncio.create_task(background_emit())
                    await bg
        finally:
            reset_trace_context(token)

    # === Acceptance asserts ===

    # All build_canonical_attributes snapshots pass v1 contract + share trace_id
    for attrs in captured_attrs:
        validate_attributes(attrs)
        assert attrs["trace_id"] == request_trace_id, (
            f"trace_id mismatch in build_canonical_attributes snapshot: "
            f"expected {request_trace_id}, got {attrs['trace_id']}"
        )

    # LangChain callback observed the request's trace_id + ctx
    assert callback.observed_trace_ids, (
        "LangChain RunnableConfig.callbacks handler never fired — "
        "PoC three-way bridge cannot be evaluated"
    )
    for trace_id in callback.observed_trace_ids:
        assert trace_id == request_trace_id, (
            f"LangChain callback saw OTel trace_id {trace_id}, "
            f"expected {request_trace_id}; bridge broken"
        )
    for ctx in callback.observed_actus_ctx:
        assert ctx is not None, (
            "LangChain callback fired without Actus contextvar binding"
        )
        assert ctx.trace_id == request_trace_id

    # Every OTel exported span shares the request trace_id
    spans = otel_exporter.get_finished_spans()
    span_trace_ids = {format(s.context.trace_id, "032x") for s in spans}
    span_names = sorted(s.name for s in spans)
    assert span_trace_ids == {request_trace_id}, (
        f"OTel span trace_ids: {span_trace_ids}, "
        f"expected only {{{request_trace_id}}}; span names: {span_names}"
    )

    # Canonical span names cover the chain
    expected = {
        "http.request",
        "graph.node.planner",
        "graph.node.executor",
        "tool.shell_execute",
        "llm.call",
        "background.task",
    }
    assert expected.issubset(set(span_names)), (
        f"missing spans: {expected - set(span_names)}; got {span_names}"
    )


async def test_create_task_child_sees_parent_actus_ctx(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Targeted probe: ``asyncio.create_task`` child inherits contextvar.

    Standalone from the chain test so a regression localises here. Mirrors
    the production scenario where ``memory_flush_service`` and other
    background tasks call ``asyncio.create_task(...)`` from inside a
    request handler.
    """
    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("http.request") as root_span:
        request_trace_id = _otel_trace_id_hex(root_span)
        actus_ctx = TraceContext(
            trace_id=request_trace_id,
            request_id=str(uuid.uuid4()),
            event_id=str(uuid.uuid4()),
        )
        token = set_trace_context(actus_ctx)
        try:
            child_observed: dict[str, str | None] = {}

            async def child() -> None:
                ctx_in_child = get_trace_context()
                otel_in_child = trace.get_current_span().get_span_context()
                child_observed["actus_trace_id"] = (
                    ctx_in_child.trace_id if ctx_in_child else None
                )
                child_observed["otel_trace_id"] = format(
                    otel_in_child.trace_id, "032x"
                )

            await asyncio.create_task(child())
        finally:
            reset_trace_context(token)

    assert child_observed["actus_trace_id"] == request_trace_id, (
        "asyncio.create_task child did not inherit Actus contextvar"
    )
    assert child_observed["otel_trace_id"] == request_trace_id, (
        "asyncio.create_task child did not inherit OTel context"
    )


async def test_concurrent_requests_isolated(
    otel_exporter: InMemorySpanExporter,
) -> None:
    """Sibling requests must not see each other's trace_id (isolation).

    Two ``asyncio.gather`` branches each take an independent root span +
    Actus context; assert each branch's emit sites only see its own
    trace_id, never the sibling's. Locks the multi-tenant invariant from
    Sprint 1's ``test_context_concurrent_isolation.py`` under the new
    OTel bridge.
    """
    tracer = trace.get_tracer(__name__)

    async def one_request(label: str) -> tuple[str, list[dict]]:
        attrs_seen: list[dict] = []
        with tracer.start_as_current_span(f"http.request.{label}") as span:
            tid = _otel_trace_id_hex(span)
            ctx = TraceContext(
                trace_id=tid,
                request_id=str(uuid.uuid4()),
                event_id=str(uuid.uuid4()),
                session_id=label,
            )
            token = set_trace_context(ctx)
            try:
                attrs_seen.append(build_canonical_attributes())
                await asyncio.sleep(0)
                attrs_seen.append(build_canonical_attributes(graph_node="planner"))
                await asyncio.sleep(0)
                attrs_seen.append(
                    build_canonical_attributes(graph_node="executor")
                )
            finally:
                reset_trace_context(token)
        return tid, attrs_seen

    (tid_a, attrs_a), (tid_b, attrs_b) = await asyncio.gather(
        one_request("a"),
        one_request("b"),
    )
    assert tid_a != tid_b, "two requests should have independent trace_ids"

    for attrs in attrs_a:
        assert attrs["trace_id"] == tid_a
    for attrs in attrs_b:
        assert attrs["trace_id"] == tid_b
