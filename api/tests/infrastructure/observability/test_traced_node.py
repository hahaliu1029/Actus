"""B5 PR-S2-2 + FOLLOW-10: ``traced_node`` decorator pins span attrs.

Locks the contract:

- Each call to a wrapped node coroutine produces exactly one
  ``graph.node.<name>`` span on the in-memory exporter.
- ``graph_node`` attribute is the wrapped function's ``__name__``.
- ``step_id`` is sourced from ``config["configurable"]["step_id"]``
  (FOLLOW-10) when the node accepts a ``config`` arg.
- ``trace_id`` / ``request_id`` / ``event_id`` ride through from
  ``build_canonical_attributes`` so the canonical join-key contract
  holds even from graph nodes (no middleware ctx required — the
  builder generates fallbacks when no ctx is bound).
- A node call that raises propagates the exception while still
  closing the span (no leaked active span).
- Nodes without a ``config`` parameter still get a span; ``step_id``
  is simply absent.
"""
from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.infrastructure.observability.otel_tracer import OtelTracer
from app.infrastructure.observability.traced_node import traced_node


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def tracer_and_exporter() -> tuple[OtelTracer, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-test"))
    return tracer, exporter


@pytest.mark.anyio
async def test_node_emits_one_span_with_graph_node_attribute(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def planner_node(state, config):
        return {"plan": "ok"}

    state = {"x": 1}
    config = {"configurable": {}}
    result = await planner_node(state, config)

    assert result == {"plan": "ok"}
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "graph.node.planner_node"
    assert span.attributes.get("graph_node") == "planner_node"


@pytest.mark.anyio
async def test_step_id_pulled_from_configurable_follow_10(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def executor_node(state, config):
        return {}

    state: dict[str, Any] = {}
    config = {"configurable": {"step_id": "step-7f3a"}}
    await executor_node(state, config)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("step_id") == "step-7f3a"


@pytest.mark.anyio
async def test_step_id_absent_when_not_bound_in_configurable(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def updater_node(state, config):
        return {}

    state: dict[str, Any] = {}
    config = {"configurable": {"thread_id": "t1"}}
    await updater_node(state, config)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert "step_id" not in spans[0].attributes


@pytest.mark.anyio
async def test_canonical_required_attrs_ride_through(tracer_and_exporter):
    """trace_id / request_id / event_id always present on the span.

    ``build_canonical_attributes`` generates fallback uuid values when
    no ``TraceContext`` is bound (CLI / unit-test path), so the span
    contract holds even without middleware.
    """
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def planner_node(state, config):
        return {}

    await planner_node({}, {"configurable": {}})
    span = exporter.get_finished_spans()[0]
    assert span.attributes.get("trace_id")
    assert span.attributes.get("request_id")
    assert span.attributes.get("event_id")


@pytest.mark.anyio
async def test_exception_propagates_and_span_still_closes(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def boom_node(state, config):
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        await boom_node({}, {"configurable": {}})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "graph.node.boom_node"


@pytest.mark.anyio
async def test_node_without_config_arg_still_traced(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def helper_node(state):
        return {}

    await helper_node({})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "graph.node.helper_node"
    assert "step_id" not in spans[0].attributes


@pytest.mark.anyio
async def test_decorator_preserves_function_name(tracer_and_exporter):
    """LangGraph's ``add_node`` introspects ``fn.__name__``; functools.wraps
    preserves it so node registration is unchanged."""
    tracer, _ = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def planner_node(state, config):
        return {}

    assert planner_node.__name__ == "planner_node"


@pytest.mark.anyio
async def test_step_id_resolved_from_state_current_step_attr(
    tracer_and_exporter,
):
    """Reviewer P1 fix: when the body hasn't yet injected step_id into
    ``configurable``, the wrapper still pulls step_id from
    ``state["current_step"].id`` so the OUTER node span carries it.
    """
    from dataclasses import dataclass

    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @dataclass
    class _Step:
        id: str

    @deco
    async def executor_node(state, config):
        return {}

    state = {"current_step": _Step(id="step-from-state")}
    config = {"configurable": {}}  # NO step_id in configurable
    await executor_node(state, config)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes.get("step_id") == "step-from-state"


@pytest.mark.anyio
async def test_step_id_resolved_from_state_current_step_dict(
    tracer_and_exporter,
):
    """Same as above but with a dict-shaped current_step (test ergonomic)."""
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def executor_node(state, config):
        return {}

    state = {"current_step": {"id": "step-via-dict"}}
    await executor_node(state, {"configurable": {}})

    spans = exporter.get_finished_spans()
    assert spans[0].attributes.get("step_id") == "step-via-dict"


@pytest.mark.anyio
async def test_state_step_id_takes_precedence_over_configurable(
    tracer_and_exporter,
):
    """When BOTH state.current_step.id and configurable.step_id are set,
    the state value wins (state is the authoritative live source).
    """
    tracer, exporter = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def executor_node(state, config):
        return {}

    state = {"current_step": {"id": "from-state"}}
    config = {"configurable": {"step_id": "from-configurable"}}
    await executor_node(state, config)

    assert exporter.get_finished_spans()[0].attributes.get("step_id") == (
        "from-state"
    )


@pytest.mark.anyio
async def test_node_body_sees_step_id_via_trace_context(tracer_and_exporter):
    """Reviewer P2 fix: downstream emit sites (tool span callback) read
    step_id from the contextvar that ``traced_node`` binds. Verify by
    calling ``build_canonical_attributes()`` from inside the body and
    asserting the snapshot carries step_id.
    """
    from app.domain.external.observability import build_canonical_attributes

    tracer, _ = tracer_and_exporter
    deco = traced_node(tracer)

    captured: dict = {}

    @deco
    async def executor_node(state, config):
        # Simulating what OtelToolSpanCallback / domain logger do.
        captured.update(build_canonical_attributes())

    state = {"current_step": {"id": "step-ctx-test"}}
    await executor_node(state, {"configurable": {}})

    assert captured.get("step_id") == "step-ctx-test"
    assert captured.get("graph_node") == "executor_node"


@pytest.mark.anyio
async def test_trace_context_restored_on_exit(tracer_and_exporter):
    """The contextvar binding is scope-bound — after the body exits,
    the prior ``TraceContext`` is restored.
    """
    from app.infrastructure.observability.context import (
        get_trace_context,
        set_trace_context,
        reset_trace_context,
    )
    from app.domain.external.observability import TraceContext

    tracer, _ = tracer_and_exporter
    deco = traced_node(tracer)

    @deco
    async def executor_node(state, config):
        ctx_inside = get_trace_context()
        assert ctx_inside is not None
        assert ctx_inside.step_id == "inner-step"
        assert ctx_inside.graph_node == "executor_node"

    # Pre-bind a parent context with an unrelated step.
    pre = TraceContext(
        trace_id="0123456789abcdef0123456789abcdef",
        request_id="11111111-1111-4111-8111-111111111111",
        event_id="22222222-2222-4222-8222-222222222222",
        step_id="outer-step",
    )
    token = set_trace_context(pre)
    try:
        state = {"current_step": {"id": "inner-step"}}
        await executor_node(state, {"configurable": {}})
        # After exit, the pre-bind restored.
        post = get_trace_context()
        assert post is pre
        assert post.step_id == "outer-step"
    finally:
        reset_trace_context(token)
