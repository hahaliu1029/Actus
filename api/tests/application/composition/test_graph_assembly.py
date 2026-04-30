"""B5 PR-S2-2: ``app.application.composition.graph_assembly`` factories.

Locks the contract:

- ``build_traced_node_decorator(tracer)`` returns a callable suitable
  for ``build_main_graph(node_decorator=...)``. Wrapping a sample node
  + invoking it produces exactly one span on the supplied tracer's
  exporter, with the expected ``graph.node.<name>`` shape.
- ``build_observability_callbacks(tracer)`` returns a non-empty list
  whose contents include an ``OtelToolSpanCallback`` instance bound to
  the supplied tracer (verified by emitting a tool span and observing
  it on the exporter).
- Both factories accept ``tracer=None`` (defaults to ``OtelTracer()``)
  for production wiring.
"""
from __future__ import annotations

from uuid import uuid4

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from app.application.composition import (
    build_observability_callbacks,
    build_traced_node_decorator,
)
from app.infrastructure.observability.otel_tool_span import OtelToolSpanCallback
from app.infrastructure.observability.otel_tracer import OtelTracer


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def tracer_and_exporter() -> tuple[OtelTracer, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OtelTracer(provider.get_tracer("actus-test"))
    return tracer, exporter


@pytest.mark.anyio
async def test_traced_node_decorator_factory_emits_span(tracer_and_exporter):
    tracer, exporter = tracer_and_exporter
    deco = build_traced_node_decorator(tracer)

    @deco
    async def planner_node(state, config):
        return {"plan": "ok"}

    await planner_node({}, {"configurable": {"step_id": "step-1"}})

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "graph.node.planner_node"
    assert spans[0].attributes.get("step_id") == "step-1"


@pytest.mark.anyio
async def test_callbacks_factory_yields_otel_tool_span_handler(
    tracer_and_exporter,
):
    tracer, exporter = tracer_and_exporter
    callbacks = build_observability_callbacks(tracer)

    assert callbacks, "expected at least one handler"
    assert any(isinstance(h, OtelToolSpanCallback) for h in callbacks)

    handler = next(h for h in callbacks if isinstance(h, OtelToolSpanCallback))
    run_id = uuid4()
    await handler.on_tool_start(
        serialized={"name": "x"},
        input_str="",
        run_id=run_id,
        inputs={"k": 1},
    )
    await handler.on_tool_end(output="ok", run_id=run_id)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "tool.x"


def test_factories_accept_none_tracer():
    """Production path: ``tracer=None`` falls through to the OTel global."""
    deco = build_traced_node_decorator(None)
    assert callable(deco)
    cbs = build_observability_callbacks(None)
    assert isinstance(cbs, list) and cbs
