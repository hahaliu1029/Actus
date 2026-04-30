"""B5 PR-S2-2: ``OtelTracer`` satisfies the ``TracerPort`` Protocol.

Domain code consumes ``TracerPort`` (start_span / start_as_current_span)
without importing OTel. The infra adapter wraps an OTel
``opentelemetry.trace.Tracer`` so the lifetime + parent-child semantics
of OTel spans pass through unchanged. Tests pin:

- The Protocol surface is satisfied.
- A ``start_as_current_span`` block produces exactly one finished span
  on the in-memory exporter, with the supplied attributes.
- ``None`` attribute values are dropped at the boundary so downstream
  exporters (OTLP / Loki / Prometheus) never see ``"step_id": null``.
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


@pytest.fixture
def in_memory_tracer() -> tuple[Any, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("actus-test")
    return tracer, exporter


def test_otel_tracer_satisfies_tracer_port_surface(in_memory_tracer):
    tracer, _ = in_memory_tracer
    impl = OtelTracer(tracer)

    # Structural Protocol check: the two methods exist + are callable.
    # The domain ``TracerPort`` Protocol uses ``...`` defaults which makes
    # ``isinstance(..., TracerPort)`` brittle across Python versions; the
    # explicit attribute check is the durable shape assertion.
    assert callable(impl.start_span)
    assert callable(impl.start_as_current_span)


def test_start_as_current_span_emits_span_with_attributes(in_memory_tracer):
    tracer, exporter = in_memory_tracer
    impl = OtelTracer(tracer)

    with impl.start_as_current_span(
        "graph.node.executor",
        attributes={
            "graph_node": "executor_node",
            "step_id": "step-42",
            "trace_id": "0123456789abcdef0123456789abcdef",
        },
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "graph.node.executor"
    assert span.attributes.get("graph_node") == "executor_node"
    assert span.attributes.get("step_id") == "step-42"


def test_none_attribute_values_dropped_before_export(in_memory_tracer):
    tracer, exporter = in_memory_tracer
    impl = OtelTracer(tracer)

    with impl.start_as_current_span(
        "graph.node.planner",
        attributes={
            "graph_node": "planner_node",
            "step_id": None,  # CLI / startup / tests with no step context
            "session_id": None,
        },
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    assert "step_id" not in span.attributes
    assert "session_id" not in span.attributes
    assert span.attributes.get("graph_node") == "planner_node"


def test_start_span_emits_outside_current_context(in_memory_tracer):
    tracer, exporter = in_memory_tracer
    impl = OtelTracer(tracer)

    span = impl.start_span(
        "background.task", attributes={"graph_node": "summarizer"}
    )
    span.end()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "background.task"
    assert spans[0].attributes.get("graph_node") == "summarizer"


def test_default_constructor_uses_global_tracer():
    """No-arg ctor pulls the OTel global tracer (post-setup_observability).

    Smoke test only: confirm it doesn't crash and produces a usable
    context manager that can enter / exit cleanly. The global may be a
    ``ProxyTracer`` if ``setup_observability`` hasn't been called, but
    the contract is "callable and exits without error".
    """
    impl = OtelTracer()
    with impl.start_as_current_span("noop"):
        pass
