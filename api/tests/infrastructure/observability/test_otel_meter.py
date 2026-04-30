"""B5 PR-S2-3: ``OtelMeter`` satisfies the ``MeterPort`` Protocol.

Locks:

- The Protocol surface is satisfied (``create_counter`` /
  ``create_histogram`` / ``create_up_down_counter``).
- Counter ``.add(value, attributes=...)`` produces a ``DataPoint``
  on the in-memory reader with the same value + attributes.
- Histogram ``.record(value, attributes=...)`` produces a histogram
  data point with sum == recorded value (single record case).
- ``unit`` / ``description`` flow through to the produced metric
  metadata.
- The default ctor (no ``meter`` arg) reads the OTel global meter
  without crashing.
"""
from __future__ import annotations

from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.infrastructure.observability.otel_meter import OtelMeter


@pytest.fixture
def meter_and_reader() -> tuple[OtelMeter, InMemoryMetricReader]:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    return OtelMeter(provider.get_meter("actus-test")), reader


def _flatten_metrics(reader: InMemoryMetricReader) -> list[Any]:
    metrics_data = reader.get_metrics_data()
    if metrics_data is None:
        return []
    out: list[Any] = []
    for resource_metric in metrics_data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                out.append(metric)
    return out


def test_otel_meter_satisfies_meter_port_surface(meter_and_reader):
    impl, _ = meter_and_reader
    assert callable(impl.create_counter)
    assert callable(impl.create_histogram)
    assert callable(impl.create_up_down_counter)


def test_counter_records_value_and_attributes(meter_and_reader):
    impl, reader = meter_and_reader
    counter = impl.create_counter(
        "cost_usd_micro",
        unit="usd_micro",
        description="LLM cost per invocation",
    )
    counter.add(1234, attributes={"model": "gpt-4", "llm_provider": "openai"})

    metrics = _flatten_metrics(reader)
    assert len(metrics) == 1
    metric = metrics[0]
    assert metric.name == "cost_usd_micro"
    assert metric.unit == "usd_micro"

    points = list(metric.data.data_points)
    assert len(points) == 1
    point = points[0]
    assert point.value == 1234
    assert point.attributes.get("model") == "gpt-4"
    assert point.attributes.get("llm_provider") == "openai"


def test_histogram_records_latency(meter_and_reader):
    impl, reader = meter_and_reader
    hist = impl.create_histogram(
        "llm.latency", unit="ms", description="LLM call latency"
    )
    hist.record(
        250, attributes={"model": "gpt-4", "llm_provider": "openai"}
    )

    metrics = _flatten_metrics(reader)
    assert len(metrics) == 1
    point = list(metrics[0].data.data_points)[0]
    # Single record → sum == that record.
    assert point.sum == 250
    assert point.count == 1
    assert point.attributes.get("model") == "gpt-4"


def test_up_down_counter_supports_negative_values(meter_and_reader):
    impl, reader = meter_and_reader
    udc = impl.create_up_down_counter(
        "active_sessions", description="active count"
    )
    udc.add(5)
    udc.add(-2)

    metrics = _flatten_metrics(reader)
    assert len(metrics) == 1
    point = list(metrics[0].data.data_points)[0]
    # Cumulative aggregation → 5 + (-2) = 3.
    assert point.value == 3


def test_default_constructor_uses_global_meter():
    """No-arg ctor pulls the OTel global meter (post-setup_observability)."""
    impl = OtelMeter()
    counter = impl.create_counter("noop_counter")
    counter.add(1)
