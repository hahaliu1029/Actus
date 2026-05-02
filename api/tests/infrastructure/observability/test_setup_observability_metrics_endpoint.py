"""B5 PR-S3-3: PrometheusMetricReader wiring contract.

When ``METRICS_ENDPOINT_TOKEN`` is set, ``setup_observability`` MUST
append a ``PrometheusMetricReader`` to the ``MeterProvider``'s readers
so ``GET /api/v1/metrics`` can serve the Prometheus exposition format.

Reader composition is additive — it MUST coexist with any OTLP /
Console reader so a deployment can run push (OTLP to Phoenix) AND
pull (Prometheus scrape from sidecar) simultaneously without choosing.

Empty token = NO ``PrometheusMetricReader`` registered. This locks the
"zero overhead when disabled" contract: no
``prometheus_client.REGISTRY`` registration, no extra memory for the
collector wrapper.
"""
from __future__ import annotations

import pytest

from app.infrastructure.observability import (
    setup_observability,
    teardown_observability,
)
from core.config import get_settings


@pytest.fixture(autouse=True)
def _reset_provider_state():
    teardown_observability()
    yield
    teardown_observability()


@pytest.fixture(autouse=True)
def _clean_default_env(monkeypatch):
    """Strip any host env that would taint the default-mode tests."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    monkeypatch.delenv("OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("METRICS_ENDPOINT_TOKEN", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _readers(provider) -> list:
    return list(getattr(provider, "_sdk_config").metric_readers)


def _reader_class_names(provider) -> list[str]:
    return [type(r).__name__ for r in _readers(provider)]


def test_default_token_unset_does_not_wire_prometheus_reader():
    """Spec ``OTLP_ENDPOINT="" + METRICS_ENDPOINT_TOKEN=""`` → zero
    readers. No Prometheus collector landing in the global
    ``prometheus_client.REGISTRY``.
    """
    p = setup_observability()
    names = _reader_class_names(p.meter_provider)
    assert names == [], (
        f"default install must not wire a PrometheusMetricReader; "
        f"got {names!r}"
    )


def test_token_set_wires_prometheus_metric_reader(monkeypatch):
    """``METRICS_ENDPOINT_TOKEN`` non-empty → exactly one
    ``PrometheusMetricReader`` registered. Reader class is what ties
    the OTel meter pipeline to ``prometheus_client.REGISTRY``.
    """
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-token-123")
    get_settings.cache_clear()
    p = setup_observability()
    names = _reader_class_names(p.meter_provider)
    assert "PrometheusMetricReader" in names, (
        f"token set must wire PrometheusMetricReader; got readers={names!r}"
    )


def test_token_set_coexists_with_otlp_reader(monkeypatch):
    """OTLP push + Prometheus pull on the same MeterProvider.

    Operators commonly want OTLP to a central trace-correlated metric
    backend AND a local scrape endpoint for alerting. The OTel SDK
    supports multiple readers on a single ``MeterProvider`` — assert
    both land.
    """
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-token-123")
    get_settings.cache_clear()
    p = setup_observability()
    names = _reader_class_names(p.meter_provider)
    assert "PeriodicExportingMetricReader" in names, (
        f"OTLP reader missing alongside Prometheus; got {names!r}"
    )
    assert "PrometheusMetricReader" in names, (
        f"Prometheus reader missing alongside OTLP; got {names!r}"
    )


def test_token_set_coexists_with_stdout_reader(monkeypatch):
    """``OTEL_EXPORTER=stdout`` (dev mode) + Prometheus token set.

    Dev workflow: console-print metrics for inspection AND scrape via
    a local Prometheus instance. Both readers MUST land.
    """
    monkeypatch.setenv("OTEL_EXPORTER", "stdout")
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-token-123")
    get_settings.cache_clear()
    p = setup_observability()
    names = _reader_class_names(p.meter_provider)
    assert "PeriodicExportingMetricReader" in names, (
        f"stdout reader missing alongside Prometheus; got {names!r}"
    )
    assert "PrometheusMetricReader" in names, (
        f"Prometheus reader missing alongside stdout; got {names!r}"
    )


def test_teardown_unregisters_prometheus_collector_from_global_registry(
    monkeypatch,
):
    """Test-cleanliness lock: ``teardown_observability`` MUST cause
    the ``PrometheusMetricReader``'s collector to disappear from
    ``prometheus_client.REGISTRY``.

    Without this, back-to-back tests that each register a fresh reader
    would crash on ``Duplicated timeseries in CollectorRegistry`` —
    every reader self-registers a ``_CustomCollector`` at construction
    time, and ``MeterProvider.shutdown`` MUST cascade to
    ``reader.shutdown`` which calls ``REGISTRY.unregister`` (OTel
    exporter contract).
    """
    from prometheus_client import REGISTRY

    def _otel_collectors() -> list:
        # ``prometheus_client`` ships with default GC / Platform /
        # Process collectors that self-register at module import. They
        # are NOT created by our reader and MUST NOT be unregistered
        # by teardown — filter by class name to keep the assertion on
        # OTel-side state only.
        return [
            c
            for c in REGISTRY._collector_to_names.keys()
            if type(c).__name__ == "_CustomCollector"
        ]

    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-token-abc")
    get_settings.cache_clear()
    p1 = setup_observability()
    pre_otel = _otel_collectors()
    assert len(pre_otel) == 1, (
        f"setup_observability must register exactly one _CustomCollector "
        f"with prometheus_client.REGISTRY; got {len(pre_otel)}"
    )

    teardown_observability()
    post_otel = _otel_collectors()
    leaked = [c for c in pre_otel if c in post_otel]
    assert leaked == [], (
        f"teardown must unregister this run's _CustomCollector from "
        f"prometheus_client.REGISTRY; leaked={leaked!r}"
    )

    # Second setup must not crash on duplicate registration —
    # ``prometheus_client.REGISTRY`` rejects re-registering the same
    # name. Without proper teardown unregistration, this would raise
    # ``ValueError: Duplicated timeseries in CollectorRegistry``.
    monkeypatch.setenv("METRICS_ENDPOINT_TOKEN", "secret-token-def")
    get_settings.cache_clear()
    p2 = setup_observability()
    assert p2 is not p1
    assert len(_otel_collectors()) == 1, (
        "second setup must register exactly one fresh _CustomCollector"
    )
