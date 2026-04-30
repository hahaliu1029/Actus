"""B5 PR-S2-1 acceptance: ``OTEL_EXPORTER=stdout`` opt-in mode.

Spec line 437: when the operator explicitly opts in via
``OTEL_EXPORTER=stdout``, ``setup_observability`` wires Console
exporters on all three providers (Tracer / Logger / Meter). Verifies
the registration only — no signal emission needed for the assertion;
emission would just write to local stdout (still zero outbound).
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


@pytest.fixture
def stdout_settings(monkeypatch):
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER", "stdout")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.otlp_endpoint == ""
    assert settings.otel_exporter == "stdout"
    yield settings
    get_settings.cache_clear()


def test_stdout_wires_console_span_exporter(stdout_settings):
    from opentelemetry import trace
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
    )

    p = setup_observability()
    # LOCK P1: the returned tracer_provider IS the OTel global so this
    # assertion exercises the runtime path, not just the in-memory handle.
    assert trace.get_tracer_provider() is p.tracer_provider
    sp = p.tracer_provider._active_span_processor
    registered = list(getattr(sp, "_span_processors", ()))
    assert len(registered) == 1
    assert isinstance(registered[0], BatchSpanProcessor)
    assert isinstance(registered[0].span_exporter, ConsoleSpanExporter)


def test_stdout_wires_console_log_exporter(stdout_settings):
    from opentelemetry.sdk._logs.export import (
        BatchLogRecordProcessor,
        ConsoleLogExporter,
    )

    p = setup_observability()
    procs = list(
        p.logger_provider._multi_log_record_processor._log_record_processors
    )
    assert len(procs) == 1
    assert isinstance(procs[0], BatchLogRecordProcessor)
    # OTel 1.41 wraps the exporter inside the inner ``_batch_processor``;
    # peek through that layer to assert the exporter type.
    assert isinstance(
        procs[0]._batch_processor._exporter, ConsoleLogExporter
    )


def test_stdout_wires_console_metric_reader(stdout_settings):
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    p = setup_observability()
    readers = list(p.meter_provider._sdk_config.metric_readers)
    assert len(readers) == 1
    assert isinstance(readers[0], PeriodicExportingMetricReader)
    assert isinstance(readers[0]._exporter, ConsoleMetricExporter)
