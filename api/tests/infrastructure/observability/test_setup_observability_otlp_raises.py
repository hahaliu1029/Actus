"""B5 PR-S3-1: ``setup_observability`` validation surface.

PR-S2-1 reserved the OTLP path with a ``NotImplementedError``. PR-S3-1
implements it: ``setup_observability`` accepts ``OTLP_ENDPOINT`` and
wires OTLP exporters through ``BatchSpanProcessor`` /
``BatchLogRecordProcessor`` / ``PeriodicExportingMetricReader`` per
``OTLP_PROTOCOL`` (``"http/protobuf"`` default, or ``"grpc"``).

This file pins the **validation** surface — ``ValueError`` paths only.
End-to-end OTLP wiring (exporter shape, processor presence) lives in
``test_setup_observability_otlp_mode.py``.
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


def test_unrecognised_otel_exporter_raises_value_error(monkeypatch):
    """``OTEL_EXPORTER`` outside {"", "stdout"} → ``ValueError``."""
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER", "jaeger")
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="not a recognised"):
            setup_observability()
    finally:
        get_settings.cache_clear()


def test_unrecognised_otlp_protocol_raises_value_error(monkeypatch):
    """When ``OTLP_ENDPOINT`` is set, ``OTLP_PROTOCOL`` must be either
    ``"http/protobuf"`` or ``"grpc"``. Anything else is rejected at
    validate time so a misconfigured deployment fails fast at startup
    rather than silently dropping spans.
    """
    monkeypatch.setenv("OTLP_ENDPOINT", "http://collector.example.com:4318")
    monkeypatch.setenv("OTLP_PROTOCOL", "thrift")
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="OTLP transport"):
            setup_observability()
    finally:
        get_settings.cache_clear()
