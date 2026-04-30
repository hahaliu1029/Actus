"""B5 PR-S2-1 acceptance: OTLP path is reserved for Sprint 3.

Spec line 707: Sprint 2 dep set introduces ``opentelemetry-api`` /
``opentelemetry-sdk`` / ``opentelemetry-instrumentation-fastapi`` only
— **not** ``opentelemetry-exporter-otlp``. Calling
``setup_observability`` with ``OTLP_ENDPOINT=...`` set in Sprint 2
must raise ``NotImplementedError`` with a clear message pointing at
PR-S3-1, rather than silently no-oping or attempting an import that
fails opaquely.
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


def test_otlp_endpoint_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("OTLP_ENDPOINT", "http://collector.example.com:4317")
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(NotImplementedError, match="Sprint 3 PR-S3-1"):
            setup_observability()
    finally:
        get_settings.cache_clear()


def test_unrecognised_otel_exporter_raises_value_error(monkeypatch):
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER", "jaeger")  # not in {"", "stdout"}
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="Sprint 2 mode"):
            setup_observability()
    finally:
        get_settings.cache_clear()
