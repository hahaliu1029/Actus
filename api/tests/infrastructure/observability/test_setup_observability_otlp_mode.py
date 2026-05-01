"""B5 PR-S3-1: OTLP exporter wiring positive path.

When ``OTLP_ENDPOINT`` is set, ``setup_observability`` must:

- Wire ``BatchSpanProcessor`` containing an OTLP span exporter onto
  the ``TracerProvider``.
- Wire ``BatchLogRecordProcessor`` containing an OTLP log exporter
  onto the ``LoggerProvider``.
- Wire ``PeriodicExportingMetricReader`` containing an OTLP metric
  exporter onto the ``MeterProvider``.

Both transports are validated:

- ``OTLP_PROTOCOL=http/protobuf`` (default) → HTTP/protobuf exporters.
  Phoenix-compatible (port 4318).
- ``OTLP_PROTOCOL=grpc`` → gRPC exporters. OpenTelemetry Collector
  default (port 4317).

We don't actually open network sockets — the exporters are
constructed but not exercised. Endpoint URLs use ``http://localhost``
sentinels so no real outbound traffic. The tests assert structural
shape (exporter class names + module paths) without exercising
network calls.
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


def _enable_otlp(monkeypatch, *, protocol: str = "http/protobuf") -> None:
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTLP_PROTOCOL", protocol)
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()


def _span_exporters(provider) -> list:
    """Pull span exporter instances attached to ``provider``."""
    out: list = []
    multi = getattr(provider, "_active_span_processor", None)
    processors = getattr(multi, "_span_processors", ()) or ()
    for proc in processors:
        exp = getattr(proc, "span_exporter", None) or getattr(
            proc, "_span_exporter", None
        )
        if exp is not None:
            out.append(exp)
    return out


def _log_exporters(provider) -> list:
    out: list = []
    multi = getattr(provider, "_multi_log_record_processor", None)
    processors = getattr(multi, "_log_record_processors", ()) or ()
    for proc in processors:
        bp = getattr(proc, "_batch_processor", None)
        exp = (
            getattr(bp, "_exporter", None)
            if bp is not None
            else getattr(proc, "_exporter", None)
        )
        if exp is not None:
            out.append(exp)
    return out


def _metric_exporters(provider) -> list:
    out: list = []
    readers = getattr(
        getattr(provider, "_sdk_config", None), "metric_readers", ()
    )
    for reader in readers:
        exp = getattr(reader, "_exporter", None)
        if exp is not None:
            out.append(exp)
    return out


def test_otlp_http_protobuf_wires_http_exporters(monkeypatch):
    """Default ``OTLP_PROTOCOL=http/protobuf`` → HTTP exporters across
    all 3 signals (tracer / logger / meter).
    """
    _enable_otlp(monkeypatch, protocol="http/protobuf")
    try:
        providers = setup_observability()

        span_exps = _span_exporters(providers.tracer_provider)
        assert any(
            type(e).__name__ == "OTLPSpanExporter"
            and "proto.http" in type(e).__module__
            for e in span_exps
        ), (
            f"expected an HTTP OTLPSpanExporter; got "
            f"{[(type(e).__name__, type(e).__module__) for e in span_exps]!r}"
        )

        log_exps = _log_exporters(providers.logger_provider)
        assert any(
            type(e).__name__ == "OTLPLogExporter"
            and "proto.http" in type(e).__module__
            for e in log_exps
        )

        metric_exps = _metric_exporters(providers.meter_provider)
        assert any(
            type(e).__name__ == "OTLPMetricExporter"
            and "proto.http" in type(e).__module__
            for e in metric_exps
        )
    finally:
        get_settings.cache_clear()


def test_otlp_grpc_wires_grpc_exporters(monkeypatch):
    """``OTLP_PROTOCOL=grpc`` → gRPC exporters across all 3 signals."""
    _enable_otlp(monkeypatch, protocol="grpc")
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        # The grpc OTLP module path is ``...proto.grpc.*``. Class names
        # are the same (``OTLPSpanExporter``) — module reveals transport.
        assert any(
            "proto.grpc" in type(e).__module__ for e in span_exps
        ), (
            f"expected gRPC OTLP span exporter; got "
            f"{[(type(e).__name__, type(e).__module__) for e in span_exps]!r}"
        )

        log_exps = _log_exporters(providers.logger_provider)
        assert any("proto.grpc" in type(e).__module__ for e in log_exps)

        metric_exps = _metric_exporters(providers.meter_provider)
        assert any(
            "proto.grpc" in type(e).__module__ for e in metric_exps
        )
    finally:
        get_settings.cache_clear()


def test_otlp_default_protocol_when_endpoint_set_without_protocol(monkeypatch):
    """``OTLP_ENDPOINT`` set + no ``OTLP_PROTOCOL`` → defaults to
    ``http/protobuf`` (Phoenix-compat).
    """
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.delenv("OTLP_PROTOCOL", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        assert any(
            type(e).__name__ == "OTLPSpanExporter"
            and "proto.http" in type(e).__module__
            for e in span_exps
        )
    finally:
        get_settings.cache_clear()


def test_otlp_overrides_otel_exporter_stdout(monkeypatch):
    """OTLP_ENDPOINT precedence: even with ``OTEL_EXPORTER=stdout``,
    OTLP wins (production deployments override dev stdout via env var
    without changing other config).
    """
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_EXPORTER", "stdout")
    monkeypatch.setenv("OTLP_PROTOCOL", "http/protobuf")
    get_settings.cache_clear()
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        # OTLP exporter is present; ConsoleSpanExporter is NOT.
        assert any(
            type(e).__name__ == "OTLPSpanExporter" for e in span_exps
        )
        assert not any(
            type(e).__name__ == "ConsoleSpanExporter" for e in span_exps
        )
    finally:
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# B5 PR-S3-1 reviewer round 2: per-signal HTTP path lock.
#
# Reviewer probe found that passing ``OTLP_ENDPOINT=http://localhost:4318``
# without the per-signal suffix made the OTel HTTP exporter POST to ``/``
# (root path) — Phoenix / Collector return 404, three signals silently
# drop. Class-name + module-path assertions don't catch this because the
# exporter classes are correct; only the resolved ``_endpoint`` URL
# reveals the bug. These tests assert the URL.
# ---------------------------------------------------------------------------


def test_otlp_http_appends_v1_traces_to_base_url(monkeypatch):
    """Base ``http://localhost:4318`` → span exporter posts to
    ``http://localhost:4318/v1/traces`` (Phoenix-compat).
    """
    _enable_otlp(monkeypatch, protocol="http/protobuf")
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        endpoints = [getattr(e, "_endpoint", None) for e in span_exps]
        assert "http://localhost:4318/v1/traces" in endpoints, (
            f"OTLP HTTP span exporter must POST to /v1/traces; "
            f"resolved endpoints={endpoints!r}"
        )
    finally:
        get_settings.cache_clear()


def test_otlp_http_appends_v1_logs_to_base_url(monkeypatch):
    """Base ``http://localhost:4318`` → log exporter posts to
    ``http://localhost:4318/v1/logs``.
    """
    _enable_otlp(monkeypatch, protocol="http/protobuf")
    try:
        providers = setup_observability()
        log_exps = _log_exporters(providers.logger_provider)
        endpoints = [getattr(e, "_endpoint", None) for e in log_exps]
        assert "http://localhost:4318/v1/logs" in endpoints, (
            f"OTLP HTTP log exporter must POST to /v1/logs; "
            f"resolved endpoints={endpoints!r}"
        )
    finally:
        get_settings.cache_clear()


def test_otlp_http_appends_v1_metrics_to_base_url(monkeypatch):
    """Base ``http://localhost:4318`` → metric exporter posts to
    ``http://localhost:4318/v1/metrics``.
    """
    _enable_otlp(monkeypatch, protocol="http/protobuf")
    try:
        providers = setup_observability()
        metric_exps = _metric_exporters(providers.meter_provider)
        endpoints = [getattr(e, "_endpoint", None) for e in metric_exps]
        assert "http://localhost:4318/v1/metrics" in endpoints, (
            f"OTLP HTTP metric exporter must POST to /v1/metrics; "
            f"resolved endpoints={endpoints!r}"
        )
    finally:
        get_settings.cache_clear()


def test_otlp_http_strips_trailing_slash_before_appending(monkeypatch):
    """Trailing slash on the base URL must not produce ``//v1/traces``.

    Common deployment pattern: operators paste the base URL with a
    trailing ``/`` (``http://localhost:4318/``). Without the strip
    the resulting endpoint would be ``http://localhost:4318//v1/traces``
    — some HTTP servers reject that as malformed.
    """
    monkeypatch.setenv("OTLP_ENDPOINT", "http://localhost:4318/")
    monkeypatch.setenv("OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        endpoints = [getattr(e, "_endpoint", None) for e in span_exps]
        assert "http://localhost:4318/v1/traces" in endpoints, (
            f"trailing slash must be stripped; got {endpoints!r}"
        )
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize(
    "suffix", ["/v1/traces", "/v1/logs", "/v1/metrics"]
)
def test_otlp_endpoint_with_signal_suffix_rejected(monkeypatch, suffix):
    """Round-3 P3 fix: ``OTLP_ENDPOINT`` is a SINGLE base URL — the
    three signal builders append their own ``/v1/{traces,logs,metrics}``.
    A pre-suffixed value like ``http://collector/v1/traces`` would
    produce ``.../v1/traces/v1/logs`` for the log signal (and same
    for metrics) — only the matching signal would be correct, the
    other two silently broken.

    Reject at validation time so operators see a clear error instead
    of debugging path-doubling at the exporter layer. Per-signal
    endpoints belong on the OTel-native env vars
    (``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` etc.), which Actus does
    not currently surface as Settings fields.
    """
    monkeypatch.setenv(
        "OTLP_ENDPOINT", f"http://collector.example.com{suffix}"
    )
    monkeypatch.setenv("OTLP_PROTOCOL", "http/protobuf")
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    try:
        with pytest.raises(ValueError, match="per-signal path"):
            setup_observability()
    finally:
        get_settings.cache_clear()


def test_otlp_grpc_endpoint_passes_through_unchanged(monkeypatch):
    """gRPC transport uses ``host:port`` routing — no HTTP path
    suffix. Endpoint must pass through verbatim.
    """
    _enable_otlp(monkeypatch, protocol="grpc")
    try:
        providers = setup_observability()
        span_exps = _span_exporters(providers.tracer_provider)
        # gRPC OTLP exporter exposes the raw endpoint on its private
        # ``_endpoint`` attribute. We assert no ``/v1/`` suffix was
        # injected (gRPC doesn't use HTTP paths, so injecting one
        # would break the connection).
        for e in span_exps:
            ep = getattr(e, "_endpoint", "")
            assert "/v1/traces" not in ep, (
                f"gRPC endpoint must not carry HTTP signal path; got {ep!r}"
            )
    finally:
        get_settings.cache_clear()
