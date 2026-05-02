"""B5 PR-S2-1 / PR-S3-1: OTel SDK bootstrap.

Idempotent setup that installs the three OTel providers
(``TracerProvider`` / ``LoggerProvider`` / ``MeterProvider``) per
``Settings.otlp_endpoint`` / ``Settings.otel_exporter`` /
``Settings.otlp_protocol``:

- **Default** (all empty) — install providers with **NO exporters**.
  Signals are recorded in-process and dropped at provider shutdown.
  Zero outbound traffic, zero stdout pollution. This is the spec's
  ``OTLP_ENDPOINT="" = no-op exporter`` mode (spec line 436), verified
  by ``test_setup_observability_no_outbound_default.py``.
- ``OTEL_EXPORTER=stdout`` — install Console exporters on all three
  providers. Dev / debugging mode; local-only output.
- ``OTLP_ENDPOINT=http://...`` (PR-S3-1) — install OTLP exporters on
  all three providers (``BatchSpanProcessor`` /
  ``BatchLogRecordProcessor`` / ``PeriodicExportingMetricReader``
  each wrapping an OTLP variant). Protocol selected by
  ``OTLP_PROTOCOL``: ``"http/protobuf"`` (default, Phoenix-compat,
  port 4318) or ``"grpc"`` (Collector default, port 4317).
  Backend-agnostic — Phoenix / Jaeger / Loki+Prometheus all consume
  OTLP.

Logger bridge (spec line 432 — "嵌入 RedactingFormatter"): a stock OTel
``LoggingHandler`` is attached to the root logger with
``RedactingFormatter`` as its formatter and Sprint 1's
``_ComponentFilter`` (via ``attach_component_filter``) as its filter so
noisy third-party loggers (httpx / openai / langchain INFO+DEBUG) do
not bypass the noise suppression rules into the OTel pipeline. The
handler delegates body formatting to ``self.format(record)`` (which
runs RedactingFormatter, including ``formatException`` traceback
rendering), so secrets in traceback frames are scrubbed in OTel
``LogRecord.body`` identically to the stdout / file path.

Idempotency / replacement: ``setup_observability()`` is safe to call
multiple times — subsequent calls return the cached handle. Tests that
need a fresh global call ``teardown_observability()`` first; that
helper resets the OTel ``_*_PROVIDER_SET_ONCE`` gates AND the
underlying provider singletons so the next ``setup_observability()``
genuinely swaps globals (rather than the previous fake-``override``
that was rejected by OTel and silently kept the original provider).

The ``_*_PROVIDER_SET_ONCE`` reset uses OTel's private API but is the
official OTel-test pattern (the OTel SDK's own test suite does the
same). It is wrapped in ``getattr`` fallbacks so an OTel internal
rename does not crash — the worst-case is a regression on the test
swap path, which the locked-global tests would catch immediately.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Optional

from opentelemetry import _logs as otel_logs
from opentelemetry import metrics, trace
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogExporter,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.util._once import Once

# B5 PR-S3-1: OTLP exporter set. ``opentelemetry-exporter-otlp`` is a
# meta-package bundling both gRPC and HTTP/protobuf transports. Imports
# are deferred to ``_build_*_provider`` so the no-op / stdout modes
# don't pay an import-time price for unused transport stacks.

from app.infrastructure.logging import attach_component_filter
from app.infrastructure.logging.redaction import RedactingFormatter


@dataclass(frozen=True)
class ObservabilityProviders:
    """Handle returned by ``setup_observability`` for downstream wiring."""

    tracer_provider: TracerProvider
    logger_provider: LoggerProvider
    meter_provider: MeterProvider
    log_handler: LoggingHandler


_INIT_LOCK = threading.Lock()
_PROVIDERS: Optional[ObservabilityProviders] = None


_OTLP_HTTP_SIGNAL_SUFFIXES: tuple[str, ...] = (
    "/v1/traces",
    "/v1/logs",
    "/v1/metrics",
)


def _derive_http_endpoint(base: str, signal_suffix: str) -> str:
    """Append the OTLP/HTTP per-signal path to a base URL.

    OTel/HTTP spec routes each signal to a distinct path:
    ``/v1/traces`` / ``/v1/logs`` / ``/v1/metrics``. The Python SDK
    only auto-derives these paths when ``endpoint`` is **not** passed
    (it then falls back to ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var +
    suffix). When ``endpoint=`` is passed explicitly, the SDK uses
    that URL **as-is** — no suffix appended (see
    ``opentelemetry/exporter/otlp/proto/http/{trace,_log,metric}_exporter/__init__.py``
    line ``self._endpoint = endpoint or environ.get(...)``).

    Phoenix / Collector at ``http://localhost:4318`` only accept the
    per-signal paths — POSTing to ``/`` returns 404. So when we pass
    ``endpoint=`` explicitly (which we must, because we read it from
    Actus's own ``OTLP_ENDPOINT`` setting), the per-signal suffix is
    OUR responsibility to append.

    Contract: ``base`` MUST be the OTLP base URL (without any signal
    suffix). ``_validate_settings`` enforces this at startup —
    rejects ``OTLP_ENDPOINT`` ending with ``/v1/{traces,logs,metrics}``
    so a config like ``http://collector/v1/traces`` (which would only
    be correct for one signal and produce broken paths like
    ``/v1/traces/v1/logs`` for the other two) never reaches this
    helper.

    Behaviour: strip trailing ``/`` from base, then append
    ``signal_suffix`` (always starts with ``/``).
    """
    return base.rstrip("/") + signal_suffix


def _build_otlp_span_exporter(endpoint: str, protocol: str) -> Any:
    """Construct an OTLP span exporter for the requested transport.

    ``http/protobuf`` (default, Phoenix-compat, port 4318) and ``grpc``
    (Collector default, port 4317) are the canonical OTLP transports.
    Imports happen lazily so no-op / stdout deployments don't pay the
    transport-stack import cost.

    HTTP path: appends ``/v1/traces`` to the base URL (see
    ``_derive_http_endpoint``). gRPC path: gRPC uses ``host:port``
    routing internally, no path suffix needed.
    """
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as _GrpcSpan,
        )
        return _GrpcSpan(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as _HttpSpan,
    )
    return _HttpSpan(endpoint=_derive_http_endpoint(endpoint, "/v1/traces"))


def _build_otlp_log_exporter(endpoint: str, protocol: str) -> Any:
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
            OTLPLogExporter as _GrpcLog,
        )
        return _GrpcLog(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.http._log_exporter import (
        OTLPLogExporter as _HttpLog,
    )
    return _HttpLog(endpoint=_derive_http_endpoint(endpoint, "/v1/logs"))


def _build_otlp_metric_exporter(endpoint: str, protocol: str) -> Any:
    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter as _GrpcMetric,
        )
        return _GrpcMetric(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter as _HttpMetric,
    )
    return _HttpMetric(
        endpoint=_derive_http_endpoint(endpoint, "/v1/metrics")
    )


def _build_prometheus_metric_reader() -> Any:
    """Construct a ``PrometheusMetricReader`` for in-process scrape.

    Pull-based reader — sits in-memory only. The reader self-registers
    a ``_CustomCollector`` with ``prometheus_client.REGISTRY`` at
    construction; ``provider.shutdown()`` cascades to
    ``reader.shutdown()`` which unregisters, so test teardown leaves
    the global REGISTRY clean.

    Lazy import keeps no-op / stdout / OTLP-only deployments from
    paying the ``prometheus_client`` import cost (~600KB resident).
    """
    from opentelemetry.exporter.prometheus import PrometheusMetricReader

    return PrometheusMetricReader()


def _build_tracer_provider(
    otel_exporter: str,
    otlp_endpoint: str = "",
    otlp_protocol: str = "http/protobuf",
) -> TracerProvider:
    provider = TracerProvider()
    if otlp_endpoint:
        # OTLP wins — same processor type (BatchSpanProcessor) so the
        # in-process behaviour matches stdout mode (batching, flush
        # on shutdown), only the export destination changes.
        provider.add_span_processor(
            BatchSpanProcessor(
                _build_otlp_span_exporter(otlp_endpoint, otlp_protocol)
            )
        )
    elif otel_exporter == "stdout":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    return provider


def _build_logger_provider(
    otel_exporter: str,
    otlp_endpoint: str = "",
    otlp_protocol: str = "http/protobuf",
) -> LoggerProvider:
    provider = LoggerProvider()
    if otlp_endpoint:
        provider.add_log_record_processor(
            BatchLogRecordProcessor(
                _build_otlp_log_exporter(otlp_endpoint, otlp_protocol)
            )
        )
    elif otel_exporter == "stdout":
        provider.add_log_record_processor(
            BatchLogRecordProcessor(ConsoleLogExporter())
        )
    return provider


def _build_meter_provider(
    otel_exporter: str,
    otlp_endpoint: str = "",
    otlp_protocol: str = "http/protobuf",
    metrics_endpoint_token: str = "",
) -> MeterProvider:
    """Construct a ``MeterProvider`` with the requested readers.

    Reader composition is additive — a deployment can run OTLP push
    AND Prometheus pull simultaneously (e.g., push to a central
    Phoenix instance for trace-correlated metric review while a local
    Prometheus / VictoriaMetrics scrapes the per-instance endpoint
    for alerting). The OTel SDK supports multiple readers on a single
    ``MeterProvider``.

    PR-S3-3: ``metrics_endpoint_token`` non-empty appends a
    ``PrometheusMetricReader`` so ``GET /api/v1/metrics`` can serve
    Prometheus exposition format. Empty token = no reader (zero
    memory + zero registration with ``prometheus_client.REGISTRY``).
    """
    readers: list[Any] = []
    if otlp_endpoint:
        readers.append(
            PeriodicExportingMetricReader(
                _build_otlp_metric_exporter(otlp_endpoint, otlp_protocol)
            )
        )
    elif otel_exporter == "stdout":
        readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
    if metrics_endpoint_token:
        readers.append(_build_prometheus_metric_reader())
    return MeterProvider(metric_readers=readers)


_FACTORY_PLACEHOLDER_VALUE = "-"
_FACTORY_PLACEHOLDER_KEYS: frozenset[str] = frozenset(
    ("trace_id", "request_id", "session_id")
)


class _PlaceholderStrippingLoggingHandler(LoggingHandler):
    """OTel ``LoggingHandler`` that drops PR-S1-4's ``"-"`` placeholder.

    The PR-S1-4 ``_actus_log_record_factory`` pins ``"-"`` onto
    ``trace_id`` / ``request_id`` / ``session_id`` whenever a
    ``TraceContext`` is missing or its corresponding field is ``None``,
    so stdlib ``%(trace_id)s`` format strings stay non-crashing. That
    placeholder is not real data — surfacing it as an OTel attribute
    is semantically wrong:

    - For required fields (``trace_id`` / ``request_id``), ``"-"``
      fails ``validate_attributes`` v1 format gate.
    - For nullable ``session_id``, the OTel-side answer to "no session
      bound" is "attribute absent", not ``"-"``.

    Override ``_get_attributes`` so the OTel pipeline sees attribute-
    absent for placeholder values. Stdlib stdout / file handlers still
    see ``"-"`` (their format strings depend on it). ``OtelLogger``'s
    synthetic ``TraceContext`` bind already keeps ``trace_id`` /
    ``request_id`` valid for the no-context emit path; this subclass
    is the second line of defence + handles the request-without-
    session scenario where middleware binds ``ctx`` with
    ``session_id=None``.
    """

    def _get_attributes(self, record: logging.LogRecord) -> dict[str, Any]:
        attrs = LoggingHandler._get_attributes(record)
        for key in _FACTORY_PLACEHOLDER_KEYS:
            if attrs.get(key) == _FACTORY_PLACEHOLDER_VALUE:
                attrs.pop(key, None)
        return attrs


def _attach_log_handler(logger_provider: LoggerProvider) -> LoggingHandler:
    """Wire OTel ``LoggingHandler`` onto the root logger with redaction.

    The OTel SDK's stock ``LoggingHandler._translate`` calls
    ``self.format(record)`` when a formatter is attached and falls back
    to ``record.getMessage()`` otherwise. Setting ``RedactingFormatter``
    here ensures the OTel ``LogRecord.body`` is the redacted full
    rendered string (including ``formatException`` traceback content) —
    secrets in tracebacks cannot leak through the OTel pipeline.

    The handler also gets Sprint 1's ``_ComponentFilter`` via
    ``attach_component_filter`` so INFO/DEBUG records from noisy
    third-party loggers (``httpx`` / ``openai`` / ``langchain`` / ...)
    are dropped before reaching the OTel pipeline — matching the
    behaviour of stdout / file handlers and preventing OTLP exporter
    cost amplification when stdout / OTLP modes are enabled later.

    Uses ``_PlaceholderStrippingLoggingHandler`` (subclass) so the
    factory ``"-"`` placeholder for missing trace_id / request_id /
    session_id never surfaces as an OTel attribute.
    """
    handler = _PlaceholderStrippingLoggingHandler(
        level=logging.NOTSET, logger_provider=logger_provider
    )
    handler.setFormatter(RedactingFormatter())
    attach_component_filter(handler)
    logging.getLogger().addHandler(handler)
    return handler


def _reset_otel_globals() -> None:
    """Reset the three OTel ``_*_PROVIDER_SET_ONCE`` gates + singletons.

    OTel's ``set_*_provider`` honours a one-shot gate — the second call
    logs ``Overriding of current TracerProvider is not allowed`` and
    silently keeps the first provider. For test-side global swaps this
    is fatal (the test asserts on the freshly-installed handle while
    the global stays stale). This helper, modelled on the OTel SDK's
    own test suite pattern, replaces each ``Once`` with a fresh
    instance and clears the cached provider singleton so the next
    ``set_*_provider`` call is honoured.

    Uses private OTel symbols. Wrapped in ``getattr`` fallbacks so a
    future OTel rename surfaces as a localized regression in the
    swap-path tests rather than a crash.
    """
    # Trace
    if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
        trace._TRACER_PROVIDER_SET_ONCE = Once()
    if hasattr(trace, "_TRACER_PROVIDER"):
        trace._TRACER_PROVIDER = None
    # Logs
    logs_internal = getattr(otel_logs, "_internal", None)
    if logs_internal is not None:
        if hasattr(logs_internal, "_LOGGER_PROVIDER_SET_ONCE"):
            logs_internal._LOGGER_PROVIDER_SET_ONCE = Once()
        if hasattr(logs_internal, "_LOGGER_PROVIDER"):
            logs_internal._LOGGER_PROVIDER = None
    # Metrics
    metrics_internal = getattr(metrics, "_internal", None)
    if metrics_internal is not None:
        if hasattr(metrics_internal, "_METER_PROVIDER_SET_ONCE"):
            metrics_internal._METER_PROVIDER_SET_ONCE = Once()
        if hasattr(metrics_internal, "_METER_PROVIDER"):
            metrics_internal._METER_PROVIDER = None


def _detach_log_handler(handler: LoggingHandler) -> None:
    try:
        logging.getLogger().removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass


def _validate_settings(
    otlp_endpoint: str,
    otel_exporter: str,
    otlp_protocol: str = "http/protobuf",
) -> tuple[str, str, str]:
    """Pure validation of observability settings.

    Returns ``(otel_exporter_normalised, otlp_endpoint_stripped,
    otlp_protocol_normalised)``. Raises ``ValueError`` for an
    unrecognised ``OTEL_EXPORTER`` or ``OTLP_PROTOCOL`` value. Pulled
    out as a free function so the validation paths can be tested
    without touching OTel global state.

    Mode resolution rules:
    - ``otlp_endpoint`` non-empty → OTLP mode (overrides
      ``otel_exporter``); ``otlp_protocol`` decides transport.
    - ``otel_exporter == "stdout"`` → Console exporters.
    - Both empty → no-op (providers without exporters).
    """
    endpoint_stripped = otlp_endpoint.strip()
    exporter_normalised = otel_exporter.strip().lower()
    protocol_normalised = (otlp_protocol or "").strip().lower()
    if exporter_normalised not in ("", "stdout"):
        raise ValueError(
            f"OTEL_EXPORTER={exporter_normalised!r} is not a recognised "
            f"mode. Use '' (no-op) or 'stdout' (dev console)."
        )
    if endpoint_stripped and protocol_normalised not in (
        "",
        "http/protobuf",
        "grpc",
    ):
        raise ValueError(
            f"OTLP_PROTOCOL={protocol_normalised!r} is not a recognised "
            f"OTLP transport. Use 'http/protobuf' (default, Phoenix) "
            f"or 'grpc' (Collector default)."
        )
    if endpoint_stripped and not protocol_normalised:
        protocol_normalised = "http/protobuf"
    # Reject signal-suffixed aggregate endpoints. ``OTLP_ENDPOINT`` is a
    # SINGLE base URL — the three signal builders each append their own
    # ``/v1/{traces,logs,metrics}``. Pre-suffixing would produce
    # ``.../v1/traces/v1/logs`` for two of the three signals (only the
    # matching signal would be correct). Fail-fast so operators see a
    # clear error instead of debugging silent path-doubling at the
    # exporter layer. Per-signal endpoints belong on the OTel-native
    # env vars (``OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`` etc.), which
    # Actus does not currently surface.
    if endpoint_stripped and any(
        endpoint_stripped.rstrip("/").endswith(s)
        for s in _OTLP_HTTP_SIGNAL_SUFFIXES
    ):
        raise ValueError(
            f"OTLP_ENDPOINT={endpoint_stripped!r} ends with a per-signal "
            f"path (one of {list(_OTLP_HTTP_SIGNAL_SUFFIXES)}). Set the "
            f"BASE URL only (e.g. 'http://localhost:4318') — Actus "
            f"appends '/v1/traces' / '/v1/logs' / '/v1/metrics' per "
            f"signal automatically."
        )
    return exporter_normalised, endpoint_stripped, protocol_normalised


def setup_observability() -> ObservabilityProviders:
    """Install OTel providers per ``core.config.Settings``. Idempotent.

    First call installs providers and globalizes them. Subsequent calls
    return the cached handle without touching globals. Tests that need
    a fresh global must call ``teardown_observability()`` first — that
    helper resets OTel's set-once gates so the next install actually
    swaps globals (rather than the previous fake-``override`` that was
    silently rejected by OTel).

    Returns:
        ``ObservabilityProviders`` handle. The contained providers are
        identical to ``trace.get_tracer_provider()`` /
        ``otel_logs.get_logger_provider()`` /
        ``metrics.get_meter_provider()`` (locked by the no-outbound and
        idempotent test suites).

    Raises:
        ValueError: ``Settings.otel_exporter`` is not in {"", "stdout"},
            or ``Settings.otlp_protocol`` (when ``otlp_endpoint`` is
            set) is not in {"http/protobuf", "grpc"}.
    """
    global _PROVIDERS
    with _INIT_LOCK:
        if _PROVIDERS is not None:
            # Defend against ``setup_logging()`` rerunning between the
            # first and second ``setup_observability()`` call —
            # ``_install_redacting_formatter`` removes AND closes every
            # root handler (including ours), so the cached
            # ``log_handler`` may already be detached and unusable.
            # Reattach a fresh handler bound to the SAME
            # ``LoggerProvider`` (the provider survives — only root
            # handlers are touched) and refresh the cached handle so
            # later callers see the live one. Tracer / Meter providers
            # are unaffected and stay as-is.
            if _PROVIDERS.log_handler in logging.getLogger().handlers:
                return _PROVIDERS
            new_handler = _attach_log_handler(_PROVIDERS.logger_provider)
            _PROVIDERS = ObservabilityProviders(
                tracer_provider=_PROVIDERS.tracer_provider,
                logger_provider=_PROVIDERS.logger_provider,
                meter_provider=_PROVIDERS.meter_provider,
                log_handler=new_handler,
            )
            return _PROVIDERS

        from core.config import get_settings

        settings = get_settings()
        otel_exporter, otlp_endpoint, otlp_protocol = _validate_settings(
            settings.otlp_endpoint,
            settings.otel_exporter,
            settings.otlp_protocol,
        )

        tracer_provider = _build_tracer_provider(
            otel_exporter, otlp_endpoint, otlp_protocol
        )
        logger_provider = _build_logger_provider(
            otel_exporter, otlp_endpoint, otlp_protocol
        )
        meter_provider = _build_meter_provider(
            otel_exporter,
            otlp_endpoint,
            otlp_protocol,
            metrics_endpoint_token=settings.metrics_endpoint_token,
        )

        trace.set_tracer_provider(tracer_provider)
        otel_logs.set_logger_provider(logger_provider)
        metrics.set_meter_provider(meter_provider)

        log_handler = _attach_log_handler(logger_provider)

        _PROVIDERS = ObservabilityProviders(
            tracer_provider=tracer_provider,
            logger_provider=logger_provider,
            meter_provider=meter_provider,
            log_handler=log_handler,
        )
        return _PROVIDERS


def teardown_observability() -> None:
    """Reset module + OTel global state. Test-only.

    Detaches the OTel ``LoggingHandler`` from root, **shuts down** the
    cached tracer / logger / meter providers (flush + stop background
    batch exporters, periodic metric reader threads, ...), clears the
    cached ``_PROVIDERS`` handle, and resets OTel's three
    ``_*_PROVIDER_SET_ONCE`` gates plus the underlying provider
    singletons so the NEXT ``setup_observability()`` call genuinely
    installs a fresh global (rather than getting silently rejected by
    OTel's "first call wins" gate).

    Without the ``.shutdown()`` calls, ``OTEL_EXPORTER=stdout`` tests
    leave a ``PeriodicExportingMetricReader`` daemon thread running that
    flushes ConsoleMetricExporter JSON to stdout AFTER the pytest
    summary — noisy and confusing for downstream readers.

    Production callers should not invoke this — providers are intended
    to live for the process lifetime. Tests use it between cases that
    need a clean slate.
    """
    global _PROVIDERS
    with _INIT_LOCK:
        if _PROVIDERS is not None:
            _detach_log_handler(_PROVIDERS.log_handler)
            for provider in (
                _PROVIDERS.tracer_provider,
                _PROVIDERS.logger_provider,
                _PROVIDERS.meter_provider,
            ):
                try:
                    provider.shutdown()
                except Exception:
                    # Best-effort shutdown — a hung exporter must not
                    # block the rest of teardown (e.g., next test).
                    pass
            _PROVIDERS = None
        _reset_otel_globals()


def get_providers() -> Optional[ObservabilityProviders]:
    """Return the installed providers handle, or ``None`` if not set up."""
    return _PROVIDERS
