"""B5 PR-S2-1 acceptance: default install has zero exporters.

Spec line 436 + line 585 lock the invariant: with ``OTLP_ENDPOINT=""``
and ``OTEL_EXPORTER=""`` (the production default), ``setup_observability``
installs the three OTel providers but wires ZERO exporters / processors
/ metric readers — nothing leaves the process. This guards against an
accidental bring-up of stdout / OTLP exporters that would (a) leak data
to the local journal in self-hosted deployments or (b) attempt to
contact a non-existent OTel collector at startup.

Verified via direct provider-state introspection rather than network
mocking: counting the registered processors is a stronger invariant
than asserting "no socket fired this round" — it proves intent.
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
def _force_default_settings(monkeypatch):
    """Force defaults regardless of host env.

    pydantic-settings reads ``OTLP_ENDPOINT`` / ``OTEL_EXPORTER`` env
    vars at ``Settings()`` construction time; ``get_settings`` is
    ``lru_cache``d so we cannot mutate the cached instance directly.
    Clear the cache and re-prime with the env vars unset.
    """
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER", raising=False)
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.otlp_endpoint == ""
    assert settings.otel_exporter == ""
    yield
    get_settings.cache_clear()


def test_default_install_zero_span_processors():
    """TracerProvider has zero registered span processors."""
    p = setup_observability()
    sp = p.tracer_provider._active_span_processor
    registered = list(getattr(sp, "_span_processors", ()))
    assert registered == [], (
        f"default install should wire zero span processors; got {registered}"
    )


def test_default_install_zero_log_record_processors():
    """LoggerProvider has zero registered log record processors."""
    p = setup_observability()
    procs = list(
        p.logger_provider._multi_log_record_processor._log_record_processors
    )
    assert procs == [], (
        f"default install should wire zero log processors; got {procs}"
    )


def test_default_install_zero_metric_readers():
    """MeterProvider has zero registered metric readers."""
    p = setup_observability()
    readers = list(p.meter_provider._sdk_config.metric_readers)
    assert readers == [], (
        f"default install should wire zero metric readers; got {readers}"
    )


def test_default_install_attaches_log_handler_to_root():
    """OTel LoggingHandler IS attached even at default settings.

    The handler is harmless when LoggerProvider has no processors —
    records are formatted, redacted, dispatched to a no-op pipeline,
    and dropped. Attaching it unconditionally simplifies the toggle
    semantics for ``OTEL_EXPORTER=stdout`` opt-in (no need to also
    rebind the handler).
    """
    import logging

    p = setup_observability()
    root_handlers = logging.getLogger().handlers
    assert p.log_handler in root_handlers, (
        "OTel LoggingHandler must be attached to root logger"
    )


def test_returned_handle_matches_otel_globals():
    """LOCK P1: returned handle IS the OTel global provider.

    Reviewer found the previous ``override=True`` semantics silently
    drifted from the OTel global because ``set_*_provider`` rejected
    second calls. ``teardown_observability()`` now resets OTel's
    set-once gates so a fresh ``setup_observability()`` truly swaps
    globals; this test locks that the returned handle and the global
    are the same object — assertions on the handle become assertions
    on the runtime providers.
    """
    from opentelemetry import _logs as otel_logs
    from opentelemetry import metrics, trace

    p = setup_observability()
    assert trace.get_tracer_provider() is p.tracer_provider
    assert otel_logs.get_logger_provider() is p.logger_provider
    assert metrics.get_meter_provider() is p.meter_provider


def test_otel_log_handler_carries_component_filter():
    """LOCK P2: OTel LoggingHandler has Sprint 1's noise filter.

    Without this, INFO/DEBUG records from ``httpx`` / ``openai`` /
    ``langchain`` etc. would be filtered on stdout / file handlers but
    flood the OTel pipeline (cost amplification + signal noise).
    """
    from app.infrastructure.logging.logging import _ComponentFilter

    p = setup_observability()
    has_component_filter = any(
        isinstance(f, _ComponentFilter) for f in p.log_handler.filters
    )
    assert has_component_filter, (
        "OTel LoggingHandler must carry _ComponentFilter to suppress "
        "INFO/DEBUG from noisy third-party loggers"
    )
