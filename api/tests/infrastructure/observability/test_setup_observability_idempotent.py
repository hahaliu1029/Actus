"""B5 PR-S2-1 acceptance: ``setup_observability`` is idempotent.

Repeating the call must:

- Return the same providers handle (so module-level cache wins).
- Not duplicate the OTel ``LoggingHandler`` on the root logger
  (otherwise every record would emit twice on the OTel side, which
  inflates downstream costs and breaks "exactly-once" semantics
  expected by Phoenix / OTLP collectors).

The test does not exercise ``override=True`` outside of fixture setup
— that path is for test fixtures that need a fresh handle and accepts
the OTel "Overriding of current TracerProvider is not allowed" warning
as the price of isolation. Production callers (``app/main.py``) MUST
take the default path.
"""
from __future__ import annotations

import logging

import pytest

from app.infrastructure.observability import (
    setup_observability,
    teardown_observability,
)


@pytest.fixture(autouse=True)
def _reset_provider_state():
    teardown_observability()
    yield
    teardown_observability()


def test_repeat_call_returns_same_handle():
    p1 = setup_observability()
    p2 = setup_observability()
    assert p1 is p2, "second call must return the cached providers handle"


def test_repeat_call_does_not_duplicate_log_handler():
    p1 = setup_observability()
    root = logging.getLogger()
    count_before = sum(1 for h in root.handlers if h is p1.log_handler)
    assert count_before == 1

    setup_observability()
    setup_observability()
    setup_observability()

    count_after = sum(1 for h in root.handlers if h is p1.log_handler)
    assert count_after == 1, (
        "OTel LoggingHandler must NOT be re-attached on subsequent calls"
    )


def test_teardown_then_setup_swaps_in_fresh_handle():
    """Teardown resets OTel set-once gates so the next setup truly swaps.

    Reviewer P1: previous ``override=True`` was a fiction — the OTel
    global was rejected on the second ``set_tracer_provider`` call and
    silently kept the original. This test now validates the fixed
    contract: teardown clears OTel global gates, next setup installs a
    genuinely new global, and the OTel runtime ``get_*_provider`` APIs
    point at the new handle.
    """
    from opentelemetry import _logs as otel_logs
    from opentelemetry import metrics, trace

    p1 = setup_observability()
    teardown_observability()
    p2 = setup_observability()

    assert p1 is not p2, "fresh handle expected after teardown"
    assert trace.get_tracer_provider() is p2.tracer_provider, (
        "OTel global tracer_provider must match the freshly-installed handle"
    )
    assert otel_logs.get_logger_provider() is p2.logger_provider
    assert metrics.get_meter_provider() is p2.meter_provider
    # Old handler off root; new handler on root.
    root_handlers = logging.getLogger().handlers
    assert p1.log_handler not in root_handlers
    assert p2.log_handler in root_handlers


def test_setup_logging_rerun_reattaches_otel_handler():
    """LOCK: ``setup_logging()`` rerun must not silently kill the OTel sink.

    ``_install_redacting_formatter`` removes AND closes every root
    handler (including ours). Without re-attach logic in
    ``setup_observability``'s cached-return branch, the second call
    after a logging rerun would happily return the cached handle whose
    ``log_handler`` is detached and closed — OTel log export silently
    dies in the process for any code that calls ``setup_logging``
    twice (uvicorn reload, test fixture, late reconfig).
    """
    from app.infrastructure.logging import setup_logging
    from opentelemetry import _logs as otel_logs
    from opentelemetry import metrics, trace

    p1 = setup_observability()
    root = logging.getLogger()
    assert p1.log_handler in root.handlers

    # setup_logging wipes root handlers (including OTel's).
    setup_logging()
    assert p1.log_handler not in root.handlers, (
        "precondition: setup_logging must detach old root handlers"
    )

    # setup_observability sees the cache + missing handler → rebuilds.
    p2 = setup_observability()
    # Provider singletons unchanged (LoggerProvider survived).
    assert p2.tracer_provider is p1.tracer_provider
    assert p2.logger_provider is p1.logger_provider
    assert p2.meter_provider is p1.meter_provider
    # Same OTel globals still bound (no new register).
    assert trace.get_tracer_provider() is p1.tracer_provider
    assert otel_logs.get_logger_provider() is p1.logger_provider
    assert metrics.get_meter_provider() is p1.meter_provider
    # Fresh handler attached to root, replacing the closed one.
    assert p2.log_handler is not p1.log_handler
    assert p2.log_handler in root.handlers
    assert metrics.get_meter_provider() is p2.meter_provider
    # Old handler off root; new handler on root.
    root_handlers = logging.getLogger().handlers
    assert p1.log_handler not in root_handlers
    assert p2.log_handler in root_handlers
