"""B5 PR-S1-4 acceptance: ``_actus_log_record_factory`` never raises (Q6).

The factory injects ``trace_id`` / ``request_id`` / ``session_id`` on
every ``LogRecord`` from the request-scoped ``TraceContext`` carrier.
The Q6 contract is uncompromising: even when the carrier itself is
broken (boot-time circular import, contextvar storage corrupted, lib
intentionally raises a ``RuntimeError``), the factory MUST return a
valid ``LogRecord`` with the three Q6 attrs populated. The fallback
value is the literal string ``"-"`` (never ``None``, never missing
attribute) so formatters using ``%(trace_id)s`` produce ``"-"``
instead of either ``"None"`` or an ``AttributeError`` blow-up.

These tests pin every escape route:

- ``get_trace_context`` returns ``None`` (cold path, no request scope)
- ``get_trace_context`` returns a valid ``TraceContext``
- ``get_trace_context`` returns a context whose ``trace_id`` is None
- ``get_trace_context`` raises ``RuntimeError`` (carrier broken)
- ``get_trace_context`` raises ``ImportError`` (boot-time import loop)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pytest

import app.infrastructure.observability.context as ctx_module
from app.infrastructure.logging.logging import _actus_log_record_factory


def _make_record() -> logging.LogRecord:
    """Build a record via the actus factory with realistic args."""
    return _actus_log_record_factory(
        "test_factory_defensive",
        logging.INFO,
        "/tmp/path.py",
        42,
        "msg",
        None,
        None,
    )


@dataclass
class _StubTraceContext:
    """Minimal duck-typed TraceContext for factory injection tests."""

    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    session_id: Optional[str] = None


class TestNoActiveContext:
    def test_returns_dashes_when_get_trace_context_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctx_module, "get_trace_context", lambda: None)

        record = _make_record()

        assert record.trace_id == "-"
        assert record.request_id == "-"
        assert record.session_id == "-"


class TestActiveContext:
    def test_populates_fields_from_full_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _StubTraceContext(
            trace_id="abc123",
            request_id="req-001",
            session_id="sess-xyz",
        )
        monkeypatch.setattr(ctx_module, "get_trace_context", lambda: ctx)

        record = _make_record()

        assert record.trace_id == "abc123"
        assert record.request_id == "req-001"
        assert record.session_id == "sess-xyz"

    def test_individual_none_fields_fall_back_to_dash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``None`` per-field still produces ``"-"`` (never the literal "None")."""
        ctx = _StubTraceContext(
            trace_id="alive",
            request_id=None,
            session_id=None,
        )
        monkeypatch.setattr(ctx_module, "get_trace_context", lambda: ctx)

        record = _make_record()

        assert record.trace_id == "alive"
        assert record.request_id == "-"
        assert record.session_id == "-"

    def test_context_missing_attribute_falls_back_to_dash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``getattr(ctx, field, None)`` short-circuits to ``"-"`` if missing."""

        class _BareContext:
            # No trace_id / request_id / session_id attrs at all.
            pass

        monkeypatch.setattr(
            ctx_module, "get_trace_context", lambda: _BareContext()
        )

        record = _make_record()

        assert record.trace_id == "-"
        assert record.request_id == "-"
        assert record.session_id == "-"


class TestExceptions:
    def test_runtime_error_in_get_trace_context_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode() -> None:
            raise RuntimeError("contextvar storage exploded")

        monkeypatch.setattr(ctx_module, "get_trace_context", _explode)

        record = _make_record()

        assert record.trace_id == "-"
        assert record.request_id == "-"
        assert record.session_id == "-"

    def test_attribute_error_in_get_trace_context_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _explode() -> None:
            raise AttributeError("trace_id deleted")

        monkeypatch.setattr(ctx_module, "get_trace_context", _explode)

        record = _make_record()

        assert record.trace_id == "-"
        assert record.request_id == "-"
        assert record.session_id == "-"

    def test_import_error_in_get_trace_context_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Boot-time circular import — local import inside factory raises ImportError."""

        def _explode() -> None:
            raise ImportError("circular import during boot")

        monkeypatch.setattr(ctx_module, "get_trace_context", _explode)

        record = _make_record()

        assert record.trace_id == "-"
        assert record.request_id == "-"
        assert record.session_id == "-"

    def test_factory_returns_valid_record_object_on_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Even on failure path, the result is a real ``LogRecord``."""

        def _explode() -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(ctx_module, "get_trace_context", _explode)

        record = _make_record()

        assert isinstance(record, logging.LogRecord)
        assert record.name == "test_factory_defensive"
        assert record.levelno == logging.INFO
        assert record.msg == "msg"


class TestBaseFactoryFailure:
    """Review-found P1: captured base factory raising must NOT escape.

    Prior implementation kept ``record = base(*args, **kwargs)`` outside
    the ``try/except`` block, so a captured upstream factory that
    raised would propagate the exception out of the actus factory and
    crash the very logging path the factory was meant to protect.
    """

    def test_orig_factory_raises_falls_back_to_stdlib(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the captured base factory raises, fall back to stdlib LogRecord."""

        def _bad_orig(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("captured base factory exploded")

        import app.infrastructure.logging.logging as logging_mod

        monkeypatch.setattr(
            logging_mod, "_ORIG_LOG_RECORD_FACTORY", _bad_orig
        )

        record = _make_record()

        assert isinstance(record, logging.LogRecord)
        assert record.name == "test_factory_defensive"
        # Trace-injection still falls through the outer try/except —
        # whether ``get_trace_context`` is reachable here is irrelevant;
        # the contract is "all three Q6 attrs exist".
        assert hasattr(record, "trace_id")
        assert hasattr(record, "request_id")
        assert hasattr(record, "session_id")


class TestReinstallNoRecursion:
    """Review-found P1: external wrapper + reinstall must NOT recurse.

    Scenario: setup_logging() installs our factory; an external lib
    then wraps it (``setLogRecordFactory(wrapper)`` where ``wrapper``
    internally calls our factory). setup_logging() is called again
    (lifespan reload). Without the ``_FACTORY_INSTALLED`` flag, the
    wrapper would be captured as the new ``_ORIG_LOG_RECORD_FACTORY``,
    and our factory calling ``_ORIG`` (= wrapper) calling our factory
    would recurse forever — ``RecursionError``.
    """

    def test_external_wrap_then_reinstall_no_recursion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.infrastructure.logging.logging as logging_mod

        # Reset module state so we can drive a clean install path.
        monkeypatch.setattr(logging_mod, "_FACTORY_INSTALLED", False)
        monkeypatch.setattr(logging_mod, "_ORIG_LOG_RECORD_FACTORY", None)
        saved_factory = logging.getLogRecordFactory()

        try:
            logging_mod._install_log_record_factory()
            installed = logging.getLogRecordFactory()
            assert installed is logging_mod._actus_log_record_factory

            # External lib wraps us.
            def _external_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
                rec = installed(*args, **kwargs)
                rec.external_marker = True
                return rec

            logging.setLogRecordFactory(_external_wrapper)

            # Re-install must NOT capture the wrapper and must not
            # recurse on the next factory invocation.
            logging_mod._install_log_record_factory()

            # Critical: actus factory is back in place; no recursion.
            assert (
                logging.getLogRecordFactory()
                is logging_mod._actus_log_record_factory
            )

            record = logging.getLogRecordFactory()(
                "test_reinstall",
                logging.INFO,
                "",
                0,
                "msg",
                None,
                None,
            )
            assert isinstance(record, logging.LogRecord)
            assert record.trace_id == "-"
        finally:
            logging.setLogRecordFactory(saved_factory)
