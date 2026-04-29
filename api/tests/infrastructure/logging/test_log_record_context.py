"""B5 PR-S1-4 acceptance: end-to-end LogRecord context injection.

Where ``test_log_record_factory_defensive.py`` proves the factory
itself is bullet-proof in isolation, this suite walks the full path:

    logger.info(...)
        → Logger._log builds LogRecord via the factory
        → propagates up to root handlers
        → ``_CaptureHandler.emit`` records the LogRecord

so every step that production code traverses (factory invocation,
propagation through ``_RedactingPropagateOnlyLogger``, root-handler
emit) gets exercised. The assertions confirm:

- Outside any request scope, ``trace_id`` / ``request_id`` /
  ``session_id`` default to the literal ``"-"``.
- With ``set_trace_context``, all three fields reflect the bound
  ``TraceContext``.
- With ``bind_session_context``, ``session_id`` reflects the binding
  and ``trace_id`` / ``request_id`` are auto-generated (non-default).
- Formatters that read ``%(trace_id)s`` render the bound value
  verbatim — no ``"None"`` literal, no ``AttributeError``.
"""
from __future__ import annotations

import logging

import pytest

from app.domain.external.observability import TraceContext
from app.infrastructure.observability.context import (
    bind_session_context,
    reset_trace_context,
    set_trace_context,
)


_TRACE_ID = "0123456789abcdef0123456789abcdef"
_REQUEST_ID = "12345678-1234-4abc-8def-012345678901"
_EVENT_ID = "abcdef01-2345-4678-89ab-cdef01234567"


class _CaptureHandler(logging.Handler):
    """Append every emitted ``LogRecord`` to ``self.records`` for inspection."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _attach_capture(root: logging.Logger) -> _CaptureHandler:
    capture = _CaptureHandler()
    root.addHandler(capture)
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)
    return capture


def test_outside_context_all_fields_default_to_dash(
    isolated_root_logger: logging.Logger,
) -> None:
    capture = _attach_capture(isolated_root_logger)

    logging.getLogger("test_logrecord_outside").info("baseline")

    assert capture.records, "logger.info did not propagate to root capture"
    rec = capture.records[-1]
    assert rec.trace_id == "-"
    assert rec.request_id == "-"
    assert rec.session_id == "-"


def test_with_set_trace_context_all_fields_populated(
    isolated_root_logger: logging.Logger,
) -> None:
    capture = _attach_capture(isolated_root_logger)

    ctx = TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
        session_id="sess-explicit-001",
    )
    token = set_trace_context(ctx)
    try:
        logging.getLogger("test_logrecord_with_ctx").info("with ctx")
    finally:
        reset_trace_context(token)

    rec = capture.records[-1]
    assert rec.trace_id == _TRACE_ID
    assert rec.request_id == _REQUEST_ID
    assert rec.session_id == "sess-explicit-001"


def test_after_reset_back_to_dash(
    isolated_root_logger: logging.Logger,
) -> None:
    """Cleanup verification — once the token resets, defaults return."""
    capture = _attach_capture(isolated_root_logger)

    ctx = TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
    )
    token = set_trace_context(ctx)
    logging.getLogger("test_logrecord_during").info("during")
    reset_trace_context(token)
    logging.getLogger("test_logrecord_after").info("after")

    assert capture.records[-2].trace_id == _TRACE_ID
    assert capture.records[-1].trace_id == "-"
    assert capture.records[-1].request_id == "-"
    assert capture.records[-1].session_id == "-"


@pytest.mark.anyio
async def test_inside_bind_session_context_session_id_populated(
    isolated_root_logger: logging.Logger,
) -> None:
    capture = _attach_capture(isolated_root_logger)

    async with bind_session_context("sess-bind-007"):
        logging.getLogger("test_logrecord_session_bind").info("inside")

    rec = capture.records[-1]
    assert rec.session_id == "sess-bind-007"
    # Outside-request fresh-context path auto-generates trace/request,
    # so they are NOT the default placeholder.
    assert rec.trace_id != "-"
    assert rec.request_id != "-"
    # 32-hex trace_id format from ``uuid4().hex``.
    assert len(rec.trace_id) == 32


def test_format_renders_trace_id_via_percent_token(
    isolated_root_logger: logging.Logger,
) -> None:
    """``%(trace_id)s`` formatter token must render the populated value.

    Pins the integration between the factory's attr names and the
    canonical formatter shape used in ``_install_redacting_formatter``
    (which in PR-S1-4 still uses the legacy format string but sets the
    foundation for ``%(trace_id)s`` once Sprint 2 wires it in).
    """
    capture = _attach_capture(isolated_root_logger)
    formatter = logging.Formatter("%(trace_id)s|%(request_id)s|%(session_id)s")
    capture.setFormatter(formatter)

    ctx = TraceContext(
        trace_id=_TRACE_ID,
        request_id=_REQUEST_ID,
        event_id=_EVENT_ID,
        session_id="fmt-sess",
    )
    token = set_trace_context(ctx)
    try:
        logging.getLogger("test_logrecord_format").info("payload")
    finally:
        reset_trace_context(token)

    rendered = capture.format(capture.records[-1])
    assert rendered == f"{_TRACE_ID}|{_REQUEST_ID}|fmt-sess"
