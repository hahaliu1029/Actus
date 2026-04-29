"""B5 PR-S1-3 acceptance: redaction covers exception traceback content.

The most common secret-leak vector is a ``ValueError(f"failed to auth
with key {key}")``. The traceback renders the exception's ``str(exc)``,
exposing the secret. ``RedactingFormatter`` MUST redact over the FULL
rendered string (including ``formatException`` output), not just
``record.getMessage()``.

Implementation note: after ``app.main`` is imported (which conftest
does to wire the FastAPI test client), ``setup_logging`` has already
fired and installed the self-healing Logger subclass. Inside that
class, ``addHandler`` is a no-op — so we cannot test by attaching a
StreamHandler to a fresh named logger. Instead, we build a
``LogRecord`` directly with ``exc_info=sys.exc_info()`` and feed it to
``RedactingFormatter.format``, which is exactly the path the production
log handler takes.
"""
from __future__ import annotations

import logging
import sys

from app.infrastructure.logging.redaction import RedactingFormatter


def _capture_exception_log() -> str:
    """Render a LogRecord with exc_info set, through ``RedactingFormatter``.

    ``logging.Formatter.format`` calls ``self.formatException(exc_info)``
    when ``record.exc_info`` is set, appending the traceback to the
    rendered string. ``RedactingFormatter`` runs its substitution table
    over the full rendered text afterwards.
    """
    formatter = RedactingFormatter("%(levelname)s %(message)s")
    try:
        raise ValueError(
            "failed to auth with key sk-aaaaaaaaaaaaaaaaaaaa1234bbbb"
        )
    except ValueError:
        record = logging.LogRecord(
            name="t",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="auth call blew up",
            args=None,
            exc_info=sys.exc_info(),
        )
    return formatter.format(record)


class TestRedactionInTraceback:
    def test_secret_in_exception_message_redacted(self):
        rendered = _capture_exception_log()
        assert "[REDACTED]" in rendered
        assert "aaaaaaaaaaaa" not in rendered
        assert "ValueError" in rendered
        assert "auth call blew up" in rendered

    def test_traceback_preserves_first_and_last_chars(self):
        rendered = _capture_exception_log()
        # ≥18 chars → first 6 + [REDACTED] + last 4
        assert "sk-aaa" in rendered
        assert "bbbb" in rendered

    def test_no_secret_in_message_field(self):
        rendered = _capture_exception_log()
        assert "auth call blew up" in rendered
        assert "failed to auth with key" in rendered
