"""B5 PR-S1-3 acceptance: Q5 closure-capture defense.

Locks the invariant that monkey-patching the module-global
``_REDACT_ENABLED`` AFTER a ``RedactingFormatter`` instance exists does
NOT disable redaction for that instance. The instance read the lock at
``__init__`` time and stashed it on ``self._enabled``; the global is no
longer consulted.

Why this matters: a misbehaving plugin / late import / test fixture
might flip the flag mid-run. Without closure capture, every formatter
across the process would silently start emitting raw secrets. With
closure capture, only formatters created AFTER the flip can be
disabled — and creating a new RedactingFormatter requires touching
``setup_logging`` (which is itself audited).
"""
from __future__ import annotations

import logging

import app.infrastructure.logging.redaction as redaction_module
from app.infrastructure.logging.redaction import RedactingFormatter


def _format_with(formatter: RedactingFormatter, msg: str) -> str:
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )
    return formatter.format(record)


class TestClosureCaptureDefense:
    def test_existing_formatter_still_redacts_after_global_flip(self):
        assert redaction_module._REDACT_ENABLED is True
        formatter = RedactingFormatter("%(message)s")

        original = redaction_module._REDACT_ENABLED
        redaction_module._REDACT_ENABLED = False
        try:
            rendered = _format_with(
                formatter,
                "leak attempt: sk-aaaaaaaaaaaaaaaaaaaa1234",
            )
            assert "[REDACTED]" in rendered
            assert "aaaaaaaaaa" not in rendered
        finally:
            redaction_module._REDACT_ENABLED = original

    def test_new_formatter_after_flip_does_not_redact(self):
        original = redaction_module._REDACT_ENABLED
        redaction_module._REDACT_ENABLED = False
        try:
            disabled = RedactingFormatter("%(message)s")
            rendered = _format_with(
                disabled,
                "explicit-opt-out: sk-aaaaaaaaaaaaaaaaaaaa1234",
            )
            assert "sk-aaaaaaaaaaaaaaaaaaaa1234" in rendered
            assert "[REDACTED]" not in rendered
        finally:
            redaction_module._REDACT_ENABLED = original

    def test_two_formatters_independent_locks(self):
        before = RedactingFormatter("%(message)s")
        original = redaction_module._REDACT_ENABLED
        redaction_module._REDACT_ENABLED = False
        try:
            after = RedactingFormatter("%(message)s")

            msg = "creds: sk-aaaaaaaaaaaaaaaaaaaa1234"
            assert "[REDACTED]" in _format_with(before, msg)
            assert "[REDACTED]" not in _format_with(after, msg)
        finally:
            redaction_module._REDACT_ENABLED = original
