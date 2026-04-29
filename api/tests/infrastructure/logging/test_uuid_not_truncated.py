"""B5 PR-S1-3 acceptance: UUIDv4 exception in token truncation.

Per the design doc canonical attribute contract, ``trace_id`` /
``session_id`` / ``event_id`` are non-secret correlation IDs that flow
through logs verbatim. If the truncation rule applied to UUIDs, every
log line carrying a trace_id would show ``cb1234[REDACTED]4736`` and
trace correlation would break.

The fix lives inside ``_truncate``: a value that fullmatches the
canonical UUIDv4 pattern is returned unchanged. This test class locks
that exception both at the helper level and through the public
``RedactingFormatter`` surface.
"""
from __future__ import annotations

import logging
import uuid

from app.infrastructure.logging.redaction import (
    RedactingFormatter,
    _truncate,
)


def _format(message: str) -> str:
    formatter = RedactingFormatter("%(message)s")
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )
    return formatter.format(record)


class TestTruncateHelperUuidException:
    def test_uuid_v4_passes_through(self):
        canonical = "00000000-0000-4000-8000-000000000001"
        assert _truncate(canonical) == canonical

    def test_random_uuid_passes_through(self):
        for _ in range(5):
            u = str(uuid.uuid4())
            assert _truncate(u) == u

    def test_long_non_uuid_string_truncated(self):
        s = "abcdef1234567890abcdXX"
        result = _truncate(s)
        assert result == "abcdef[REDACTED]cdXX"

    def test_uuid_v3_not_exempted(self):
        # UUIDv3 (version=3) is not the canonical attribute format;
        # treat as a regular long-token secret.
        v3 = "00000000-0000-3000-8000-000000000001"
        result = _truncate(v3)
        assert result != v3
        assert "[REDACTED]" in result

    def test_uuid_with_uppercase_not_exempted(self):
        upper = "AABBCCDD-AABB-4CCC-8DDD-EEFF00112233"
        result = _truncate(upper)
        # canonical regex is lowercase-only; uppercase falls through
        assert "[REDACTED]" in result


class TestFormatterDoesNotEatUuidsInTraceContext:
    def test_uuid_in_authorization_value_kept(self):
        # Pattern #15 captures the whole line value, but the truncation
        # rule's UUID exception keeps it readable when the captured
        # slice itself fullmatches UUIDv4.
        rendered = _format(
            "Authorization: 00000000-0000-4000-8000-000000000001"
        )
        assert "00000000-0000-4000-8000-000000000001" in rendered

    def test_uuid_in_logger_extra_field_passes_through(self):
        # Simulate the canonical_attributes-shaped log line PR-S1-4
        # produces: correlation IDs flow through redaction untouched.
        trace_id = "cb1234567890abcdef1234567890abcd"  # 32 hex Sprint-1 form
        session_id = "00000000-0000-4000-8000-000000000003"
        event_id = "00000000-0000-4000-8000-000000000005"
        msg = f"trace_id={trace_id} session_id={session_id} event_id={event_id}"
        rendered = _format(msg)
        assert session_id in rendered
        assert event_id in rendered
        # The 32-hex trace_id is not a UUIDv4 (no dashes / version
        # nibble) and no v1 secret pattern matches it either, so it
        # also flows through.
        assert trace_id in rendered
