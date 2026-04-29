"""B5 PR-S1-3 acceptance: ReDoS guard for the v1 pattern set.

If any v1 pattern admits catastrophic backtracking (nested quantifiers,
overlapping alternations inside a repetition), pathological input
could push ``re.sub`` time from microseconds to seconds — and an
attacker who controls log content could DoS the logger thread.

This guard feeds 100KB of pathological-shape inputs through the
formatter and asserts each call returns within 50ms. The bound is
loose (the design doc target is 10ms) so the test stays robust on
shared CI runners; a regression introducing real backtracking would
push past 50ms by orders of magnitude.
"""
from __future__ import annotations

import logging
import time

import pytest

from app.infrastructure.logging.redaction import RedactingFormatter

# Loose CI-friendly threshold. Local dev runs typically format in
# under 5ms; a real ReDoS regression would blow well past 50ms.
_REDOS_BUDGET_SECONDS = 0.05


def _format(formatter: RedactingFormatter, message: str) -> float:
    record = logging.LogRecord(
        name="t",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=message,
        args=None,
        exc_info=None,
    )
    start = time.perf_counter()
    formatter.format(record)
    return time.perf_counter() - start


@pytest.fixture(scope="module")
def formatter() -> RedactingFormatter:
    return RedactingFormatter("%(message)s")


class TestRedosGuard:
    def test_long_run_of_a_chars(self, formatter):
        msg = "a" * 100_000
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"

    def test_long_run_with_authorization_prefix_no_value(self, formatter):
        msg = "Authorization: " + "a" * 100_000
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"

    def test_repeated_set_cookie_prefixes(self, formatter):
        msg = ("Set-Cookie: foo=bar; ") * 5_000
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"

    def test_long_pem_body_without_end_marker(self, formatter):
        msg = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + "x" * 100_000
            + "\n(no end marker)"
        )
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"

    def test_many_jwt_prefixes_no_dot(self, formatter):
        msg = ("eyJaaaaaaaaa") * 5_000
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"

    def test_bulk_real_workload_under_budget(self, formatter):
        chunks = [
            "request id 00000000-0000-4000-8000-000000000001 ",
            "Authorization: Bearer abc123def456ghi789jkl0 ",
            'json {"apiKey": "secret-payload-12345678"} ',
            "DSN postgresql://app:hunter2longerthan18@db:5432/main ",
        ]
        msg = "".join(chunks * 1000)
        elapsed = _format(formatter, msg)
        assert elapsed < _REDOS_BUDGET_SECONDS, f"elapsed {elapsed:.3f}s"
