"""B5 PR-S1-1 acceptance: ``attempt_ix`` distinguishes None / 0 / missing.

POST Q7 decision in the design doc. The contract treats:

- ``attempt_ix=None`` — explicit "no attempt context" (e.g., the
  initial LLM call before any Recovery retry has fired)
- ``attempt_ix=0`` — "first retry has occurred"
- ``attempt_ix`` absent — caller did not set the field at all (also
  treated as no-attempt-context, but distinct in the wire payload)

These three states are SEMANTICALLY DIFFERENT in downstream Recovery
analytics. Defaulting a missing key to 0 would silently merge "first
retry" and "no retry" into the same bucket and corrupt the funnel.
"""
from __future__ import annotations

from app.domain.external.observability import validate_attributes

# Spec example values that satisfy the v1 format constraints (trace_id =
# 32 hex, request_id / event_id = UUIDv4 with dashes). Required for the
# tightened validator (PR-S1-1 follow-up: NEVER null + format check).
_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
_EVENT_ID = "00000000-0000-4000-8000-000000000005"


def _required() -> dict[str, object]:
    return {
        "trace_id": _TRACE_ID,
        "request_id": _REQUEST_ID,
        "event_id": _EVENT_ID,
    }


class TestAttemptIxPreservation:
    def test_attempt_ix_zero_preserved(self):
        d = _required() | {"attempt_ix": 0}
        result = validate_attributes(d)
        assert result["attempt_ix"] == 0
        assert result["attempt_ix"] is not None

    def test_attempt_ix_none_preserved(self):
        d = _required() | {"attempt_ix": None}
        result = validate_attributes(d)
        assert "attempt_ix" in result
        assert result["attempt_ix"] is None

    def test_attempt_ix_missing_stays_missing(self):
        d = _required()
        result = validate_attributes(d)
        assert "attempt_ix" not in result

    def test_zero_and_none_are_distinct_in_output(self):
        zero_payload = validate_attributes(_required() | {"attempt_ix": 0})
        none_payload = validate_attributes(_required() | {"attempt_ix": None})
        assert zero_payload["attempt_ix"] != none_payload["attempt_ix"]
        assert zero_payload["attempt_ix"] == 0
        assert none_payload["attempt_ix"] is None

    def test_missing_and_none_are_distinguishable(self):
        missing_payload = validate_attributes(_required())
        none_payload = validate_attributes(_required() | {"attempt_ix": None})
        assert "attempt_ix" not in missing_payload
        assert "attempt_ix" in none_payload
        assert none_payload["attempt_ix"] is None
