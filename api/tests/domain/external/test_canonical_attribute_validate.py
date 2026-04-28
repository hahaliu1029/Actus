"""B5 PR-S1-1 acceptance: ``validate_attributes`` contract behavior.

Locks the v1 join-key contract:

- Unknown keys are silently dropped (caller cannot leak ad-hoc fields
  into downstream JSONL or OTel span attributes).
- Missing keys from ``REQUIRED_ATTRIBUTES`` raise ``ValueError``.
- Required values must be non-empty ``str`` matching the v1 format
  pattern (trace_id = 32 hex, request_id / event_id = UUIDv4).
- ``None`` / non-``str`` / empty string for a required attr fails fast.
"""
from __future__ import annotations

import uuid

import pytest

from app.domain.external.observability import (
    CANONICAL_ATTRIBUTES,
    REQUIRED_ATTRIBUTES,
    validate_attributes,
)

# Spec examples (B5 design doc §"Canonical Attribute Contract v1"). Tests
# build their dicts from these so a contract-correct payload always
# survives the validator and a contract-violating payload always fails.
VALID_TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
VALID_REQUEST_ID = "00000000-0000-4000-8000-000000000001"
VALID_EVENT_ID = "00000000-0000-4000-8000-000000000005"
VALID_SESSION_ID = "00000000-0000-4000-8000-000000000003"


def _required_only() -> dict[str, object]:
    return {
        "trace_id": VALID_TRACE_ID,
        "request_id": VALID_REQUEST_ID,
        "event_id": VALID_EVENT_ID,
    }


class TestValidateAttributesDropsUnknownKeys:
    def test_unknown_key_dropped(self):
        d = {
            **_required_only(),
            "rogue_field": "should-not-survive",
        }
        result = validate_attributes(d)
        assert "rogue_field" not in result
        assert result["trace_id"] == VALID_TRACE_ID

    def test_only_canonical_keys_survive(self):
        d = {
            **_required_only(),
            "session_id": VALID_SESSION_ID,
            "extra_a": 1,
            "extra_b": [1, 2, 3],
            "extra_c": {"nested": True},
        }
        result = validate_attributes(d)
        assert set(result.keys()).issubset(set(CANONICAL_ATTRIBUTES))


class TestValidateAttributesRequiredKeys:
    def test_missing_trace_id_raises(self):
        with pytest.raises(ValueError, match="missing required canonical attrs"):
            validate_attributes(
                {"request_id": VALID_REQUEST_ID, "event_id": VALID_EVENT_ID}
            )

    def test_missing_request_id_raises(self):
        with pytest.raises(ValueError, match="missing required canonical attrs"):
            validate_attributes(
                {"trace_id": VALID_TRACE_ID, "event_id": VALID_EVENT_ID}
            )

    def test_missing_event_id_raises(self):
        with pytest.raises(ValueError, match="missing required canonical attrs"):
            validate_attributes(
                {"trace_id": VALID_TRACE_ID, "request_id": VALID_REQUEST_ID}
            )

    def test_all_required_present_passes(self):
        result = validate_attributes(_required_only())
        assert result == _required_only()


class TestRequiredAttrNoneRejected:
    """Spec ``trace_id`` / ``request_id`` / ``event_id`` are NEVER null.

    The previous implementation only checked key presence and would
    silently pass ``None`` through to downstream JSONL / OTel emit
    sites, breaking trace correlation.
    """

    def test_trace_id_none_raises(self):
        d = _required_only() | {"trace_id": None}
        with pytest.raises(ValueError, match="must not be None"):
            validate_attributes(d)

    def test_request_id_none_raises(self):
        d = _required_only() | {"request_id": None}
        with pytest.raises(ValueError, match="must not be None"):
            validate_attributes(d)

    def test_event_id_none_raises(self):
        d = _required_only() | {"event_id": None}
        with pytest.raises(ValueError, match="must not be None"):
            validate_attributes(d)


class TestRequiredAttrEmptyRejected:
    def test_trace_id_empty_raises(self):
        d = _required_only() | {"trace_id": ""}
        with pytest.raises(ValueError, match="must not be empty"):
            validate_attributes(d)

    def test_request_id_empty_raises(self):
        d = _required_only() | {"request_id": ""}
        with pytest.raises(ValueError, match="must not be empty"):
            validate_attributes(d)

    def test_event_id_empty_raises(self):
        d = _required_only() | {"event_id": ""}
        with pytest.raises(ValueError, match="must not be empty"):
            validate_attributes(d)


class TestRequiredAttrTypeRejected:
    def test_trace_id_int_raises(self):
        d = _required_only() | {"trace_id": 123}
        with pytest.raises(TypeError, match="must be str"):
            validate_attributes(d)

    def test_request_id_uuid_object_raises(self):
        d = _required_only() | {"request_id": uuid.UUID(VALID_REQUEST_ID)}
        with pytest.raises(TypeError, match="must be str"):
            validate_attributes(d)

    def test_event_id_bytes_raises(self):
        d = _required_only() | {
            "event_id": b"00000000-0000-4000-8000-000000000005"
        }
        with pytest.raises(TypeError, match="must be str"):
            validate_attributes(d)


class TestRequiredAttrFormatRejected:
    """V1 format constraints:

    - ``trace_id``: 32 lowercase hex chars (matches both Sprint-1
      ``uuid4().hex`` fallback and Sprint-2 OTel-native trace IDs).
    - ``request_id`` / ``event_id``: UUIDv4 with dashes (8-4-4-4-12,
      version=4, variant in 8|9|a|b).
    """

    @pytest.mark.parametrize(
        "bad_value",
        [
            "short",
            "x" * 32,
            "ABCDEF" + "0" * 26,
            "4bf92f3577b34da6a3ce929d0e0e4736-extra",
            VALID_REQUEST_ID,
        ],
    )
    def test_trace_id_bad_format_raises(self, bad_value):
        d = _required_only() | {"trace_id": bad_value}
        with pytest.raises(ValueError, match="fails v1 format check"):
            validate_attributes(d)

    @pytest.mark.parametrize(
        "bad_value",
        [
            "not-a-uuid",
            "00000000000040008000000000000001",
            VALID_TRACE_ID,
            "00000000-0000-3000-8000-000000000001",
            "00000000-0000-4000-1000-000000000001",
            "00000000-0000-4000-8000-00000000000",
        ],
    )
    def test_request_id_bad_format_raises(self, bad_value):
        d = _required_only() | {"request_id": bad_value}
        with pytest.raises(ValueError, match="fails v1 format check"):
            validate_attributes(d)

    @pytest.mark.parametrize(
        "bad_value",
        [
            "not-a-uuid",
            VALID_TRACE_ID,
            "00000000-0000-5000-8000-000000000001",
            "00000000-0000-4000-c000-000000000001",
        ],
    )
    def test_event_id_bad_format_raises(self, bad_value):
        d = _required_only() | {"event_id": bad_value}
        with pytest.raises(ValueError, match="fails v1 format check"):
            validate_attributes(d)


class TestRequiredAttrValidFormatAccepted:
    def test_spec_examples_pass(self):
        result = validate_attributes(_required_only())
        assert result["trace_id"] == VALID_TRACE_ID
        assert result["request_id"] == VALID_REQUEST_ID
        assert result["event_id"] == VALID_EVENT_ID

    def test_uuid4_hex_pass_for_trace_id(self):
        d = _required_only() | {"trace_id": uuid.uuid4().hex}
        result = validate_attributes(d)
        assert result["trace_id"]

    def test_uuid4_str_pass_for_request_id_and_event_id(self):
        d = _required_only() | {
            "request_id": str(uuid.uuid4()),
            "event_id": str(uuid.uuid4()),
        }
        result = validate_attributes(d)
        assert result["request_id"]
        assert result["event_id"]


class TestContractTuplesFrozen:
    """Lock the contract size + identity at v1.

    Any add/remove to the tuples must update this test. The assertion
    here is intentionally noisy so a drive-by edit to
    ``CANONICAL_ATTRIBUTES`` triggers a CI failure that prompts the
    spec-review process described in the design doc.
    """

    def test_canonical_v1_size(self):
        assert len(CANONICAL_ATTRIBUTES) == 15

    def test_required_v1_size(self):
        assert len(REQUIRED_ATTRIBUTES) == 3

    def test_required_subset_of_canonical(self):
        assert set(REQUIRED_ATTRIBUTES).issubset(set(CANONICAL_ATTRIBUTES))

    def test_v1_field_names(self):
        assert set(CANONICAL_ATTRIBUTES) == {
            "trace_id",
            "request_id",
            "session_id",
            "user_id_hash",
            "graph_node",
            "step_id",
            "tool_name",
            "tool_call_id",
            "tool_args_hash",
            "tool_args_size",
            "llm_provider",
            "model",
            "attempt_ix",
            "event_id",
            "decision_reason",
        }
