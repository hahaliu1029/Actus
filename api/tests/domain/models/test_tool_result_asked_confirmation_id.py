"""PE-0: Asked.confirmation_id additive field. Must not break R2 CS2
golden JSON matrix decoding (extra='forbid')."""

import json

from app.domain.models.tool_result import (
    Asked,
    DecisionReason,
    TOOL_OUTCOME_ADAPTER,
)


def test_asked_with_confirmation_id_roundtrip():
    asked = Asked(
        content="Confirm file_write",
        reason=DecisionReason(
            type="approval_policy",
            code="ask:once",
            message="user confirmation required",
        ),
        confirmation_id="s1:tc1",
    )
    dumped = asked.model_dump()
    assert dumped["confirmation_id"] == "s1:tc1"
    decoded = TOOL_OUTCOME_ADAPTER.validate_python(dumped)
    assert isinstance(decoded, Asked)
    assert decoded.confirmation_id == "s1:tc1"


def test_asked_without_confirmation_id_defaults_to_none_back_compat():
    """Legacy callers building Asked without confirmation_id MUST still work."""
    asked = Asked(
        content="legacy ask",
        reason=DecisionReason(type="approval_policy", code="ask:session", message="legacy"),
    )
    assert asked.confirmation_id is None


def test_old_wire_payload_decodes_without_confirmation_id():
    """R4 CS3 envelope round-trip: old payloads (no confirmation_id) must
    still decode under extra='forbid'."""
    payload = {
        "variant": "asked",
        "content": "old client",
        "reason": {"type": "approval_policy", "code": "ask:once", "message": "old"},
    }
    decoded = TOOL_OUTCOME_ADAPTER.validate_python(payload)
    assert isinstance(decoded, Asked)
    assert decoded.confirmation_id is None
