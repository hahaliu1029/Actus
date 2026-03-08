from __future__ import annotations

from datetime import datetime

from app.domain.models.skill_creation_state import SkillCreationState


def test_skill_creation_state_round_trip() -> None:
    state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
        last_tool_name="brainstorm_skill",
        last_tool_call_id="call_123",
        saved_tool_result_json='{"success": true}',
        blueprint={"skill_name": "meeting-audio-analyzer"},
        blueprint_json='{"skill_name":"meeting-audio-analyzer"}',
        requested_at=datetime(2026, 3, 8, 12, 0, 0),
    )

    payload = state.model_dump(mode="json")
    restored = SkillCreationState.model_validate(payload)

    assert restored.pending_action == "generate"
    assert restored.last_tool_name == "brainstorm_skill"
    assert restored.blueprint is not None
    assert restored.blueprint["skill_name"] == "meeting-audio-analyzer"

