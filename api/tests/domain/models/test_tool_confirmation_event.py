from app.domain.models.event import ToolConfirmationEvent


class TestToolConfirmationEvent:
    def test_create(self):
        evt = ToolConfirmationEvent(
            tool_call_id="tc-1", tool_name="shell_execute",
            tool_args={"command": "rm -rf /"}, risk_level="high",
            risk_reason="recursive delete", matched_patterns=["recursive_delete"],
            timeout_seconds=300,
        )
        assert evt.type == "tool_confirmation"
        assert evt.id is not None

    def test_default_approval_options(self):
        evt = ToolConfirmationEvent(
            tool_call_id="tc-1", tool_name="shell_execute",
            tool_args={}, risk_level="high", risk_reason="test",
            matched_patterns=[], timeout_seconds=300,
        )
        assert evt.approval_options == ["once", "session", "always", "deny"]
