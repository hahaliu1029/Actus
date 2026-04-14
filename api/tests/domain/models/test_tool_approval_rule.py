from app.domain.models.tool_approval_rule import ToolApprovalRule

class TestToolApprovalRule:
    def test_create(self):
        rule = ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="pip install *")
        assert rule.id is not None
        assert rule.dir_pattern == ""

    def test_matches_command_no_dir(self):
        rule = ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="pip install *")
        assert rule.matches("pip install requests", None) is True
        assert rule.matches("rm -rf /", None) is False

    def test_matches_with_dir(self):
        rule = ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="rm -rf *", dir_pattern="/app/tmp/*")
        assert rule.matches("rm -rf build", "/app/tmp/workspace") is True
        assert rule.matches("rm -rf build", "/") is False

    def test_matches_empty_dir_pattern_ignores_dir(self):
        rule = ToolApprovalRule(user_id="u1", tool_name="file_write", rule="always_allow", command_pattern="/app/src/*")
        assert rule.matches("/app/src/main.py", None) is True
        assert rule.matches("/app/src/main.py", "/any/dir") is True
