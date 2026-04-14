import asyncio
from unittest.mock import AsyncMock
from app.domain.services.approval_cache import ApprovalCache
from app.domain.models.tool_approval_rule import ToolApprovalRule


def _run(coro):
    """Run an async coroutine synchronously — no pytest-asyncio needed."""
    return asyncio.run(coro)


class TestApprovalCache:
    def setup_method(self):
        self.redis = AsyncMock()
        self.rule_repo = AsyncMock()
        self.cache = ApprovalCache(redis=self.redis, rule_repo=self.rule_repo)

    def test_always_allow_match(self):
        self.rule_repo.find_by_user_and_tool.return_value = [
            ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="pip install *")
        ]
        self.redis.get.return_value = None
        assert _run(self.cache.check("u1", "s1", "shell_execute", "abc", "pip install requests", None)) == "allow"

    def test_always_deny_match(self):
        self.rule_repo.find_by_user_and_tool.return_value = [
            ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_deny", command_pattern="rm -rf *")
        ]
        self.redis.get.return_value = None
        assert _run(self.cache.check("u1", "s1", "shell_execute", "abc", "rm -rf /", None)) == "deny"

    def test_deny_takes_precedence(self):
        self.rule_repo.find_by_user_and_tool.return_value = [
            ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="*"),
            ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_deny", command_pattern="rm *"),
        ]
        self.redis.get.return_value = None
        assert _run(self.cache.check("u1", "s1", "shell_execute", "abc", "rm -rf /", None)) == "deny"

    def test_no_match(self):
        self.rule_repo.find_by_user_and_tool.return_value = [
            ToolApprovalRule(user_id="u1", tool_name="shell_execute", rule="always_allow", command_pattern="pip *")
        ]
        self.redis.get.return_value = None
        assert _run(self.cache.check("u1", "s1", "shell_execute", "abc", "ls", None)) == "no_match"

    def test_session_hit(self):
        self.rule_repo.find_by_user_and_tool.return_value = []
        self.redis.get.return_value = "1"
        assert _run(self.cache.check("u1", "s1", "shell_execute", "abc", "ls", None)) == "allow"

    def test_write_session(self):
        _run(self.cache.write_session("s1", "shell_execute", "abc123"))
        self.redis.set.assert_called_once()
