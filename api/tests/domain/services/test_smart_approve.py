import asyncio
from unittest.mock import AsyncMock, MagicMock
from app.domain.services.smart_approve import SmartApprove


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestSmartApprove:
    def setup_method(self):
        self.llm = AsyncMock()
        self.smart = SmartApprove(llm=self.llm)

    def test_approve(self):
        self.llm.ainvoke.return_value = MagicMock(content="APPROVE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "listing files")) == "approve"

    def test_deny(self):
        self.llm.ainvoke.return_value = MagicMock(content="DENY")
        assert _run(self.smart.evaluate("shell_execute", {"command": "rm -rf /"}, "high", ["recursive_delete"], "cleanup")) == "deny"

    def test_escalate(self):
        self.llm.ainvoke.return_value = MagicMock(content="ESCALATE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "pip install foo"}, "high", [], "deps")) == "escalate"

    def test_fallback_on_error(self):
        self.llm.ainvoke.side_effect = Exception("timeout")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "test")) == "escalate"

    def test_unexpected_response_escalates(self):
        self.llm.ainvoke.return_value = MagicMock(content="MAYBE")
        assert _run(self.smart.evaluate("shell_execute", {"command": "ls"}, "high", [], "test")) == "escalate"
