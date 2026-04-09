import asyncio
import time
from unittest.mock import AsyncMock
from app.domain.services.confirmation_manager import ConfirmationManager, ConfirmationDetail


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConfirmationManager:
    def setup_method(self):
        self.redis = AsyncMock()
        self.mgr = ConfirmationManager(redis=self.redis, timeout_seconds=300)

    def test_store(self):
        detail = ConfirmationDetail(
            session_id="s1", tool_call_id="tc1", user_id="u1",
            tool_name="shell_execute", tool_args={"command": "ls"},
            risk_level="high", arg_digest="abc123",
            primary_arg="ls", dir_arg="/app", matched_patterns=[],
            deadline_ts=time.time() + 300,
        )
        _run(self.mgr.store(detail))
        self.redis.zadd.assert_called_once()
        self.redis.hset.assert_called_once()

    def test_read_returns_detail(self):
        self.redis.hgetall.return_value = {
            "session_id": "s1", "tool_call_id": "tc1", "user_id": "u1",
            "tool_name": "shell_execute", "tool_args_json": '{"command": "ls"}',
            "risk_level": "high", "arg_digest": "abc123",
            "primary_arg": "ls", "dir_arg": "/app",
            "matched_patterns_json": "[]", "deadline_ts": "1234567890",
            "status": "pending",
        }
        detail = _run(self.mgr.read("s1", "tc1"))
        assert detail is not None
        assert detail.tool_name == "shell_execute"
        assert detail.status == "pending"

    def test_read_missing(self):
        self.redis.hgetall.return_value = {}
        assert _run(self.mgr.read("s1", "tc1")) is None

    def test_mark_processing(self):
        _run(self.mgr.mark_processing("s1", "tc1"))
        self.redis.hset.assert_called_once()

    def test_cleanup(self):
        _run(self.mgr.cleanup("s1", "tc1"))
        self.redis.zrem.assert_called_once()
        self.redis.delete.assert_called_once()
