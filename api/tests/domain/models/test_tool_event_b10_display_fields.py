"""B10 §7.5 域事件面 — INV-B10-0: 内部流形与持久化形只多出 null 新 key."""
from __future__ import annotations

import json

from app.domain.models.event import ToolEvent, ToolEventStatus


def _evt() -> ToolEvent:
    return ToolEvent(
        tool_call_id="tc-b10-d",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"command": "ls"},
        status=ToolEventStatus.CALLING,
    )


def test_legacy_json_without_new_keys_still_validates():
    # 历史事件缺新字段 → Optional 默认反序列化 (spec §3.1 涟漪声明)
    legacy = {
        "type": "tool",
        "tool_call_id": "tc-legacy",
        "tool_name": "file",
        "function_name": "file_read",
        "function_args": {},
        "status": "calling",
    }
    evt = ToolEvent.model_validate(legacy)
    assert evt.read_only is None
    assert evt.destructive is None


def test_model_dump_json_internal_stream_shape():
    # agent_task_runner.py:901 内部流形 (event.model_dump_json())
    dumped = json.loads(_evt().model_dump_json())
    assert dumped["read_only"] is None
    assert dumped["destructive"] is None
    assert dumped["function_name"] == "shell_execute"


def test_model_dump_mode_json_persistence_shape():
    # db_session_repository.py:161 持久化形 (sessions.events JSONB)
    dumped = _evt().model_dump(mode="json")
    assert dumped["read_only"] is None
    assert dumped["destructive"] is None
    assert dumped["status"] == "calling"
