"""R4 regression: `_is_unknown_tool_event` must detect unknown-tool events
produced by `_translate_outcome` under the R4 artifact shape.

Pre-R4, react_graph emitted `ToolEvent.function_result.data["code"] == "UNKNOWN_TOOL"`.
Post-R4, react_graph at `react_graph.py:1232` synthesizes
`AllowError(reason=DecisionReason(type="exception", code="unknown_tool", ...))`
and `_translate_outcome` persists that payload on `event.artifact` (dict) while
`event.function_result` only carries {success, message}. The pre-R4 legacy
dict check at `agent_task_runner._is_unknown_tool_event` therefore misses real
R4 unknown-tool events silently, breaking step-skill reselect.

This regression test drives through the real `_translate_outcome` producer and
asserts `_is_unknown_tool_event` returns True on its output — locking the
invariant that unknown-tool detection survives the R4 artifact shape.
"""
from __future__ import annotations

import pytest

from app.domain.models.tool_result import AllowError, DecisionReason
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.graphs.react_graph import _translate_outcome
from app.domain.services.tools.tool_source_resolver import ToolSource

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


async def test_is_unknown_tool_event_detects_r4_artifact_shape() -> None:
    """React_graph 合成 AllowError(reason.code="unknown_tool") 的事件, 走完
    _translate_outcome 后应被 _is_unknown_tool_event 识别."""
    tool_source = ToolSource(
        source="native", category="unknown", canonical_name="missing_tool",
    )
    unknown_outcome = AllowError(
        content="Error: Unknown tool 'missing_tool'",
        reason=DecisionReason(
            type="exception",
            code="unknown_tool",
            message="Tool 'missing_tool' not in this graph's tool_map",
        ),
    )
    tool_call = {"id": "c1", "name": "missing_tool", "args": {}}

    _, _, events = await _translate_outcome(
        outcome=unknown_outcome,
        tool_call=tool_call,
        tool_source=tool_source,
        session_ctx=None,
        tool_result_max_chars=8000,
        guide_injector=None,
    )

    assert len(events) == 1
    tool_event = events[0]
    # This assertion is what codex Round 5 P2 flagged as currently broken.
    assert AgentTaskRunner._is_unknown_tool_event(tool_event) is True


async def test_is_unknown_tool_event_preserves_legacy_detection() -> None:
    """Pre-R4 事件日志回放: function_result.data["code"] == "UNKNOWN_TOOL" 仍需识别."""
    from app.domain.models.event import ToolEvent, ToolEventStatus
    from app.domain.models.tool_result import ToolResult

    legacy_event = ToolEvent(
        tool_call_id="c1",
        tool_name="unknown",
        function_name="missing_tool",
        function_args={},
        function_result=ToolResult(
            success=False,
            message="UNKNOWN_TOOL: missing_tool",
            data={"code": "UNKNOWN_TOOL", "tool_name": "missing_tool"},
        ),
        status=ToolEventStatus.CALLED,
    )
    assert AgentTaskRunner._is_unknown_tool_event(legacy_event) is True


async def test_is_unknown_tool_event_false_for_successful_call() -> None:
    """Sanity: 成功的工具调用不应被识别为 unknown."""
    from app.domain.models.tool_result import AllowSuccess

    tool_source = ToolSource(
        source="native", category="shell", canonical_name="shell_execute",
    )
    success_outcome = AllowSuccess(content="ok")
    tool_call = {"id": "c1", "name": "shell_execute", "args": {}}

    _, _, events = await _translate_outcome(
        outcome=success_outcome,
        tool_call=tool_call,
        tool_source=tool_source,
        session_ctx=None,
        tool_result_max_chars=8000,
        guide_injector=None,
    )

    assert len(events) == 1
    assert AgentTaskRunner._is_unknown_tool_event(events[0]) is False


async def test_is_unknown_tool_event_false_for_other_exception_codes() -> None:
    """R4: 其他 AllowError(reason.code != "unknown_tool") 不应被识别为 unknown."""
    tool_source = ToolSource(
        source="native", category="shell", canonical_name="shell_execute",
    )
    other_error = AllowError(
        content="Connection refused",
        reason=DecisionReason(
            type="exception",
            code="ECONNREFUSED",
            message="Failed to connect",
        ),
    )
    tool_call = {"id": "c1", "name": "shell_execute", "args": {}}

    _, _, events = await _translate_outcome(
        outcome=other_error,
        tool_call=tool_call,
        tool_source=tool_source,
        session_ctx=None,
        tool_result_max_chars=8000,
        guide_injector=None,
    )

    assert len(events) == 1
    assert AgentTaskRunner._is_unknown_tool_event(events[0]) is False
