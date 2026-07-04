"""Task 20 — react_graph 扩展统计埋点（B9 PR-3）。

`_record_finalize`（PE 路径）与 `_legacy_record_finalize`（legacy 路径）在 tracker/
metrics 记录之后，additive 地调用 `configurable["extension_stats_recorder"].record(
tool_name_raw, is_success, latency_ms)`——fire-and-forget、fail-open。recorder 为 None
（flag off = lifespan 未注入）时零调用、零行为。

这些测试驱动 **真实** `build_react_graph` 内层图，触发真实闭包，不 mock 埋点点本身。
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool

from app.domain.services.graphs.react_graph import build_react_graph


class _FakeRecorder:
    """结构化满足 ExtensionStatsRecorder Protocol；捕获 record 调用。"""

    def __init__(self, *, raises: bool = False) -> None:
        self.calls: list[tuple[str, bool, float]] = []
        self._raises = raises

    def record(self, tool_name: str, success: bool, latency_ms: float) -> None:
        self.calls.append((tool_name, success, latency_ms))
        if self._raises:
            raise RuntimeError("boom — 埋点内部故障，必须被 fail-open 吞掉")


def _build_fake_llm(tool_name: str, tool_args: dict) -> MagicMock:
    """第一轮返回单工具调用，第二轮返回终态 JSON。"""
    call_count = 0

    async def fake_ainvoke(messages, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": tool_name, "args": tool_args}],
            )
        return AIMessage(content='{"success": true, "result": "done", "attachments": []}')

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.ainvoke = fake_ainvoke
    return llm


def _initial_state() -> dict:
    return {
        "messages": [HumanMessage(content="do it")],
        "events": [],
        "attempt_count": 0,
        "failure_count": 0,
        "soft_hint_sent": False,
        "should_interrupt": False,
        "llm_input_messages": [],
    }


def _run(graph, configurable: dict | None):
    config: dict = {"recursion_limit": 10}
    if configurable is not None:
        config["configurable"] = configurable
    return asyncio.run(graph.ainvoke(_initial_state(), config=config))


def test_record_finalize_calls_stats_recorder():
    """recorder 非 None → 工具执行后 record(tool_name_raw, is_success, latency_ms)。"""

    @lc_tool
    async def mcp_srv_tool(x: str) -> str:
        """A fake mcp tool."""
        return "ok"

    llm = _build_fake_llm("mcp_srv_tool", {"x": "1"})
    graph = build_react_graph(llm=llm, tools=[mcp_srv_tool])
    recorder = _FakeRecorder()

    _run(graph, {"extension_stats_recorder": recorder})

    assert len(recorder.calls) == 1, f"expected exactly 1 record call, got {recorder.calls}"
    name, success, latency = recorder.calls[0]
    assert name == "mcp_srv_tool"
    assert success is True
    assert isinstance(latency, float) and latency >= 0.0


def test_stats_recorder_exception_does_not_break_tool_flow():
    """record() 抛异常 → fail-open：图正常跑完、工具结果不受影响。"""

    @lc_tool
    async def mcp_srv_tool(x: str) -> str:
        """A fake mcp tool."""
        return "ok"

    llm = _build_fake_llm("mcp_srv_tool", {"x": "1"})
    graph = build_react_graph(llm=llm, tools=[mcp_srv_tool])
    recorder = _FakeRecorder(raises=True)

    final_state = _run(graph, {"extension_stats_recorder": recorder})

    # record 被调用（并抛），但异常被吞，图仍产出 ToolMessage。
    assert len(recorder.calls) == 1
    from langchain_core.messages import ToolMessage

    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1, "工具执行必须照常完成，不被埋点异常打断"


def test_recorder_absent_zero_behavior():
    """configurable 无 extension_stats_recorder key → 不抛、图正常跑完。"""

    @lc_tool
    async def mcp_srv_tool(x: str) -> str:
        """A fake mcp tool."""
        return "ok"

    llm = _build_fake_llm("mcp_srv_tool", {"x": "1"})
    graph = build_react_graph(llm=llm, tools=[mcp_srv_tool])

    final_state = _run(graph, {})  # 无 key
    from langchain_core.messages import ToolMessage

    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) >= 1


def test_recorder_records_failure_on_tool_error():
    """工具抛错 → record(..., success=False, ...)。"""

    @lc_tool
    async def mcp_srv_tool(x: str) -> str:
        """A fake mcp tool that fails."""
        raise ValueError("tool blew up")

    llm = _build_fake_llm("mcp_srv_tool", {"x": "1"})
    graph = build_react_graph(llm=llm, tools=[mcp_srv_tool])
    recorder = _FakeRecorder()

    _run(graph, {"extension_stats_recorder": recorder})

    assert len(recorder.calls) == 1, f"got {recorder.calls}"
    name, success, _latency = recorder.calls[0]
    assert name == "mcp_srv_tool"
    assert success is False
