"""Task 20 — PlannerReActFlow 把 extension_stats_recorder 穿进 configurable（B9 PR-3）。

五跳链的最后一跳：`_build_config()` 必须把 ctor 收到的 recorder 逐字放到
`cfg["configurable"]["extension_stats_recorder"]`，react_graph 埋点从这里读取。
recorder 未注入（默认 None）时 key 值为 None（flag off = 零调用语义）。
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig

from tests.conftest import TEST_USER_ID_FIXED


def _make_flow(**overrides):
    """Create a PlannerReActFlow with all required mocks（对齐 test_flush_gate._make_flow）。"""
    from app.domain.services.flows.planner_react import PlannerReActFlow

    kwargs = {
        "llm": MagicMock(),
        "agent_config": AgentConfig(),
        "session_id": "test-session",
        "user_id": TEST_USER_ID_FIXED,
        "uow_factory": AsyncMock(),
        "browser_accessor": EagerBrowserAccessor(MagicMock()),
        "sandbox_accessor": EagerSandboxAccessor(MagicMock()),
        "search_engine": MagicMock(),
        "mcp_tool": MagicMock(),
        "a2a_tool": MagicMock(),
        "skill_tool": MagicMock(),
    }
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


def test_planner_react_threads_recorder_into_configurable():
    sentinel = object()
    flow = _make_flow(extension_stats_recorder=sentinel)
    cfg = flow._build_config()
    assert cfg["configurable"]["extension_stats_recorder"] is sentinel


def test_planner_react_recorder_defaults_none_into_configurable():
    """默认（不传 recorder）→ configurable 值为 None（flag off 语义）。"""
    flow = _make_flow()
    cfg = flow._build_config()
    assert cfg["configurable"]["extension_stats_recorder"] is None
