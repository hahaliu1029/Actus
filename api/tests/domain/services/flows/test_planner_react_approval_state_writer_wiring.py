"""R5b-3 wiring 回归：``approval_state_writer`` 从 DI 注入 → 存 self →
configurable 下发给 react_graph 的 tool_node（SmartApprove 写路径消费）。

锁死三个契约（对称 reader wiring）：
1. 默认参数 ``approval_state_writer=None``（向后兼容现有 test fixture）
2. ``__init__`` 保存到 ``self._approval_state_writer``
3. ``_build_config()["configurable"]["approval_state_writer"]`` 是同一个对象
   （react_graph 的 SmartApprove 分支通过 ``configurable.get("approval_state_writer")``
   调 writer.write(...) 持久化 grant）
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.domain.models.app_config import AgentConfig
from app.domain.services.flows.planner_react import PlannerReActFlow


def _make_flow(*, approval_state_writer=None) -> PlannerReActFlow:
    """Construct a minimal PlannerReActFlow for configurable assertions."""
    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(
            max_iterations=10, max_retries=3, max_search_results=5
        ),
        session_id="test-session",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        approval_state_writer=approval_state_writer,
        _allow_default_prompt_assembler=True,
    )


def test_default_approval_state_writer_is_none() -> None:
    """Bare construction → ``_approval_state_writer is None``（向后兼容）。"""
    flow = _make_flow()
    assert flow._approval_state_writer is None


def test_constructor_stores_approval_state_writer() -> None:
    """Injected writer is stored on the instance for configurable exposure."""
    sentinel = object()
    flow = _make_flow(approval_state_writer=sentinel)
    assert flow._approval_state_writer is sentinel


def test_build_config_exposes_approval_state_writer_in_configurable() -> None:
    """``_build_config()["configurable"]["approval_state_writer"]`` 就是注入对象。

    SmartApprove 写路径通过 ``configurable.get("approval_state_writer")``
    拿到 Writer 并调 ``writer.write(ApprovalDecision(..., source_type='smart_approve'))``
    持久化 grant 行 —— AST guard Rule 7（R5b-4 翻开）依赖 configurable 下游是
    Writer 而不是 ApprovalCache。
    """
    sentinel = object()
    flow = _make_flow(approval_state_writer=sentinel)
    cfg = flow._build_config()
    assert cfg["configurable"]["approval_state_writer"] is sentinel


def test_build_config_passes_none_when_writer_not_injected() -> None:
    """未注入时 configurable 里是 None；SmartApprove 守卫 fail-open 跳过持久化。"""
    flow = _make_flow()
    cfg = flow._build_config()
    assert cfg["configurable"]["approval_state_writer"] is None
