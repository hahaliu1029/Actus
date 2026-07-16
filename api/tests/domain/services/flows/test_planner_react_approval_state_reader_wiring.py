"""R5b-2 wiring 回归：``approval_state_reader`` 从 DI 注入 → 存 self →
configurable 下发给 react_graph 的 tool_node。

锁死三个契约：
1. 默认参数 ``approval_state_reader=None``（向后兼容现有 16 个 test fixture）
2. ``__init__`` 保存到 ``self._approval_state_reader``
3. ``_build_config()["configurable"]["approval_state_reader"]`` 是同一个对象
   （react_graph 的 tool_node 通过 ``configurable.get("approval_state_reader")`` 读）
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig
from app.domain.services.flows.planner_react import PlannerReActFlow


def _make_flow(*, approval_state_reader=None) -> PlannerReActFlow:
    """Construct a minimal PlannerReActFlow for configurable assertions.

    Follows the same mock pattern as ``test_planner_react_di_gate.py``
    — only the constructor signature shape matters; no flow execution.
    """
    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(
            max_iterations=10, max_retries=3, max_search_results=5
        ),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        approval_state_reader=approval_state_reader,
        _allow_default_prompt_assembler=True,
    )


def test_default_approval_state_reader_is_none() -> None:
    """Bare construction → ``_approval_state_reader is None``（向后兼容）。"""
    flow = _make_flow()
    assert flow._approval_state_reader is None


def test_constructor_stores_approval_state_reader() -> None:
    """Injected reader is stored on the instance for later configurable exposure."""
    sentinel = object()
    flow = _make_flow(approval_state_reader=sentinel)
    assert flow._approval_state_reader is sentinel


def test_build_config_exposes_approval_state_reader_in_configurable() -> None:
    """``_build_config()["configurable"]["approval_state_reader"]`` 就是注入对象。

    react_graph tool_node 通过 ``configurable.get("approval_state_reader")``
    读取——任何绑定断裂都会让 pre-check fall-through 到 confirmation interrupt
    并让 I5/I7 invariant 静默回退，必须锁死。
    """
    sentinel = object()
    flow = _make_flow(approval_state_reader=sentinel)
    cfg = flow._build_config()
    assert cfg["configurable"]["approval_state_reader"] is sentinel


def test_build_config_passes_none_when_reader_not_injected() -> None:
    """未注入时 configurable 里是 None——tool_node 守卫 (if reader and ...) 自然 fall-through。"""
    flow = _make_flow()  # 默认 None
    cfg = flow._build_config()
    assert cfg["configurable"]["approval_state_reader"] is None
