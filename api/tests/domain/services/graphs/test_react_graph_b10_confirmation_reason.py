"""B10 §7.4 — PE Asked reason → ToolConfirmationEvent.decision_reason 直通.

INV-B10-9: 不受 tool_display_metadata_enabled 门控 (测试显式 flag-off);
risk_reason 现有降维行为回归锁定 (F0.11); pre-existing 字段快照相等 (R10#6).
Harness 镜像 test_react_graph_b1_running.py (option (b), P-10 私有 fixture).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.app_config import ToolRuntimeConfig
from app.domain.models.event import ToolConfirmationEvent
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import Asked, DecisionReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


_REASON = DecisionReason(
    type="smart_approve", code="llm_escalate", message="needs human review"
)


class AskedPE:
    async def evaluate(self, call, ctx):
        return Asked(content="waiting for user", reason=_REASON)

    async def preflight_resume(self, *a, **kw):
        pass

    async def commit_resume(self, *a, **kw):
        pass


def _ssm():
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 1)
    )
    return ssm


def _state() -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "id": "t1", "name": "file_write",
            "args": {"path": "/x", "content": "y"}, "type": "tool_call",
        }])],
        "llm_input_messages": [],
        "step_description": "t", "original_request": "t", "language": "en",
        "attachments": [], "image_content_blocks": [], "events": [],
        "should_interrupt": False, "soft_hint_sent": False,
        "attempt_count": 0, "failure_count": 0,
        "completed_tool_call_prefix": [], "approved_tool_call_ids": [],
        "pending_ask_outcome": None, "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None, "pending_ask_tool_args": None,
    }


def _build_tool_node():
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return f"wrote {path}"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)
    graph = build_react_graph(
        stub_llm, [file_write],
        # INV-B10-9 关键: display flag 显式 OFF, decision_reason 仍直通
        tool_runtime_config=ToolRuntimeConfig(tool_display_metadata_enabled=False),
    )
    return graph.nodes["tool_node"].bound.afunc


async def _run_asked_and_get_confirmation() -> ToolConfirmationEvent:
    queue: asyncio.Queue = asyncio.Queue()
    tool_node = _build_tool_node()
    config = {"configurable": {
        "permission_engine": AskedPE(),
        "session_state_machine": _ssm(),
        "tool_confirmation_config": SimpleNamespace(enabled=True),
        "user_id": "u", "session_id": "s", "thread_id": "s",
        "event_queue": queue,
    }}
    await tool_node(_state(), config)
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    confs = [e for e in drained if isinstance(e, ToolConfirmationEvent)]
    assert confs, f"no ToolConfirmationEvent in queue, got {drained!r}"
    return confs[0]


class TestDecisionReasonPassthrough:
    async def test_pe_asked_reason_flows_structurally(self):
        evt = await _run_asked_and_get_confirmation()
        assert evt.decision_reason == _REASON  # 三字段整体直通, 不再只取 message

    async def test_risk_reason_dimension_reduction_preserved(self):
        # F0.11 现状保持: risk_reason = _pe_reason.message or assessment 降维
        evt = await _run_asked_and_get_confirmation()
        assert evt.risk_reason == "needs human review"

    async def test_pre_existing_fields_snapshot(self):
        # R10#6: flag-off + PE Asked → 所有 pre-existing 字段与现状一致
        evt = await _run_asked_and_get_confirmation()
        assert evt.tool_call_id == "t1"
        assert evt.tool_name == "file_write"
        assert evt.tool_args == {"path": "/x", "content": "y"}
        assert evt.risk_level == "medium"       # detail/assessment 缺失兜底 (:2127-2130)
        assert evt.matched_patterns == []
        assert evt.suggested_alternative is None
        assert evt.approval_options == ["once", "session", "always", "deny"]
        assert evt.timeout_seconds == 300        # configurable 默认 (:2108-2110)


def test_domain_event_default_none():
    # 老构造代码不传 decision_reason → None (additive)
    evt = ToolConfirmationEvent(
        tool_call_id="t0", tool_name="shell_execute", tool_args={},
        risk_level="high", risk_reason="r", matched_patterns=[],
        timeout_seconds=60,
    )
    assert evt.decision_reason is None
