"""B1-1b RUNNING emission semantics (spec §4.2 + §4.1.1 R7#3).

RUNNING means "about to actually invoke" — emitted inside the PE execute
thunk AFTER the live-mode recheck passes, BEFORE _invoke_wrapper. A
mode-denied thunk must emit NO RUNNING (R14#2). flag OFF = zero change.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from app.domain.models.app_config import ToolRuntimeConfig
from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import AllowSuccess

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class AllowPE:
    def __init__(self):
        self.calls = []

    async def evaluate(self, call, ctx):
        self.calls.append(call)
        return AllowSuccess(content="auto", data={})

    async def preflight_resume(self, *a, **kw):
        pass

    async def commit_resume(self, *a, **kw):
        pass


def _ssm(modes: list):
    """Sequenced SSM: one (mode, rev) per get_mode_with_revision call.
    Per-tc gate read + thunk fresh recheck = 2 reads for a single tc."""
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(side_effect=list(modes))
    return ssm


def _state(call_id: str = "t1") -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=[{
            "id": call_id, "name": "file_write",
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


def _config(pe, ssm, queue=None):
    configurable: dict = {
        "permission_engine": pe,
        "session_state_machine": ssm,
        "tool_confirmation_config": SimpleNamespace(enabled=True),
        "user_id": "u", "session_id": "s", "thread_id": "s",
    }
    if queue is not None:
        configurable["event_queue"] = queue
    return {"configurable": configurable}


def _build_tool_node(running_enabled: bool, probe: list | None = None,
                     queue: asyncio.Queue | None = None):
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    probe_list = probe if probe is not None else []

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        # 执行时刻探针：RUNNING 必须已在 queue 里（先于 wrapper）
        probe_list.append(queue.qsize() if queue is not None else -1)
        return f"wrote {path}"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)
    graph = build_react_graph(
        stub_llm, [file_write],
        tool_runtime_config=ToolRuntimeConfig(
            tool_running_events_enabled=running_enabled
        ),
    )
    return graph.nodes["tool_node"].bound.afunc


class TestEnumAndFlagExist:
    """R3#2：显式存在性断言——给 Step 2 一个不依赖行为断言顺序的干净 RED。"""

    def test_running_enum_member_exists(self):
        assert getattr(ToolEventStatus, "RUNNING", None) == "running"

    def test_flag_field_exists(self):
        assert "tool_running_events_enabled" in ToolRuntimeConfig.model_fields


class TestRunningEmission:
    async def test_running_queued_after_recheck_before_wrapper(self):
        queue: asyncio.Queue = asyncio.Queue()
        probe: list = []
        tool_node = _build_tool_node(True, probe, queue)
        ssm = _ssm([(SessionStatus.RUNNING, 1), (SessionStatus.RUNNING, 2)])
        result = await tool_node(_state(), _config(AllowPE(), ssm, queue))
        # wrapper 执行时 queue 已含 RUNNING（发射先于执行）
        assert probe == [1]
        assert queue.qsize() == 1
        evt = queue.get_nowait()
        assert isinstance(evt, ToolEvent)
        assert evt.status == ToolEventStatus.RUNNING
        assert evt.tool_call_id == "t1"
        # 单通道：state events 只有 CALLED，无 RUNNING
        statuses = [e.status for e in result.update["events"] if isinstance(e, ToolEvent)]
        assert statuses == [ToolEventStatus.CALLED]

    async def test_mode_denied_thunk_emits_no_running(self):
        """R14#2 + R15#1 前置：flag ON + queue 存在，否则「零 RUNNING」空真。
        SSM：gate 读 RUNNING、thunk 重读 TAKEOVER → wrapper 不执行、零 RUNNING。"""
        queue: asyncio.Queue = asyncio.Queue()
        probe: list = []
        tool_node = _build_tool_node(True, probe, queue)
        ssm = _ssm([(SessionStatus.RUNNING, 1), (SessionStatus.TAKEOVER, 2)])
        result = await tool_node(_state(), _config(AllowPE(), ssm, queue))
        assert probe == [], "mode-denied thunk must not invoke the wrapper"
        assert queue.empty(), "mode-denied thunk must emit NO RUNNING"
        msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert msgs and "MODE_DENIED" in str(msgs[0].content)

    async def test_flag_off_zero_running(self):
        queue: asyncio.Queue = asyncio.Queue()
        tool_node = _build_tool_node(False, None, queue)
        ssm = _ssm([(SessionStatus.RUNNING, 1), (SessionStatus.RUNNING, 2)])
        result = await tool_node(_state(), _config(AllowPE(), ssm, queue))
        assert queue.empty()
        statuses = [e.status for e in result.update["events"] if isinstance(e, ToolEvent)]
        assert ToolEventStatus.RUNNING not in statuses

    async def test_queue_missing_falls_back_to_state_path(self):
        """P-9：queue 缺失 → per-attempt 回退，RUNNING 经 PerTcResult.events
        批尾返回（不丢失），且先于同 tc 的 CALLED。"""
        tool_node = _build_tool_node(True, None, None)
        ssm = _ssm([(SessionStatus.RUNNING, 1), (SessionStatus.RUNNING, 2)])
        result = await tool_node(_state(), _config(AllowPE(), ssm, None))
        statuses = [e.status for e in result.update["events"] if isinstance(e, ToolEvent)]
        assert statuses == [ToolEventStatus.RUNNING, ToolEventStatus.CALLED]

    async def test_legacy_path_never_emits_running(self):
        """legacy 冻结面：flag ON 也零 RUNNING（enhancements_enabled=False）。"""
        queue: asyncio.Queue = asyncio.Queue()
        tool_node = _build_tool_node(True, None, queue)
        # 无 permission_engine → legacy 直接执行路径
        result = await tool_node(
            _state(), {"configurable": {"user_id": "u", "session_id": "s",
                                        "event_queue": queue}}
        )
        assert queue.empty()
        statuses = [e.status for e in result.update["events"] if isinstance(e, ToolEvent)]
        assert ToolEventStatus.RUNNING not in statuses
