"""B10 §7.2 — react_graph 构造点 display 元数据 attach 矩阵.

覆盖: flag-on CALLING/RUNNING/CALLED 三态携带 + flag-off 精确语义 (R12#1)
+ 幻觉名降级 (INV-B10-2) + root/child 同 flag 继承 (R5#5/R7#3)
+ RUNNING 态 flag 叠加矩阵 (R8#1) + 动态 CALLED family 兜底 (R8#3)
+ message_ask_user 特殊 CALLED 点 (:1538/:2537, R2#2).
Harness 镜像 test_react_graph_b1_running.py (option (b): 直调 node 闭包).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.domain.models.app_config import ToolRuntimeConfig
from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import AllowSuccess
from app.domain.services.tools.tool_source_resolver import ToolSource

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class AllowPE:
    async def evaluate(self, call, ctx):
        return AllowSuccess(content="auto", data={})

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


def _state(tool_calls: list[dict] | None = None) -> dict:
    messages = (
        [AIMessage(content="", tool_calls=tool_calls)]
        if tool_calls
        else [HumanMessage(content="hi")]
    )
    return {
        "messages": messages,
        "llm_input_messages": [],
        "step_description": "t", "original_request": "t", "language": "en",
        "attachments": [], "image_content_blocks": [], "events": [],
        "should_interrupt": False, "soft_hint_sent": False,
        "attempt_count": 0, "failure_count": 0,
        "completed_tool_call_prefix": [], "approved_tool_call_ids": [],
        "pending_ask_outcome": None, "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None, "pending_ask_tool_args": None,
    }


def _config(pe=None, ssm=None, queue=None):
    configurable: dict = {"user_id": "u", "session_id": "s", "thread_id": "s"}
    if pe is not None:
        configurable.update(
            permission_engine=pe,
            session_state_machine=ssm,
            tool_confirmation_config=SimpleNamespace(enabled=True),
        )
    if queue is not None:
        configurable["event_queue"] = queue
    return {"configurable": configurable}


def _tc(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _build_fns(
    display_enabled: bool,
    running_enabled: bool = False,
    llm_tool_calls: list[dict] | None = None,
    cfg: ToolRuntimeConfig | None = None,
):
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_read(filepath: str) -> str:
        """Read a file."""
        return "content"

    ai = AIMessage(content="done", tool_calls=llm_tool_calls or [])
    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(return_value=ai)
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)
    graph = build_react_graph(
        stub_llm, [file_read],
        tool_runtime_config=cfg or ToolRuntimeConfig(
            tool_display_metadata_enabled=display_enabled,
            tool_running_events_enabled=running_enabled,
        ),
    )
    return SimpleNamespace(
        llm_node=graph.nodes["llm_node"].bound.afunc,
        tool_node=graph.nodes["tool_node"].bound.afunc,
    )


def _tool_events(update: dict, status: ToolEventStatus) -> list[ToolEvent]:
    # 注意返回形态差异 (react_graph.py:1202-1205): llm_node 返回 dict
    # {"messages", "events"} — 直接传 result; tool_node 返回 Command —
    # 传 result.update (与 b1_running/b1_streaming 消费形态一致).
    return [
        e for e in update.get("events", [])
        if isinstance(e, ToolEvent) and e.status == status
    ]


class TestCallingAttach:
    """构造点 :1171 (llm_node 主路径 CALLING)."""

    async def test_flag_on_calling_carries_source_and_triple(self):
        fns = _build_fns(True, llm_tool_calls=[_tc("file_read", {"filepath": "/x"})])
        result = await fns.llm_node(_state(), {"configurable": {}})
        [evt] = _tool_events(result, ToolEventStatus.CALLING)
        assert evt.tool_source is not None
        assert evt.tool_source.source == "native"
        assert evt.display_icon == "file"
        assert evt.read_only is True
        assert evt.destructive is False

    async def test_flag_off_calling_all_none(self):
        # R12#1: flag-off CALLING 的 tool_source 为 null (现状) + 三元组 null
        fns = _build_fns(False, llm_tool_calls=[_tc("file_read", {"filepath": "/x"})])
        result = await fns.llm_node(_state(), {"configurable": {}})
        [evt] = _tool_events(result, ToolEventStatus.CALLING)
        assert evt.tool_source is None
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None

    async def test_flag_on_hallucinated_name_degrades_not_raises(self):
        # INV-B10-2: resolver 失败永不抛出到事件流
        fns = _build_fns(True, llm_tool_calls=[_tc("no_such_tool_xyz", {})])
        result = await fns.llm_node(_state(), {"configurable": {}})
        [evt] = _tool_events(result, ToolEventStatus.CALLING)
        assert evt.tool_name == "unknown"   # 既有 category 降级保持
        assert evt.tool_source is None
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None


class TestStreamingCallingAttach:
    """构造点 :1122 (B1-2 streaming incremental CALLING, queue 路径).
    Harness 镜像 test_react_graph_b1_streaming.py 的 _build_llm_node/_chunk."""

    @staticmethod
    def _chunk(tccs=None, content=""):
        from langchain_core.messages import AIMessageChunk

        kwargs = {"content": content}
        if tccs is not None:
            kwargs["tool_call_chunks"] = [
                {"type": "tool_call_chunk", **t} for t in tccs
            ]
        return AIMessageChunk(**kwargs)

    def _build_streaming_llm_node(self, display_enabled: bool):
        from langchain_core.tools import tool as lc_tool

        from app.domain.services.graphs.react_graph import build_react_graph

        @lc_tool
        async def file_read(filepath: str) -> str:
            """Read a file."""
            return "ok"

        chunks = [
            self._chunk([{"index": 0, "id": "t1", "name": "file_read",
                          "args": '{"filepath": "/x"}'}]),
        ]
        stub = AsyncMock()

        async def _gen(_messages):
            for c in chunks:
                yield c

        stub.astream = MagicMock(side_effect=lambda m: _gen(m))
        stub.ainvoke = AsyncMock(
            side_effect=AssertionError("streaming ON must not call ainvoke")
        )
        stub.bind_tools = MagicMock(return_value=stub)
        graph = build_react_graph(
            stub, [file_read],
            tool_runtime_config=ToolRuntimeConfig(
                llm_tool_call_streaming_enabled=True,
                llm_incremental_calling_events_enabled=True,
                tool_display_metadata_enabled=display_enabled,
            ),
        )
        return graph.nodes["llm_node"].bound.afunc

    async def test_streaming_incremental_calling_carries_metadata(self):
        queue: asyncio.Queue = asyncio.Queue()
        llm_node = self._build_streaming_llm_node(True)
        await llm_node(_state(), {"configurable": {"event_queue": queue}})
        evt = queue.get_nowait()
        assert evt.status == ToolEventStatus.CALLING
        assert evt.tool_call_id == "t1"
        assert evt.tool_source is not None
        assert evt.display_icon == "file"
        assert evt.read_only is True
        assert evt.destructive is False

    async def test_streaming_incremental_calling_flag_off_all_none(self):
        queue: asyncio.Queue = asyncio.Queue()
        llm_node = self._build_streaming_llm_node(False)
        await llm_node(_state(), {"configurable": {"event_queue": queue}})
        evt = queue.get_nowait()
        assert evt.status == ToolEventStatus.CALLING
        assert evt.tool_source is None
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None


class TestCalledAttach:
    """构造点 :680 (CALLED 主路径, 经 _translate_outcome 参数)."""

    async def test_flag_on_called_carries_triple(self):
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("file_read", {"filepath": "/x"})]),
            _config(AllowPE(), _ssm()),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        assert evt.tool_source is not None       # 既有行为保持
        assert evt.display_icon == "file"
        assert evt.read_only is True
        assert evt.destructive is False

    async def test_flag_off_called_keeps_tool_source_nulls_triple(self):
        # R12#1 精确语义: CALLED 的 tool_source 保持既有值 (F0.6,
        # _translate_outcome 现状无条件填写), 仅 display 三元组 null
        fns = _build_fns(False)
        result = await fns.tool_node(
            _state([_tc("file_read", {"filepath": "/x"})]),
            _config(AllowPE(), _ssm()),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        assert evt.tool_source is not None       # flag-off 不得改变它 (badge 可用)
        assert evt.tool_source.category == "file"
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None

    async def test_root_child_same_config_same_behavior(self):
        # R5#5/R7#3: 同一 ToolRuntimeConfig 构建两个 graph (模拟根/子 runner)
        cfg = ToolRuntimeConfig(tool_display_metadata_enabled=True)
        results = []
        for _ in range(2):
            fns = _build_fns(True, cfg=cfg)
            result = await fns.tool_node(
                _state([_tc("file_read", {"filepath": "/x"})]),
                _config(AllowPE(), _ssm()),
            )
            [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
            results.append(
                (evt.display_icon, evt.read_only, evt.destructive)
            )
        assert results[0] == results[1] == ("file", True, False)


class TestRunningFlagMatrix:
    """构造点 :1468 — R8#1: RUNNING 发射本身受 B1 flag 门控, 必须叠加测试."""

    async def test_both_flags_on_running_carries_metadata(self):
        queue: asyncio.Queue = asyncio.Queue()
        fns = _build_fns(True, running_enabled=True)
        await fns.tool_node(
            _state([_tc("file_read", {"filepath": "/x"})]),
            _config(AllowPE(), _ssm(), queue),
        )
        evt = queue.get_nowait()
        assert evt.status == ToolEventStatus.RUNNING
        assert evt.tool_source is not None
        assert evt.display_icon == "file"
        assert evt.read_only is True
        assert evt.destructive is False

    async def test_running_flag_off_no_running_event(self):
        queue: asyncio.Queue = asyncio.Queue()
        fns = _build_fns(True, running_enabled=False)
        result = await fns.tool_node(
            _state([_tc("file_read", {"filepath": "/x"})]),
            _config(AllowPE(), _ssm(), queue),
        )
        assert queue.empty()
        assert _tool_events(result.update, ToolEventStatus.RUNNING) == []

    async def test_display_off_running_on_running_exists_triple_null(self):
        queue: asyncio.Queue = asyncio.Queue()
        fns = _build_fns(False, running_enabled=True)
        await fns.tool_node(
            _state([_tc("file_read", {"filepath": "/x"})]),
            _config(AllowPE(), _ssm(), queue),
        )
        evt = queue.get_nowait()
        assert evt.status == ToolEventStatus.RUNNING
        assert evt.tool_source is None            # RUNNING 现状 (F0.6)
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None


class TestMessageAskUserCalledAttach:
    """构造点 :1538 (PE 路径) 与 :2537 (legacy 路径) — R2#2 取消豁免."""

    async def test_pe_path_mau_called_carries_metadata(self):
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?"})]),
            _config(AllowPE(), _ssm()),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        assert evt.function_name == "message_ask_user"
        assert evt.display_icon == "message"
        assert evt.read_only is False
        assert evt.destructive is False
        assert evt.tool_source is not None

    async def test_legacy_path_mau_called_carries_metadata(self):
        # 无 permission_engine → legacy 直接路径 (:2537)
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?"})]),
            _config(),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        assert evt.function_name == "message_ask_user"
        assert evt.display_icon == "message"

    async def test_legacy_path_mau_flag_off_all_none(self):
        # INV-B10-0: legacy 冻结面 flag-off no-op (B1 合同不受影响)
        fns = _build_fns(False)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?"})]),
            _config(),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        assert evt.tool_source is None
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None


class TestMessageAskUserSoftHintWireContract:
    """SOFT_HINT 软门控 vs WAITING_FOR_USER 真阻塞的 wire 区分面.

    FE (ui/src/lib/session-ui.ts getToolDisplayCopy) 依赖 envelope
    function_result.message 上的这两个哨兵值把软门控渲染成非阻塞提示卡
    (kind:"hint")、真阻塞保持提问卡 (kind:"ask")。此处把两条 gate 路径
    构造的 ToolEvent → ToolEventEnvelopeV1 投影钉死, 后端改动若使哨兵
    不再上 wire, 这里先红。
    """

    @staticmethod
    def _project(evt: ToolEvent):
        from app.application.services.tool_event_envelope_v1 import (
            project_tool_event_to_envelope_v1,
        )
        return project_tool_event_to_envelope_v1(evt)

    async def test_pe_path_first_ask_soft_hint_on_wire(self):
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?"})]),
            _config(AllowPE(), _ssm()),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        env = self._project(evt)
        assert env.function_result is not None
        assert env.function_result.message == "SOFT_HINT"
        assert env.function_result.status == "ok"

    async def test_legacy_path_first_ask_soft_hint_on_wire(self):
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?"})]),
            _config(),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        env = self._project(evt)
        assert env.function_result is not None
        assert env.function_result.message == "SOFT_HINT"

    async def test_pe_path_takeover_ask_waiting_on_wire(self):
        fns = _build_fns(True)
        result = await fns.tool_node(
            _state([_tc("message_ask_user", {"text": "?", "suggest_user_takeover": "browser"})]),
            _config(AllowPE(), _ssm()),
        )
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        env = self._project(evt)
        assert env.function_result is not None
        assert env.function_result.message == "WAITING_FOR_USER"

    async def test_legacy_path_second_ask_waiting_on_wire(self):
        # soft_hint_sent=True（react_graph.py:2496 state 派生）→ 第二次
        # ask 必须投影真阻塞哨兵
        fns = _build_fns(True)
        state = _state([_tc("message_ask_user", {"text": "?"})])
        state["soft_hint_sent"] = True
        result = await fns.tool_node(state, _config())
        [evt] = _tool_events(result.update, ToolEventStatus.CALLED)
        env = self._project(evt)
        assert env.function_result is not None
        assert env.function_result.message == "WAITING_FOR_USER"


class TestTranslateOutcomeDirect:
    """CALLED 点直调面: 动态 family 兜底 (R8#3) + 默认参零改动."""

    async def test_dynamic_mcp_called_family_fallback(self):
        # 未知动态 function + existing ToolSource(mcp) →
        # display_icon="mcp" + 策略位 family fallback (锁定"已有
        # tool_source 时仍跑 registry"语义)
        from app.domain.services.graphs.react_graph import _translate_outcome

        src = ToolSource(source="mcp", category="mcp", canonical_name="weather_lookup")
        _, _, events = await _translate_outcome(
            AllowSuccess(content="ok", data={}),
            _tc("weather_lookup", {}, "c9"),
            src,
            None,
            tool_result_max_chars=8000,
            guide_injector=None,
            display_metadata_enabled=True,
        )
        [evt] = [e for e in events if isinstance(e, ToolEvent)]
        assert evt.tool_source is src
        assert evt.display_icon == "mcp"
        assert evt.read_only is False
        assert evt.destructive is False

    async def test_default_kwarg_preserves_existing_direct_calls(self):
        # display_metadata_enabled 默认 False → 既有直调测试零改动 (§4.2)
        from app.domain.services.graphs.react_graph import _translate_outcome

        src = ToolSource(source="native", category="file", canonical_name="file_read")
        _, _, events = await _translate_outcome(
            AllowSuccess(content="ok", data={}),
            _tc("file_read", {"filepath": "/x"}, "c8"),
            src,
            None,
            tool_result_max_chars=8000,
            guide_injector=None,
        )
        [evt] = [e for e in events if isinstance(e, ToolEvent)]
        assert evt.tool_source is src          # 现状保持
        assert evt.display_icon is None
        assert evt.read_only is None
        assert evt.destructive is None
