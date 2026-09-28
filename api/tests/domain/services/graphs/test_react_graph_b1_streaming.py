"""B1-2 llm_node astream branch (spec §5.1/§5.3): incremental CALLING,
per-attempt exclusion set, #4 llm_chunk_boundary live, MessageEvent rule."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.domain.models.app_config import ToolRuntimeConfig
from app.domain.models.event import MessageEvent, ToolEvent, ToolEventStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _chunk(tccs=None, content=""):
    kwargs = {"content": content}
    if tccs is not None:
        kwargs["tool_call_chunks"] = [{"type": "tool_call_chunk", **t} for t in tccs]
    return AIMessageChunk(**kwargs)


def _tcc(index, *, id=None, name=None, args=None):
    return {"index": index, "id": id, "name": name, "args": args}


TWO_CALL_CHUNKS = [
    _chunk([_tcc(0, id="t1", name="file_write", args='{"path": ')]),
    _chunk([_tcc(0, args='"/x"}')]),
    _chunk([_tcc(1, id="t2", name="file_read", args='{"path": "/y"}')]),
]


def _build_llm_node(*, streaming: bool, incremental: bool,
                    chunks=None, on_yield=None):
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        return "ok"

    @lc_tool
    async def file_read(path: str) -> str:
        """Read a file."""
        return "ok"

    stub = AsyncMock()

    async def _gen(_messages):
        for i, c in enumerate(chunks or []):
            if on_yield is not None:
                on_yield(i)
            yield c

    stub.astream = MagicMock(side_effect=lambda m: _gen(m))
    stub.ainvoke = AsyncMock(
        side_effect=AssertionError("streaming ON must not call ainvoke")
    )
    stub.bind_tools = MagicMock(return_value=stub)
    graph = build_react_graph(
        stub, [file_write, file_read],
        tool_runtime_config=ToolRuntimeConfig(
            llm_tool_call_streaming_enabled=streaming,
            llm_incremental_calling_events_enabled=incremental,
        ),
    )
    return graph.nodes["llm_node"].bound.afunc


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="go")],
        "llm_input_messages": [],
        "step_description": "t", "original_request": "t", "language": "en",
        "attachments": [], "image_content_blocks": [], "events": [],
        "should_interrupt": False, "soft_hint_sent": False,
        "attempt_count": 0, "failure_count": 0,
        "completed_tool_call_prefix": [], "approved_tool_call_ids": [],
        "pending_ask_outcome": None, "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None, "pending_ask_tool_args": None,
    }


class TestIncrementalCalling:
    async def test_streaming_on_incremental_on_calling_via_queue(self):
        queue: asyncio.Queue = asyncio.Queue()
        llm_node = _build_llm_node(streaming=True, incremental=True,
                                   chunks=TWO_CALL_CHUNKS)
        result = await llm_node(_state(), {"configurable": {"event_queue": queue}})
        queued = []
        while not queue.empty():
            queued.append(queue.get_nowait())
        calling = [e for e in queued
                   if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLING]
        assert [e.tool_call_id for e in calling] == ["t1", "t2"]
        # 单通道排除：已 queue 发射的 CALLING 不再出现在节点尾 state events
        state_calling = [e for e in result["events"]
                        if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLING]
        assert state_calling == []
        # 执行输入 = 权威合并消息（两 tool_calls 完整）
        assert [tc["id"] for tc in result["messages"][0].tool_calls] == ["t1", "t2"]

    async def test_streaming_on_incremental_off_tail_calling_state_path_matches_c2(self):
        """R14#3：streaming ON + incremental OFF → 节点尾批量 state-path CALLING
        （C2 同形），queue 零发射。"""
        queue: asyncio.Queue = asyncio.Queue()
        llm_node = _build_llm_node(streaming=True, incremental=False,
                                   chunks=TWO_CALL_CHUNKS)
        result = await llm_node(_state(), {"configurable": {"event_queue": queue}})
        assert queue.empty()
        calling = [e for e in result["events"]
                   if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLING]
        assert [e.tool_call_id for e in calling] == ["t1", "t2"]

    async def test_queue_missing_per_attempt_fallback_state_path(self):
        llm_node = _build_llm_node(streaming=True, incremental=True,
                                   chunks=TWO_CALL_CHUNKS)
        result = await llm_node(_state(), {"configurable": {}})
        calling = [e for e in result["events"]
                   if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLING]
        assert [e.tool_call_id for e in calling] == ["t1", "t2"], \
            "queue 缺失 → 全量节点尾 state-path，无混合通道"

    async def test_mixed_content_and_tool_calls_no_message_event(self):
        """R10#1：最终合并消息 tool_calls 非空 ⇒ 不发 MessageEvent，
        content 只留在 AIMessage。"""
        chunks = [_chunk(None, content="thinking...")] + TWO_CALL_CHUNKS
        llm_node = _build_llm_node(streaming=True, incremental=True, chunks=chunks)
        queue: asyncio.Queue = asyncio.Queue()
        result = await llm_node(_state(), {"configurable": {"event_queue": queue}})
        assert not [e for e in result["events"] if isinstance(e, MessageEvent)]
        assert result["messages"][0].content == "thinking..."

    async def test_duplicate_calling_bounded_by_retry(self):
        """有界重复（≤ max_attempts × 唯一 id 数）：排除集 per-attempt——
        两次 node 调用（模拟 RetryPolicy 两个 attempt）各发一次同 id CALLING。"""
        queue: asyncio.Queue = asyncio.Queue()
        llm_node = _build_llm_node(streaming=True, incremental=True,
                                   chunks=TWO_CALL_CHUNKS)
        await llm_node(_state(), {"configurable": {"event_queue": queue}})
        await llm_node(_state(), {"configurable": {"event_queue": queue}})
        t1_count = 0
        while not queue.empty():
            e = queue.get_nowait()
            if isinstance(e, ToolEvent) and e.tool_call_id == "t1":
                t1_count += 1
        assert t1_count == 2, "per-attempt 排除集：跨 attempt 重复是 accepted 且有界"


class TestChunkBoundaryCancel:
    async def test_truncated_tool_arguments_fail_before_returning_execution_state(self):
        from app.application.errors.exceptions import ServerRequestsError

        queue: asyncio.Queue = asyncio.Queue()
        llm_node = _build_llm_node(
            streaming=True, incremental=True,
            chunks=[_chunk([_tcc(0, id="t1", name="file_write", args='{"path":"/x"')])],
        )
        with pytest.raises(ServerRequestsError, match="incomplete streamed tool arguments"):
            await llm_node(_state(), {"configurable": {"event_queue": queue}})
        assert queue.empty()

    async def test_chunk_boundary_cancel_behavior(self):
        """R14#1 P1（行为测试，替代纯静态 tripwire）+ R15#1 前置：
        streaming ON + incremental ON + queue 存在；首 chunk 含不完整 tool_call。
        首 chunk 消费后翻转 cancel_event → 第二 chunk 不被消费、
        CancelledByEventError("llm_chunk_boundary") 抛出、零 CALLING/MessageEvent。"""
        from app.domain.services.graphs.react_graph import CancelledByEventError

        cancel_event = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()
        chunks = [
            _chunk([_tcc(0, id="t1", name="file_write", args='{"pa')]),  # 不完整
            _chunk([_tcc(0, args='th": "/x"}')]),
        ]

        def on_yield(i: int) -> None:
            if i == 1:
                cancel_event.set()  # 第二 chunk 产出前置位 → 顶部 checkpoint 拦截

        llm_node = _build_llm_node(streaming=True, incremental=True,
                                   chunks=chunks, on_yield=on_yield)
        with pytest.raises(CancelledByEventError) as exc_info:
            await llm_node(_state(), {
                "configurable": {"event_queue": queue, "cancel_event": cancel_event},
            })
        assert "llm_chunk_boundary" in str(exc_info.value)
        assert queue.empty(), "checkpoint 后零 CALLING（首 chunk 的 call 不完整）"
