"""B1-1c graph-level concurrency wiring (spec §4.3/§4.4/§4.5).

Deterministic asyncio.Event-controlled tools; asserts ordered milestones,
never wall-clock. Fixture 复用 P-10 惯例（私有副本）。"""
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


def _ssm(mode=SessionStatus.RUNNING):
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(mode, 1))
    return ssm


def _flip_ssm(gate_reads: int, flipped=SessionStatus.TAKEOVER):
    """前 gate_reads 次读返回 RUNNING，之后返回 flipped（mode-flip 场景）。"""
    counter = {"n": 0}

    async def read(_sid):
        counter["n"] += 1
        if counter["n"] <= gate_reads:
            return (SessionStatus.RUNNING, counter["n"])
        return (flipped, counter["n"])

    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(side_effect=read)
    return ssm


def _config(pe, ssm, queue=None, extra: dict | None = None):
    configurable: dict = {
        "permission_engine": pe,
        "session_state_machine": ssm,
        "tool_confirmation_config": SimpleNamespace(enabled=True),
        "user_id": "u", "session_id": "s", "thread_id": "s",
    }
    if queue is not None:
        configurable["event_queue"] = queue
    if extra:
        configurable.update(extra)
    return {"configurable": configurable}


def _tc(name: str, args: dict, call_id: str) -> dict:
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _state(tool_calls: list[dict]) -> dict:
    return {
        "messages": [AIMessage(content="", tool_calls=tool_calls)],
        "llm_input_messages": [],
        "step_description": "t", "original_request": "t", "language": "en",
        "attachments": [], "image_content_blocks": [], "events": [],
        "should_interrupt": False, "soft_hint_sent": False,
        "attempt_count": 0, "failure_count": 0,
        "completed_tool_call_prefix": [], "approved_tool_call_ids": [],
        "pending_ask_outcome": None, "pending_ask_tool_call_id": None,
        "pending_ask_artifact": None, "pending_ask_tool_args": None,
    }


def _build(record: list, *, concurrency=True, running=False, max_concurrency=3,
           evt_a: asyncio.Event | None = None, evt_b: asyncio.Event | None = None):
    """shell_read_output（安全集）×2 + file_write（互斥）。
    slow 实例等 evt_a；fast 实例完成时 set evt_a。"""
    from langchain_core.tools import tool as lc_tool

    from app.domain.services.graphs.react_graph import build_react_graph

    @lc_tool
    async def shell_read_output(session_id: str = "") -> str:
        """Read shell output."""
        record.append(("start", session_id))
        if session_id == "slow" and evt_a is not None:
            await evt_a.wait()
        record.append(("end", session_id))
        if session_id == "fast" and evt_a is not None:
            evt_a.set()
        if evt_b is not None and session_id == "fast":
            evt_b.set()
        return f"out-{session_id}"

    @lc_tool
    async def file_write(path: str, content: str = "") -> str:
        """Write to a file."""
        record.append(("start", "file_write"))
        record.append(("end", "file_write"))
        return f"wrote {path}"

    stub_llm = AsyncMock()
    stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    stub_llm.bind_tools = MagicMock(return_value=stub_llm)
    graph = build_react_graph(
        stub_llm, [shell_read_output, file_write],
        tool_runtime_config=ToolRuntimeConfig(
            tool_concurrency_enabled=concurrency,
            tool_max_concurrency=max_concurrency,
            tool_running_events_enabled=running,
        ),
    )
    return graph.nodes["tool_node"].bound.afunc


class TestWindowThroughGraph:
    async def test_safe_tools_overlap_and_message_order(self):
        record: list = []
        evt_a = asyncio.Event()
        tool_node = _build(record, evt_a=evt_a)
        state = _state([
            _tc("shell_read_output", {"session_id": "slow"}, "t1"),
            _tc("shell_read_output", {"session_id": "fast"}, "t2"),
        ])
        result = await tool_node(state, _config(AllowPE(), _ssm()))
        assert record.index(("start", "fast")) < record.index(("end", "slow")), \
            "安全集工具必须在窗口内重叠"
        msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in msgs] == ["t1", "t2"]

    async def test_mutually_exclusive_drains_first(self):
        record: list = []
        evt_a = asyncio.Event()
        evt_a.set()  # slow 不阻塞（互斥 drain 语义仍可由顺序断言）
        tool_node = _build(record, evt_a=evt_a)
        state = _state([
            _tc("shell_read_output", {"session_id": "slow"}, "t1"),
            _tc("file_write", {"path": "/a", "content": "x"}, "t2"),
        ])
        await tool_node(state, _config(AllowPE(), _ssm()))
        assert record.index(("end", "slow")) < record.index(("start", "file_write"))

    async def test_flag_off_serial_c4_shape(self):
        record: list = []
        tool_node = _build(record, concurrency=False)
        state = _state([
            _tc("shell_read_output", {"session_id": "a"}, "t1"),
            _tc("shell_read_output", {"session_id": "b"}, "t2"),
        ])
        await tool_node(state, _config(AllowPE(), _ssm()))
        assert record == [("start", "a"), ("end", "a"), ("start", "b"), ("end", "b")]


class TestWireShapes:
    async def test_all_flags_on_pe_path_wire_shape(self):
        """§4.4 wire-shape #1：全 flag ON + PE 路径 → RUNNING 出现、CALLED 经
        queue 完成序。R6#3：configurable 投毒 enhancements_enabled=False——
        硬编码实现无视它；读 config 的实现在此翻车。"""
        record: list = []
        evt_a = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()
        tool_node = _build(record, running=True, evt_a=evt_a)
        state = _state([
            _tc("shell_read_output", {"session_id": "slow"}, "t1"),
            _tc("shell_read_output", {"session_id": "fast"}, "t2"),
        ])
        result = await tool_node(
            state,
            _config(AllowPE(), _ssm(), queue,
                    extra={"enhancements_enabled": False}),  # 毒饵（R11#5）
        )
        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        called = [e for e in drained
                  if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED]
        running_evts = [e for e in drained
                        if isinstance(e, ToolEvent) and e.status == ToolEventStatus.RUNNING]
        assert [e.tool_call_id for e in called] == ["t2", "t1"], "CALLED queue 完成序"
        assert {e.tool_call_id for e in running_evts} == {"t1", "t2"}
        state_called = [e for e in result.update["events"]
                        if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED]
        assert state_called == [], "queue 模式下 CALLED 不得再走 state-path"

    async def test_all_flags_on_legacy_fallback_wire_shape(self):
        """§4.4 wire-shape #2：全 flag ON + 混合批次 legacy 回退 → 零 RUNNING、
        CALLED state-path tc 序（enhancements_enabled=False override）。
        R6#3：configurable 投毒 enhancements_enabled=True——legacy 冻结面
        必须无视它。"""
        record: list = []
        queue: asyncio.Queue = asyncio.Queue()
        tool_node = _build(record, running=True)
        state = _state([
            _tc("totally_unknown_tool_xyz", {}, "t1"),   # 混合批次 → legacy
            _tc("shell_read_output", {"session_id": "a"}, "t2"),
        ])
        result = await tool_node(
            state,
            _config(AllowPE(), _ssm(), queue,
                    extra={"enhancements_enabled": True}),  # 毒饵（R11#5）
        )
        assert queue.empty(), "legacy 回退：零 RUNNING、零 queue-CALLED（与 flag 无关）"
        called = [e for e in result.update["events"]
                  if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED]
        # R4#2：unknown t1 的 AllowError 经 _translate_outcome（react_graph.py:668）
        # 同样产 CALLED 事件——legacy state-path 按 tc 序含两者。
        assert [e.tool_call_id for e in called] == ["t1", "t2"], \
            "legacy CALLED 走 state-path tc 序"

    async def test_asked_drain_queued_called_precede_confirmation(self):
        """C9-ON 序（spec §4.3.4/§4.5，R2#3 修）：并发 ON + Asked 批次——
        窗口内已完成工具的 queue-CALLED 必须全部先于 ToolConfirmationEvent
        入队（drain → 组装 → react_graph 发 conf 的构造性顺序）。"""
        from app.domain.models.event import ToolConfirmationEvent
        from app.domain.models.tool_result import Asked, DecisionReason

        class SeqPE:
            def __init__(self, outcomes):
                self.outcomes = list(outcomes)

            async def evaluate(self, call, ctx):
                kind = self.outcomes.pop(0)
                if kind == "allow":
                    return AllowSuccess(content="auto", data={})
                return Asked(
                    content="waiting",
                    reason=DecisionReason(
                        type="risk_enforce", code="medium", message="t"
                    ),
                )

            async def preflight_resume(self, *a, **kw):
                pass

            async def commit_resume(self, *a, **kw):
                pass

        record: list = []
        evt_a = asyncio.Event()
        queue: asyncio.Queue = asyncio.Queue()
        tool_node = _build(record, evt_a=evt_a)
        state = _state([
            _tc("shell_read_output", {"session_id": "slow"}, "t1"),
            _tc("shell_read_output", {"session_id": "fast"}, "t2"),
            _tc("file_write", {"path": "/a", "content": "x"}, "t3"),  # Asked
        ])
        result = await tool_node(
            state,
            _config(SeqPE(["allow", "allow", "ask"]), _ssm(), queue),
        )
        assert result.goto == "interrupt_helper"
        drained = []
        while not queue.empty():
            drained.append(queue.get_nowait())
        conf_idx = [i for i, e in enumerate(drained)
                    if isinstance(e, ToolConfirmationEvent)]
        called_idx = [i for i, e in enumerate(drained)
                      if isinstance(e, ToolEvent)
                      and e.status == ToolEventStatus.CALLED]
        assert len(conf_idx) == 1
        assert len(called_idx) == 2, "窗口两工具的 CALLED 都在 queue（并发 ON）"
        assert max(called_idx) < conf_idx[0], \
            "全部 window-CALLED 必须先于 ToolConfirmationEvent（drain-then-emit）"
        assert result.update["completed_tool_call_prefix"] == ["t2", "t1"], \
            "prefix 含窗口内已完成 ids（完成序：fast 先于 slow）"


class TestEnhancementsHardcodedStatic:
    def test_executor_constructors_pass_literal_enhancements_flag(self):
        """R7#1（R11#5 的静态面，补 wire-shape 毒饵的假阴性）：两个 dispatch
        调用点的 BatchToolExecutor 构造必须传**字面量** enhancements_enabled
        （_pe_dispatch=True / tool_node=False）——`configurable.get(...)` 之类
        的动态来源在此翻红。毒饵测试（上方两条）与本静态断言构成双防线。"""
        import ast
        import inspect

        from app.domain.services.graphs import react_graph

        tree = ast.parse(inspect.getsource(react_graph))
        expected = {"_pe_dispatch": True, "tool_node": False}
        found: dict[str, list] = {name: [] for name in expected}
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if fn.name not in expected:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and (
                    (isinstance(node.func, ast.Name)
                     and node.func.id == "BatchToolExecutor")
                    or (isinstance(node.func, ast.Attribute)
                        and node.func.attr == "BatchToolExecutor")
                ):
                    found[fn.name].append(node)
        for name, want in expected.items():
            calls = found[name]
            assert calls, f"{name} must construct BatchToolExecutor"
            for call in calls:
                kwargs = {k.arg: k.value for k in call.keywords}
                assert "enhancements_enabled" in kwargs, (
                    f"{name}: BatchToolExecutor missing enhancements_enabled kwarg"
                )
                value = kwargs["enhancements_enabled"]
                assert isinstance(value, ast.Constant) and value.value is want, (
                    f"{name}: enhancements_enabled must be the literal {want} "
                    "(R11#5 — never from config/state)"
                )

    def test_tool_runtime_config_defaults_all_off(self):
        """R3-FIX (P3): pin the zero-arg ToolRuntimeConfig() defaults directly.

        C4 dispatches through UNSAFE tools, so an accidental default-True flip on
        any B1 runtime flag (or a widened default max-concurrency) would silently
        change dark-launch behavior and could slip past the graph-shape tests
        that only exercise explicitly-constructed configs. This asserts the
        default surface stays OFF / conservative."""
        cfg = ToolRuntimeConfig()
        assert cfg.tool_running_events_enabled is False
        assert cfg.tool_concurrency_enabled is False
        assert cfg.llm_tool_call_streaming_enabled is False
        assert cfg.llm_incremental_calling_events_enabled is False
        assert cfg.tool_max_concurrency == 3


class TestModeFlipAndCancel:
    async def test_mode_flip_serial_recheck_denies_later_tool(self):
        """基础面：串行模式下 flip 后的后续工具在 thunk recheck 被拒。
        SSM 读序（单 tc = gate 1 读 + thunk 1 读）：_flip_ssm(3) → 前 3 读
        RUNNING、第 4 读起 TAKEOVER → t2 的 thunk recheck 拒绝。"""
        record: list = []
        tool_node = _build(record, concurrency=False)
        state = _state([
            _tc("shell_read_output", {"session_id": "a"}, "t1"),
            _tc("shell_read_output", {"session_id": "b"}, "t2"),
        ])
        result = await tool_node(state, _config(AllowPE(), _flip_ssm(3)))
        msgs = {m.tool_call_id: m for m in result.update["messages"]
                if isinstance(m, ToolMessage)}
        assert "out-a" in str(msgs["t1"].content), "flip 前的工具正常完成"
        assert "MODE_DENIED" in str(msgs["t2"].content), "flip 后未启动的工具被拒"
        assert ("start", "b") not in record, "被拒工具的 wrapper 不得执行"

    async def test_mode_flip_window_semantics(self):
        """spec §4.3 R4#1 具名测试（R3#4 修：并发 ON 窗口版）：已启动（flip 前
        过 recheck）的窗口工具完成；flip 后仍在排队的工具在 thunk recheck 被拒、
        wrapper 不执行。flip 由 t1 工具体内触发（确定性：已启动之后才翻转）；
        max_concurrency=1 确保 t2 严格排在 t1 之后。"""
        from langchain_core.tools import tool as lc_tool

        from app.domain.services.graphs.react_graph import build_react_graph

        holder = {"mode": SessionStatus.RUNNING}
        rev = {"n": 0}

        async def read_mode(_sid):
            rev["n"] += 1
            return (holder["mode"], rev["n"])

        ssm = AsyncMock()
        ssm.get_mode_with_revision = AsyncMock(side_effect=read_mode)

        record: list = []

        @lc_tool
        async def shell_read_output(session_id: str = "") -> str:
            """Read shell output."""
            record.append(("start", session_id))
            if session_id == "flipper":
                holder["mode"] = SessionStatus.TAKEOVER  # 已启动后才 flip
            record.append(("end", session_id))
            return f"out-{session_id}"

        stub_llm = AsyncMock()
        stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)
        graph = build_react_graph(
            stub_llm, [shell_read_output],
            tool_runtime_config=ToolRuntimeConfig(
                tool_concurrency_enabled=True, tool_max_concurrency=1,
            ),
        )
        tool_node = graph.nodes["tool_node"].bound.afunc
        state = _state([
            _tc("shell_read_output", {"session_id": "flipper"}, "t1"),
            _tc("shell_read_output", {"session_id": "later"}, "t2"),
        ])
        result = await tool_node(state, _config(AllowPE(), ssm))
        msgs = {m.tool_call_id: m for m in result.update["messages"]
                if isinstance(m, ToolMessage)}
        assert "out-flipper" in str(msgs["t1"].content), \
            "已启动的窗口工具不因 mid-flight flip 中止（执行起点锚定）"
        assert "MODE_DENIED" in str(msgs["t2"].content), \
            "flip 后未启动（排队）的工具在 thunk recheck 被拒"
        assert ("start", "later") not in record

    async def test_cancel_after_semaphore_via_graph(self):
        """cancel_event 在首工具执行中置位 → 排队工具过 semaphore 后被
        checkpoint 拦截 → CancelledByEventError("tool_window_entry") 逃逸。"""
        from app.domain.services.graphs.react_graph import CancelledByEventError

        record: list = []
        cancel_event = asyncio.Event()
        evt_a = asyncio.Event()
        evt_a.set()

        from langchain_core.tools import tool as lc_tool

        from app.domain.services.graphs.react_graph import build_react_graph

        @lc_tool
        async def shell_read_output(session_id: str = "") -> str:
            """Read shell output."""
            record.append(("start", session_id))
            cancel_event.set()  # 首个执行者触发取消
            record.append(("end", session_id))
            return "out"

        stub_llm = AsyncMock()
        stub_llm.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
        stub_llm.bind_tools = MagicMock(return_value=stub_llm)
        graph = build_react_graph(
            stub_llm, [shell_read_output],
            tool_runtime_config=ToolRuntimeConfig(
                tool_concurrency_enabled=True, tool_max_concurrency=1,
            ),
        )
        tool_node = graph.nodes["tool_node"].bound.afunc
        state = _state([
            _tc("shell_read_output", {"session_id": "a"}, "t1"),
            _tc("shell_read_output", {"session_id": "b"}, "t2"),
        ])
        with pytest.raises(CancelledByEventError) as exc_info:
            await tool_node(
                state,
                _config(AllowPE(), _ssm(), extra={"cancel_event": cancel_event}),
            )
        assert "tool_window_entry" in str(exc_info.value)
        assert ("start", "b") not in record
