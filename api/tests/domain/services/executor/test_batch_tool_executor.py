"""B1-1a BatchToolExecutor serial-mode unit tests (pure, no react_graph)."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from app.domain.services.executor import (
    Ask,
    AskPayload,
    BatchToolExecutor,
    Execute,
    FinalizeMeta,
    PerTcResult,
    Skip,
    Surfaced,
    TcMeta,
    Waiting,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _tc(cid: str, name: str = "fake_tool") -> dict:
    return {"id": cid, "name": name, "args": {}, "type": "tool_call"}


def _res(
    cid: str,
    *,
    content: str = "ok",
    name: str = "fake_tool",
    failure: bool = False,
    deferred: list | None = None,
    events: list | None = None,
    meta: FinalizeMeta | None = None,
    consumed: str | None = None,
    completed: list[str] | None = None,
) -> PerTcResult:
    return PerTcResult(
        tool_message=ToolMessage(content=content, tool_call_id=cid, name=name),
        deferred_human=list(deferred or []),
        events=list(events or []),
        completed_ids=list(completed if completed is not None else [cid]),
        is_failure=failure,
        consumed_resume_id=consumed,
        finalize_meta=meta,
    )


def _meta(cid: str, *, failure: bool = False) -> FinalizeMeta:
    return FinalizeMeta(
        tool_name_raw=f"tool-{cid}",
        args={"k": cid},
        is_failure=failure,
        started_at=1.0,
        ended_at=2.0,
    )


def _execute(cid: str, record: list, result: PerTcResult) -> Execute:
    async def thunk() -> PerTcResult:
        record.append(("thunk", cid))
        return result

    return Execute(
        thunk=thunk,
        tc_meta=TcMeta(
            tool_call_id=cid,
            raw_function_name="fake_tool",
            category_tool_name="fake",
            args={},
        ),
    )


def _ask(cid: str) -> Ask:
    return Ask(
        payload=AskPayload(
            pending_ask_outcome={"variant": "asked"},
            pending_ask_tool_call_id=cid,
            pending_ask_artifact={"tool_call_id": cid},
            pending_ask_tool_args={"x": 1},
            confirmation_event=object(),
        )
    )


def _gate(script: dict):
    calls: list[str] = []

    async def gate(tc: dict):
        calls.append(tc["id"])
        return script[tc["id"]]

    gate.calls = calls  # type: ignore[attr-defined]
    return gate


def _executor(**kw) -> BatchToolExecutor:
    kw.setdefault("enhancements_enabled", True)
    return BatchToolExecutor(**kw)


async def _run(ex, tcs, gate, **kw):
    kw.setdefault("has_prior_soft_hint", False)
    kw.setdefault("attempt_count", 0)
    kw.setdefault("failure_count", 0)
    kw.setdefault("already_done", [])
    return await ex.run(tcs, gate, **kw)


class TestSerialScheduling:
    async def test_execute_thunks_serial_in_tc_order_and_flush(self):
        record: list = []
        deferred = [HumanMessage(content="deferred-multimodal")]
        gate = _gate({
            "t1": _execute("t1", record, _res("t1", deferred=deferred)),
            "t2": _execute("t2", record, _res("t2")),
        })
        result = await _run(_executor(), [_tc("t1"), _tc("t2")], gate)
        assert record == [("thunk", "t1"), ("thunk", "t2")]
        msgs = result.update["messages"]
        assert [m.tool_call_id for m in msgs if isinstance(m, ToolMessage)] == ["t1", "t2"]
        # deferred HumanMessage 殿后
        assert isinstance(msgs[-1], HumanMessage)
        assert result.interrupted is False

    async def test_skip_is_zero_accounting(self):
        gate = _gate({"t1": Skip(tool_call_id="t1"), "t2": _execute("t2", [], _res("t2"))})
        result = await _run(_executor(), [_tc("t1"), _tc("t2")], gate)
        assert result.update["completed_tool_call_prefix"] == []  # clean tail resets anyway
        msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in msgs] == ["t2"]
        assert result.update["failure_count"] == 0

    async def test_surfaced_accounts_failure(self):
        gate = _gate({"t1": Surfaced(result=_res("t1", failure=True, content="denied"))})
        result = await _run(_executor(), [_tc("t1")], gate, failure_count=2)
        assert result.update["failure_count"] == 3
        assert result.update["attempt_count"] == 1

    async def test_waiting_sets_should_interrupt_and_continues(self):
        record: list = []
        gate = _gate({
            "t1": Waiting(result=_res("t1", content="WAITING_FOR_USER", name="message_ask_user")),
            "t2": _execute("t2", record, _res("t2")),
        })
        result = await _run(_executor(), [_tc("t1"), _tc("t2")], gate)
        assert record == [("thunk", "t2")], "Waiting must NOT stop the batch"
        assert result.should_interrupt is True
        assert result.update["should_interrupt"] is True
        assert result.interrupted is False


class TestAskedEarlyExit:
    async def test_ask_stops_gating_and_update_keys_exact(self):
        record: list = []
        gate = _gate({
            "t1": _execute("t1", record, _res("t1")),
            "t2": _ask("t2"),
            "t3": _execute("t3", record, _res("t3")),
        })
        result = await _run(
            _executor(), [_tc("t1"), _tc("t2"), _tc("t3")], gate,
            attempt_count=4, failure_count=1, already_done=["old1"],
        )
        assert gate.calls == ["t1", "t2"], "tcs after Ask must not be gated"
        assert record == [("thunk", "t1")]
        assert result.interrupted is True
        assert result.ask_payload is not None
        assert set(result.update.keys()) == {
            "messages", "events", "attempt_count", "failure_count",
            "completed_tool_call_prefix", "pending_ask_outcome",
            "pending_ask_tool_call_id", "pending_ask_artifact",
            "pending_ask_tool_args",
        }
        assert result.update["completed_tool_call_prefix"] == ["old1", "t1"]
        assert result.update["attempt_count"] == 5
        assert result.update["failure_count"] == 1
        assert result.update["pending_ask_tool_call_id"] == "t2"

    async def test_ask_preserves_duplicate_completed_ids_quirk(self):
        # eligibility fail-closed 分支今日双 append 同一 id — list 保留该 quirk
        gate = _gate({
            "t1": Surfaced(result=_res("t1", failure=True, completed=["t1", "t1"])),
            "t2": _ask("t2"),
        })
        result = await _run(_executor(), [_tc("t1"), _tc("t2")], gate)
        assert result.update["completed_tool_call_prefix"] == ["t1", "t1"]


class TestCleanTail:
    async def test_clean_batch_update_resets(self):
        gate = _gate({"t1": _execute("t1", [], _res("t1"))})
        result = await _run(_executor(), [_tc("t1")], gate)
        assert result.update["completed_tool_call_prefix"] == []
        assert result.update["approved_tool_call_ids"] == []
        for key in (
            "pending_ask_outcome", "pending_ask_tool_call_id",
            "pending_ask_artifact", "pending_ask_tool_args",
        ):
            assert result.update[key] is None
        assert "should_interrupt" not in result.update
        assert "soft_hint_sent" not in result.update

    async def test_soft_hint_written_only_at_clean_tail(self):
        soft = Surfaced(result=_res("t1", content="SOFT_HINT", name="message_ask_user"))
        gate = _gate({"t1": soft})
        result = await _run(_executor(), [_tc("t1")], gate, has_prior_soft_hint=False)
        assert result.update["soft_hint_sent"] is True

        # has_prior_soft_hint=True → 不写
        gate2 = _gate({"t1": Surfaced(result=_res("t1", content="SOFT_HINT", name="message_ask_user"))})
        result2 = await _run(_executor(), [_tc("t1")], gate2, has_prior_soft_hint=True)
        assert "soft_hint_sent" not in result2.update

        # Asked 早退 → 不写（不对称保持）
        gate3 = _gate({
            "t1": Surfaced(result=_res("t1", content="SOFT_HINT", name="message_ask_user")),
            "t2": _ask("t2"),
        })
        result3 = await _run(_executor(), [_tc("t1"), _tc("t2")], gate3)
        assert "soft_hint_sent" not in result3.update

    async def test_resume_cleanup_merged_at_both_exits(self):
        cleanup = {"pe_resume_outcomes": {"kept": {"variant": "allow_success"}}}
        gate = _gate({"t1": _execute("t1", [], _res("t1", consumed="t1"))})
        result = await _run(
            _executor(), [_tc("t1")], gate, resume_cleanup=lambda: dict(cleanup)
        )
        assert result.update["pe_resume_outcomes"] == cleanup["pe_resume_outcomes"]

        gate2 = _gate({"t1": _ask("t1")})
        result2 = await _run(
            _executor(), [_tc("t1")], gate2, resume_cleanup=lambda: dict(cleanup)
        )
        assert result2.update["pe_resume_outcomes"] == cleanup["pe_resume_outcomes"]


class TestRecordingAndTransparency:
    async def test_record_finalize_called_in_tc_order_with_meta(self):
        recorded: list[str] = []
        ex = _executor(record_finalize=lambda m: recorded.append(m.tool_name_raw))
        gate = _gate({
            "t1": _execute("t1", [], _res("t1", meta=_meta("t1"))),
            "t2": Surfaced(result=_res("t2", failure=True, meta=_meta("t2", failure=True))),
            "t3": Waiting(result=_res("t3", content="WAITING_FOR_USER", name="message_ask_user")),  # meta=None → 不记录
        })
        await _run(ex, [_tc("t1"), _tc("t2"), _tc("t3")], gate)
        assert recorded == ["tool-t1", "tool-t2"]

    async def test_gate_exception_propagates_unwrapped(self):
        class Boom(RuntimeError):
            pass

        async def gate(tc):
            raise Boom("gate blew up")

        with pytest.raises(Boom):
            await _run(_executor(), [_tc("t1")], gate)

    async def test_thunk_exception_propagates_unwrapped(self):
        class Boom(RuntimeError):
            pass

        async def thunk():
            raise Boom("thunk blew up")

        gate = _gate({
            "t1": Execute(
                thunk=thunk,
                tc_meta=TcMeta(
                    tool_call_id="t1", raw_function_name="fake_tool",
                    category_tool_name="fake", args={},
                ),
            )
        })
        with pytest.raises(Boom):
            await _run(_executor(), [_tc("t1")], gate)

    async def test_cancelled_by_event_error_propagates_original_type(self):
        """B1-1a 异常穿透三件套之一（spec §4.1 R5#1，R2#2 补）：真实逃逸类型
        CancelledByEventError 原类型穿透 executor（graph unwind 合同 :101）。"""
        from app.domain.services.graphs.react_graph import CancelledByEventError

        async def gate(tc):
            raise CancelledByEventError("tool_node_entry")

        with pytest.raises(CancelledByEventError):
            await _run(_executor(), [_tc("t1")], gate)

    async def test_child_scope_violation_propagates_original_type(self):
        """三件套之二：ChildScopeViolation（gate 内 PE broad-except 之前显式
        re-raise :1863）原类型穿透——runner 依赖类型分支（:4640）。"""
        from app.domain.services.permission.child_scope_gate import ScopeDecision
        from app.domain.services.permission.child_scope_violation import (
            ChildScopeViolation,
        )

        async def gate(tc):
            # R5#1：真实构造签名 = (decision, *, tool_name, target_path=None)
            raise ChildScopeViolation(
                ScopeDecision.OUT_OF_PATH_LEASE,
                tool_name="fake_tool",
                target_path="/outside",
            )

        with pytest.raises(ChildScopeViolation):
            await _run(_executor(), [_tc("t1")], gate)

    async def test_empty_batch_clean_tail(self):
        async def gate(tc):  # pragma: no cover — never called
            raise AssertionError

        result = await _run(_executor(), [], gate, attempt_count=7)
        assert result.update["attempt_count"] == 8
        assert result.interrupted is False


# ============================================================================
# Task 8 (B1-1c) — concurrency window: semaphore / drain / cancel lanes /
# tc-order record / CALLED emitter.
# ============================================================================
import asyncio

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.domain.services.executor import CONCURRENCY_SAFE_TOOLS


def _called_event(cid: str) -> ToolEvent:
    return ToolEvent(
        tool_call_id=cid, tool_name="shell", function_name="shell_read_output",
        function_args={}, status=ToolEventStatus.CALLED,
    )


def _timed_execute(
    cid: str,
    record: list,
    *,
    name: str = "shell_read_output",
    wait_on: asyncio.Event | None = None,
    set_on_start: asyncio.Event | None = None,
    set_on_done: asyncio.Event | None = None,
    raise_exc: BaseException | None = None,
    meta: FinalizeMeta | None = None,
) -> Execute:
    async def thunk() -> PerTcResult:
        record.append(("start", cid))
        if set_on_start is not None:
            set_on_start.set()
        if wait_on is not None:
            await wait_on.wait()
        if raise_exc is not None:
            raise raise_exc
        record.append(("end", cid))
        if set_on_done is not None:
            set_on_done.set()
        return _res(cid, events=[_called_event(cid)], meta=meta)

    return Execute(
        thunk=thunk,
        tc_meta=TcMeta(
            tool_call_id=cid, raw_function_name=name,
            category_tool_name="shell", args={},
        ),
    )


def _win_executor(emitted: list | None = None, recorded: list | None = None, **kw):
    async def emit(evt):
        if emitted is not None:
            emitted.append(evt)

    kw.setdefault("enhancements_enabled", True)
    kw.setdefault("concurrency_enabled", True)
    kw.setdefault("max_concurrency", 3)
    if emitted is not None:
        kw.setdefault("emit_queue_event", emit)
    if recorded is not None:
        kw.setdefault(
            "record_finalize", lambda m: recorded.append(m.tool_name_raw)
        )
    return BatchToolExecutor(**kw)


class TestSafeSetChangeControl:
    def test_concurrency_safe_tools_pinned_exact(self):
        """R6#4：安全集变更 = 安全决策（spec R4#2/§8 扩容前置条件）——
        drive-by 扩容必须在此翻红并触发 codex 审查流程。"""
        assert CONCURRENCY_SAFE_TOOLS == frozenset({"shell_read_output"})


class TestConcurrencyWindow:
    async def test_safe_tools_overlap_in_window(self):
        record: list = []
        evt = asyncio.Event()
        gate = _gate({
            "t1": _timed_execute("t1", record, wait_on=evt),
            "t2": _timed_execute("t2", record, set_on_done=evt),
        })
        result = await _run(_win_executor(), [_tc("t1"), _tc("t2")], gate)
        # t2 在 t1 结束前开始 = 真并发
        assert record.index(("start", "t2")) < record.index(("end", "t1"))
        msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in msgs] == ["t1", "t2"]  # flush 原始 tc 序
        # completed_ids 完成序（t2 先完成）
        # （Asked 早退才暴露 prefix；此处经 emitted 序断言完成序——见下测）

    async def test_unsafe_tool_drains_window_first(self):
        record: list = []
        evt = asyncio.Event()

        script = {
            "t1": _timed_execute("t1", record, wait_on=evt),
            "t2": _timed_execute("t2", record, name="file_write"),  # 不在安全集 → 互斥
        }
        calls: list[str] = []

        async def gate(tc):
            calls.append(tc["id"])
            if tc["id"] == "t2":
                evt.set()  # 释放窗口，drain 得以完成
            return script[tc["id"]]

        await _run(_win_executor(), [_tc("t1"), _tc("t2", "file_write")], gate)
        assert record.index(("end", "t1")) < record.index(("start", "t2")), \
            "互斥工具必须等窗口 drain 后单独执行"

    async def test_gate_overlaps_window_execution(self):
        record: list = []
        evt = asyncio.Event()
        started = asyncio.Event()

        script = {
            "t1": _timed_execute("t1", record, set_on_start=started, wait_on=evt),
            "t2": _timed_execute("t2", record, set_on_done=evt),
        }

        async def gate(tc):
            if tc["id"] == "t2":
                await started.wait()  # t1 已在执行中 —— gate 与窗口重叠
                record.append(("gate", "t2"))
            return script[tc["id"]]

        await _run(_win_executor(), [_tc("t1"), _tc("t2")], gate)
        assert record.index(("gate", "t2")) > record.index(("start", "t1"))
        assert record.index(("gate", "t2")) < record.index(("end", "t1"))

    async def test_asked_drains_window_prefix_includes_completed(self):
        record: list = []
        emitted: list = []
        evt = asyncio.Event()

        script = {
            "t1": _timed_execute("t1", record, wait_on=evt),
            "t2": _ask("t2"),
        }

        async def gate(tc):
            if tc["id"] == "t2":
                evt.set()
            return script[tc["id"]]

        result = await _run(
            _win_executor(emitted), [_tc("t1"), _tc("t2")], gate,
            already_done=["old"],
        )
        assert result.interrupted is True
        assert result.update["completed_tool_call_prefix"] == ["old", "t1"]
        # 窗口 CALLED 在 run() 返回前已经 queue 发射（先于 react_graph 发 conf）
        assert [e.tool_call_id for e in emitted] == ["t1"]

    async def test_completion_order_ids_but_original_order_messages(self):
        record: list = []
        emitted: list = []
        evt = asyncio.Event()
        gate = _gate({
            "t1": _timed_execute("t1", record, wait_on=evt),   # 慢
            "t2": _timed_execute("t2", record, set_on_done=evt),  # 快
            "t3": _ask("t3"),  # 用 Asked 暴露 prefix 完成序
        })
        result = await _run(_win_executor(emitted), [_tc("t1"), _tc("t2"), _tc("t3")], gate)
        assert result.update["completed_tool_call_prefix"] == ["t2", "t1"], \
            "completed_ids 按完成序单点追加"
        assert [e.tool_call_id for e in emitted] == ["t2", "t1"], \
            "CALLED queue 发射按完成序"
        msgs = [m for m in result.update["messages"] if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in msgs] == ["t1", "t2"], "ToolMessage 原始 tc 序"

    async def test_semaphore_caps_concurrency(self):
        peak = {"now": 0, "max": 0}

        def _tracked(cid):
            async def thunk():
                peak["now"] += 1
                peak["max"] = max(peak["max"], peak["now"])
                await asyncio.sleep(0)
                peak["now"] -= 1
                return _res(cid, events=[_called_event(cid)])
            return Execute(thunk=thunk, tc_meta=TcMeta(
                tool_call_id=cid, raw_function_name="shell_read_output",
                category_tool_name="shell", args={},
            ))

        gate = _gate({f"t{i}": _tracked(f"t{i}") for i in range(1, 6)})
        ex = _win_executor(max_concurrency=2)
        await _run(ex, [_tc(f"t{i}") for i in range(1, 6)], gate)
        assert peak["max"] <= 2

    async def test_flag_off_strict_serial(self):
        record: list = []
        evt = asyncio.Event()
        evt.set()  # 串行下无人释放——直接放行
        gate = _gate({
            "t1": _timed_execute("t1", record, wait_on=evt),
            "t2": _timed_execute("t2", record),
        })
        ex = BatchToolExecutor(enhancements_enabled=True, concurrency_enabled=False)
        await _run(ex, [_tc("t1"), _tc("t2")], gate)
        assert record == [("start", "t1"), ("end", "t1"), ("start", "t2"), ("end", "t2")]

    async def test_enhancements_false_forces_serial_state_path(self):
        emitted: list = []
        record: list = []
        gate = _gate({"t1": _timed_execute("t1", record)})
        ex = _win_executor(emitted, enhancements_enabled=False)
        result = await _run(ex, [_tc("t1")], gate)
        assert emitted == [], "legacy（enhancements=False）永远 state-path CALLED"
        called = [e for e in result.update["events"]
                  if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED]
        assert [e.tool_call_id for e in called] == ["t1"]

    async def test_queue_missing_fallback_state_tc_order(self):
        record: list = []
        evt = asyncio.Event()
        gate = _gate({
            "t1": _timed_execute("t1", record, wait_on=evt),
            "t2": _timed_execute("t2", record, set_on_done=evt),
        })
        ex = _win_executor(emitted=None)  # emit_queue_event=None → per-attempt 回退
        result = await _run(ex, [_tc("t1"), _tc("t2")], gate)
        called = [e for e in result.update["events"]
                  if isinstance(e, ToolEvent) and e.status == ToolEventStatus.CALLED]
        assert [e.tool_call_id for e in called] == ["t1", "t2"], \
            "queue 缺失回退 state-path 且按原始 tc 序（C5 形状）"


class TestCancelAndExceptionLanes:
    async def test_cancel_after_semaphore_before_thunk(self):
        """排队任务过 semaphore 后被 cancel 检查拦截：thunk 未执行、零发射。"""
        class Cancelled(Exception):
            pass

        flag = {"set": False}

        def check_cancel():
            if flag["set"]:
                raise Cancelled("checkpoint")

        record: list = []
        emitted: list = []

        def _setting_execute(cid):
            async def thunk():
                record.append(("start", cid))
                flag["set"] = True
                return _res(cid, events=[_called_event(cid)])
            return Execute(thunk=thunk, tc_meta=TcMeta(
                tool_call_id=cid, raw_function_name="shell_read_output",
                category_tool_name="shell", args={},
            ))

        gate = _gate({
            "t1": _setting_execute("t1"),
            "t2": _timed_execute("t2", record),
        })
        ex = _win_executor(emitted, max_concurrency=1, check_cancel=check_cancel)
        with pytest.raises(Cancelled):
            await _run(ex, [_tc("t1"), _tc("t2")], gate)
        assert ("start", "t2") not in record, "被取消任务的 thunk 不得执行"
        assert [e.tool_call_id for e in emitted] == ["t1"], \
            "cancelled-before-start 任务零 completed/零发射（R14-E7）"

    async def test_self_raised_cancellederror_fails_loud(self):
        gate = _gate({
            "t1": _timed_execute("t1", [], raise_exc=asyncio.CancelledError()),
        })
        with pytest.raises(asyncio.CancelledError):
            await _run(_win_executor(), [_tc("t1")], gate)

    async def test_window_task_failure_does_not_hang_on_stuck_sibling(self):
        """R3#3：一个窗口任务 raise、另一个永久阻塞——drain 必须以
        FIRST_EXCEPTION 语义取消 stuck sibling 并原类型重抛，绝不挂起。
        （生产语境：普通工具失败是 AllowError 不 raise（Non-goal 6 不级联）；
        raise = 逃逸类型（cancel/bug），spec §4.1.1 要求「首个优先、其余取消」。）"""
        class Boom(RuntimeError):
            pass

        never = asyncio.Event()
        record: list = []
        gate = _gate({
            "t1": _timed_execute("t1", record, wait_on=never),      # 永久阻塞
            "t2": _timed_execute("t2", record, raise_exc=Boom()),   # 逃逸异常
        })
        with pytest.raises(Boom):
            await asyncio.wait_for(
                _run(_win_executor(), [_tc("t1"), _tc("t2")], gate),
                timeout=5,
            )

    async def test_sibling_cancel_suppressed_when_primary_chosen(self):
        """逃逸异常路径：executor 自己 cancel 的兄弟任务 CancelledError 被抑制，
        重抛的是 primary（Boom），不是 CancelledError。"""
        class Boom(RuntimeError):
            pass

        never = asyncio.Event()
        record: list = []
        script = {"t1": _timed_execute("t1", record, wait_on=never)}  # 永久阻塞（可被 cancel）

        async def gate(tc):
            if tc["id"] == "t1":
                return script["t1"]
            raise Boom("gate escape")  # t1 仍在窗口时 primary 从 gate 逃逸

        with pytest.raises(Boom):
            await _run(_win_executor(), [_tc("t1"), _tc("t2")], gate)

    async def test_tracker_drain_order_same_signature(self):
        """R15#2 杀伤测试：同签名成败乱序完成，record_finalize 仍按原始 tc 序。"""
        recorded: list = []
        evt = asyncio.Event()
        gate = _gate({
            "t1": _timed_execute("t1", [], wait_on=evt, meta=_meta("t1")),      # 慢
            "t2": _timed_execute("t2", [], set_on_done=evt, meta=_meta("t2")),  # 快
        })
        await _run(_win_executor(recorded=recorded), [_tc("t1"), _tc("t2")], gate)
        assert recorded == ["tool-t1", "tool-t2"], \
            "窗口任务的 tracker/metrics 记录必须在 drain 点按原始 tc 序"
