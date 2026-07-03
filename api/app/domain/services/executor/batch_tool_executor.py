"""B1-1a batch tool executor — scheduling + bookkeeping ONLY (spec §4.1/§4.1.1).

Security decisions (replay-skip → eligibility → N1 → PE evaluation) live in
react_graph's gate closures; execution encapsulation (live-mode recheck →
RUNNING emission → wrapper invocation) lives in react_graph's thunk factories.

Hard constraints (INV-5 v2 rule 3 asserts these statically — including
STRING-level asserts, so the literal names of the wrapper helper, the PE
evaluate call, and the two thunk factories must never appear anywhere in
this file, not even in comments):
- no graph-framework import (BatchResult is neutral; Command wrapping stays in react_graph)
- no references to react_graph's execution/security symbols (see above)
- no catch-all around gate/thunk — exceptions propagate with original types
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from langchain_core.messages import ToolMessage  # noqa: F401 — soft-hint scan uses duck attrs

from app.domain.models.event import ToolEvent, ToolEventStatus

# B1-1c fail-closed 并发安全集（spec §4.3 R4#2）：键 = 原始函数名。
# 扩容前置条件见 spec §8（file_read 等待 sandbox read 侧 special-file 拒绝）。
# 变更本集合 = 安全决策（codex review required）。
CONCURRENCY_SAFE_TOOLS: frozenset[str] = frozenset({"shell_read_output"})


@dataclass(frozen=True)
class TcMeta:
    tool_call_id: str
    raw_function_name: str
    category_tool_name: str
    args: dict


@dataclass(frozen=True)
class FinalizeMeta:
    tool_name_raw: str
    args: dict
    is_failure: bool
    started_at: float
    ended_at: float


@dataclass
class PerTcResult:
    tool_message: Any | None = None
    deferred_human: list = field(default_factory=list)
    events: list = field(default_factory=list)
    completed_ids: list = field(default_factory=list)
    is_failure: bool = False
    consumed_resume_id: str | None = None
    finalize_meta: FinalizeMeta | None = None


@dataclass(frozen=True)
class AskPayload:
    pending_ask_outcome: Any
    pending_ask_tool_call_id: str
    pending_ask_artifact: Any
    pending_ask_tool_args: dict | None
    confirmation_event: Any


@dataclass(frozen=True)
class Skip:
    tool_call_id: str


@dataclass(frozen=True)
class Execute:
    thunk: Callable[[], Awaitable[PerTcResult]]
    tc_meta: TcMeta


@dataclass(frozen=True)
class Surfaced:
    result: PerTcResult


@dataclass(frozen=True)
class Ask:
    payload: AskPayload


@dataclass(frozen=True)
class Waiting:
    result: PerTcResult


GateOutcome = Skip | Execute | Surfaced | Ask | Waiting


@dataclass
class BatchResult:
    update: dict
    interrupted: bool
    should_interrupt: bool
    ask_payload: AskPayload | None


@dataclass
class _BatchAccumulator:
    new_completed_ids: list = field(default_factory=list)
    new_failures: int = 0
    should_interrupt: bool = False
    soft_hint_sent_this_batch: bool = False
    consumed_resume_ids: list = field(default_factory=list)


class BatchToolExecutor:
    """Two-phase gate→thunk batch driver. Serial in B1-1a; window mode in B1-1c."""

    def __init__(
        self,
        *,
        enhancements_enabled: bool,
        concurrency_enabled: bool = False,
        max_concurrency: int = 3,
        running_events_enabled: bool = False,
        emit_queue_event: Callable[[Any], Awaitable[None]] | None = None,
        check_cancel: Callable[[], None] | None = None,
        record_finalize: Callable[[FinalizeMeta], None] | None = None,
    ) -> None:
        self._enhancements_enabled = enhancements_enabled
        self._concurrency_enabled = concurrency_enabled
        self._max_concurrency = max_concurrency
        self._running_events_enabled = running_events_enabled
        self._emit_queue_event = emit_queue_event
        self._check_cancel = check_cancel
        self._record_finalize = record_finalize

    async def run(
        self,
        tool_calls: list,
        gate: Callable[[dict], Awaitable[GateOutcome]],
        *,
        has_prior_soft_hint: bool,
        attempt_count: int,
        failure_count: int,
        already_done: list,
        resume_cleanup: Callable[[], dict] | None = None,
    ) -> BatchResult:
        acc = _BatchAccumulator()
        slots: dict[int, PerTcResult] = {}          # idx → result（flush 排序用）
        window: list[tuple[int, TcMeta, "asyncio.Task"]] = []
        pending_finalize: list[tuple[int, FinalizeMeta]] = []
        window_active = self._concurrency_enabled and self._enhancements_enabled
        queue_called = window_active and self._emit_queue_event is not None
        sem = asyncio.Semaphore(self._max_concurrency)

        async def _emit_called_if_queued(result: PerTcResult) -> None:
            """完成点动作（queue 模式）：CALLED 实时入队并从 state events 摘除。"""
            if not queue_called:
                return
            kept = []
            for evt in result.events:
                if isinstance(evt, ToolEvent) and evt.status == ToolEventStatus.CALLED:
                    await self._emit_queue_event(evt)
                else:
                    kept.append(evt)
            result.events = kept

        async def _on_completion(idx: int, result: PerTcResult, *, defer_finalize: bool) -> None:
            """完成点单点落账（事件循环内，无锁）。completed_ids 按完成序。"""
            slots[idx] = result
            acc.new_completed_ids.extend(result.completed_ids)
            if result.is_failure:
                acc.new_failures += 1
            if result.consumed_resume_id is not None:
                acc.consumed_resume_ids.append(result.consumed_resume_id)
            await _emit_called_if_queued(result)
            if result.finalize_meta is not None:
                if defer_finalize:
                    pending_finalize.append((idx, result.finalize_meta))
                elif self._record_finalize is not None:
                    self._record_finalize(result.finalize_meta)

        async def _windowed(idx: int, thunk) -> None:
            async with sem:
                if self._check_cancel is not None:
                    self._check_cancel()   # cancel 点：semaphore 后、RUNNING/wrapper 前
                result = await thunk()
                await _on_completion(idx, result, defer_finalize=True)

        async def _drain() -> None:
            """窗口收敛（R3#3 修：FIRST_EXCEPTION 语义，spec §4.1.1「首个优先、
            其余任务取消」——gather 等全员会在 raise+stuck-sibling 场景挂起）。
            自 raise CancelledError 的任务呈现为 cancelled-task：正常 drain 从
            不主动 cancel，出现即 fail-loud（歧义默认，R12#2）。"""
            if not window:
                return
            tasks = [t for _, _, t in window]
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_EXCEPTION
            )
            primary: BaseException | None = None
            for _, _, task in window:                    # 窗口序选首个异常
                if task not in done:
                    continue
                if task.cancelled():
                    # 正常 drain 从未 cancel 任何任务 → 这是任务自 raise 的
                    # CancelledError（lane ii）→ fail-loud 原类型重抛
                    primary = asyncio.CancelledError()
                    break
                if task.exception() is not None:
                    primary = task.exception()
                    break
            if primary is not None:
                for p in pending:
                    p.cancel()                           # 其余任务取消（primary 已选定 → lane i 抑制）
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
                window.clear()
                pending_finalize.clear()
                raise primary
            window.clear()
            # drain 点：record_finalize 按原始 tc 序（R11#4）
            if self._record_finalize is not None:
                for _idx, meta in sorted(pending_finalize, key=lambda p: p[0]):
                    self._record_finalize(meta)
            pending_finalize.clear()

        try:
            for idx, tc in enumerate(tool_calls):
                outcome = await gate(tc)
                if isinstance(outcome, Skip):
                    continue
                if isinstance(outcome, Waiting):
                    acc.should_interrupt = True
                    await _on_completion(idx, outcome.result, defer_finalize=False)
                    continue
                if isinstance(outcome, Surfaced):
                    await _on_completion(idx, outcome.result, defer_finalize=False)
                    continue
                if isinstance(outcome, Execute):
                    safe = (
                        window_active
                        and outcome.tc_meta.raw_function_name in CONCURRENCY_SAFE_TOOLS
                    )
                    if safe:
                        window.append((
                            idx,
                            outcome.tc_meta,
                            asyncio.create_task(_windowed(idx, outcome.thunk)),
                        ))
                        continue
                    await _drain()                       # 互斥：先清窗
                    result = await outcome.thunk()       # inline = 今日串行等价
                    await _on_completion(idx, result, defer_finalize=False)
                    continue
                if isinstance(outcome, Ask):
                    await _drain()
                    return self._asked_result(
                        acc, slots, outcome.payload,
                        attempt_count=attempt_count,
                        failure_count=failure_count,
                        already_done=already_done,
                        resume_cleanup=resume_cleanup,
                    )
                raise TypeError(f"unknown GateOutcome: {type(outcome).__name__}")
            await _drain()
        except BaseException:
            # 逃逸车道（R11#2）：primary 已选定 → cancel 全窗口并抑制其
            # CancelledError（executor 自己发起的取消 = 正常清扫），原异常重抛。
            for _, _, task in window:
                task.cancel()
            if window:
                await asyncio.gather(
                    *(t for _, _, t in window), return_exceptions=True
                )
                window.clear()
            raise
        return self._clean_result(
            acc, slots,
            has_prior_soft_hint=has_prior_soft_hint,
            attempt_count=attempt_count,
            failure_count=failure_count,
            resume_cleanup=resume_cleanup,
        )

    # ---- flush / result assembly (indexed-slot order in B1-1c) ----

    @staticmethod
    def _flush(slots: dict[int, "PerTcResult"]) -> tuple[list, list, list]:
        msgs, deferred, events = [], [], []
        for idx in sorted(slots):
            r = slots[idx]
            if r.tool_message is not None:
                msgs.append(r.tool_message)
            deferred.extend(r.deferred_human)
            events.extend(r.events)
        return msgs, deferred, events

    def _asked_result(
        self,
        acc: _BatchAccumulator,
        slots: dict[int, PerTcResult],
        payload: AskPayload,
        *,
        attempt_count: int,
        failure_count: int,
        already_done: list,
        resume_cleanup: Callable[[], dict] | None,
    ) -> BatchResult:
        new_messages, new_deferred, new_events = self._flush(slots)
        update: dict = {
            "messages": new_messages + new_deferred,
            "events": new_events,
            "attempt_count": attempt_count + 1,
            "failure_count": failure_count + acc.new_failures,
            "completed_tool_call_prefix": list(already_done) + list(acc.new_completed_ids),
            "pending_ask_outcome": payload.pending_ask_outcome,
            "pending_ask_tool_call_id": payload.pending_ask_tool_call_id,
            "pending_ask_artifact": payload.pending_ask_artifact,
            "pending_ask_tool_args": payload.pending_ask_tool_args,
        }
        cleanup = resume_cleanup() if resume_cleanup is not None else {}
        if cleanup:
            update.update(cleanup)
        return BatchResult(
            update=update,
            interrupted=True,
            should_interrupt=acc.should_interrupt,
            ask_payload=payload,
        )

    def _clean_result(
        self,
        acc: _BatchAccumulator,
        slots: dict[int, PerTcResult],
        *,
        has_prior_soft_hint: bool,
        attempt_count: int,
        failure_count: int,
        resume_cleanup: Callable[[], dict] | None,
    ) -> BatchResult:
        new_messages, new_deferred, new_events = self._flush(slots)
        flushed = new_messages + new_deferred
        update: dict = {
            "messages": flushed,
            "events": new_events,
            "attempt_count": attempt_count + 1,
            "failure_count": failure_count + acc.new_failures,
            "completed_tool_call_prefix": [],
            "approved_tool_call_ids": [],
            "pending_ask_outcome": None,
            "pending_ask_tool_call_id": None,
            "pending_ask_artifact": None,
            "pending_ask_tool_args": None,
        }
        cleanup = resume_cleanup() if resume_cleanup is not None else {}
        if cleanup:
            update.update(cleanup)
        if acc.should_interrupt:
            update["should_interrupt"] = True
        # 镜像抽取前 react_graph 批尾 soft-hint 扫描（B1-1a 抽取时的现状语义）；表达式 bug-for-bug，
        # 不加 isinstance —— HumanMessage 也有 .content/.name 属性。
        acc.soft_hint_sent_this_batch = bool(
            not has_prior_soft_hint
            and any(
                getattr(m, "content", None) == "SOFT_HINT"
                and getattr(m, "name", None) == "message_ask_user"
                for m in flushed
            )
        )
        if acc.soft_hint_sent_this_batch:
            update["soft_hint_sent"] = True
        return BatchResult(
            update=update,
            interrupted=False,
            should_interrupt=acc.should_interrupt,
            ask_payload=None,
        )
