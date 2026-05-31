"""[C2 finish-core §5.1.1 G1a] Invoke-adapter wrapping a child AgentTaskRunner.

Satisfies ``CoordinatorChildInnerRunner.invoke_until_done`` by driving a
``Task`` (the same RedisStreamTask used by AgentService) and draining its
output stream. Outcome detection is via the OUTPUT STREAM + the coordinator
cancel_event — NOT by catching the child's exception, because
``RedisStreamTask._execute_task`` swallows ``Exception`` (including the
re-raised ``CancelledByEventError``) and only generic failures surface as a
terminal ``ErrorEvent``.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import TypeAdapter

from app.application.services.coordinator_child_runner import ChildRunResult
from app.domain.models.event import (
    ControlEvent, DoneEvent, ErrorEvent, Event, MessageEvent, ToolEvent,
    ToolEventStatus, WaitEvent,
)
from app.domain.services.graphs.react_graph import CancelledByEventError

logger = logging.getLogger(__name__)

# Bounded poll so a cancel set mid-block is observed within one cycle. Mirrors
# agent_service.py's OUTPUT_STREAM_POLL_BLOCK_MS usage.
_DRAIN_BLOCK_MS = 1000

_TERMINAL_TYPES = (DoneEvent, ErrorEvent, WaitEvent, ControlEvent)
_event_adapter = TypeAdapter(Event)


class ChildInnerRunError(Exception):
    """The child runner terminated with an ErrorEvent (generic failure).

    Raised so ``CoordinatorChildRunner.run_work_unit``'s ``except Exception``
    routes to ``_finalize_failed`` → RESULT_READY(FAILED).
    """


class AgentTaskRunnerInvokeAdapter:
    """Wraps a fully-constructed child ``AgentTaskRunner``."""

    def __init__(self, *, runner: Any, cancel_event: asyncio.Event, task_cls: Any) -> None:
        self._runner = runner
        self._cancel_event = cancel_event
        self._task_cls = task_cls
        # Wire the coordinator cancel_event into the child flow so react_graph
        # cancel checkpoints observe it (child has coord_deps=None → its own
        # prime no-ops, so this injection survives). §5.1.1 INV-F1.11.
        setter = getattr(runner, "set_coordinator_cancel_event", None)
        if setter is not None:
            setter(cancel_event)

    async def invoke_until_done(self, *, user_message: str) -> ChildRunResult:
        task = self._task_cls.create(task_runner=self._runner)
        await task.input_stream.put(
            MessageEvent(role="user", message=user_message).model_dump_json()
        )
        # task.invoke() backgrounds _execute_task so task.cancel() is effective.
        await task.invoke()
        try:
            terminal, tool_calls = await self._drain(task)
        except BaseException:
            # [R3 hardening] An outer cancel/error unwinding the adapter (e.g.
            # asyncio.CancelledError through output_stream.get) would otherwise
            # leave the backgrounded child RedisStreamTask running with no drainer
            # and no finalizer (coordinator-step terminal publisher is disabled).
            # Cancel the inner task so it can't orphan; then propagate.
            if not getattr(task, "done", False):
                try:
                    task.cancel("adapter_unwind")
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "adapter unwind: inner task.cancel failed", exc_info=True
                    )
            raise
        return ChildRunResult(done_event=terminal, tool_calls=tuple(tool_calls))

    async def _drain(self, task: Any):
        tool_calls: list[ToolEvent] = []
        terminal = None
        latest_event_id = None
        while True:
            if self._cancel_event.is_set():
                task.cancel("coordinator_cancel")
                raise CancelledByEventError("coordinator cancel during child run")
            event_id, event_str = await task.output_stream.get(
                start_id=latest_event_id, block_ms=_DRAIN_BLOCK_MS
            )
            if event_str is None:
                if task.done:
                    # Ended with no terminal event. If the cancel_event is set this
                    # was a cooperative cancel (the child's CancelledByEventError
                    # was swallowed by _execute_task); else it ended abnormally.
                    if self._cancel_event.is_set():
                        raise CancelledByEventError(
                            "coordinator cancel (child terminated mid-cancel)"
                        )
                    raise ChildInnerRunError(
                        "child task ended without a terminal event"
                    )
                # yield so a non-blocking queue impl can't busy-spin (prod XREAD BLOCK already blocks).
                await asyncio.sleep(0)
                continue
            latest_event_id = event_id
            event = _event_adapter.validate_json(event_str)
            if isinstance(event, ToolEvent) and event.status == ToolEventStatus.CALLING:
                tool_calls.append(event)
                continue
            if isinstance(event, ErrorEvent):
                raise ChildInnerRunError(event.error or "child runner error")
            if isinstance(event, DoneEvent):
                # R3 P1 — cost-flush race: AgentTaskRunner.invoke emits DoneEvent
                # (agent_task_runner.py:4181) BEFORE its terminal cost
                # flush_pending (:3112), and on_llm_end persists cost on a
                # background task (cost_callback_handler.py:404). Returning here
                # would let CoordinatorChildRunner publish RESULT_READY before the
                # CostRecords commit → the reducer's ledger read races → cost 0.
                # So wait (BOUNDED — R4 P1) for task.done before returning.
                terminal = event
                await self._await_task_done_bounded(task, grace_seconds=5.0)
                return terminal, tool_calls
            if isinstance(event, _TERMINAL_TYPES):
                # WaitEvent/ControlEvent: a coordinator child must run to a
                # DoneEvent; any other terminal is an abnormal stop.
                raise ChildInnerRunError(
                    f"child runner stopped on non-done terminal: {type(event).__name__}"
                )

    async def _await_task_done_bounded(self, task: Any, *, grace_seconds: float) -> None:
        """[R3 P1 + R4 P1] Wait for task.done so the runner's post-DoneEvent cost
        flush_pending commits — but BOUNDED, so a stalled terminal I/O (DB commit
        / supervisor stop / tool cleanup, agent_task_runner.py:4181-4199/4294)
        can't hang the adapter forever (graph watchdog already ended; child
        wallclock watchdog isn't wired — coordinator_child_runner.py:236)."""
        deadline = time.monotonic() + grace_seconds
        while not task.done:
            if time.monotonic() >= deadline:
                logger.warning(
                    "child task not done %.1fs after DoneEvent; cancelling to "
                    "avoid hang (cost may be partial)", grace_seconds,
                )
                task.cancel("coordinator_post_done_timeout")
                return
            await asyncio.sleep(0.05)
