import asyncio
import pytest
from app.application.services.agent_task_runner_invoke_adapter import (
    AgentTaskRunnerInvokeAdapter,
    ChildInnerRunError,
)
from app.application.services.coordinator_child_runner import ChildRunResult
from app.domain.models.event import (
    DoneEvent, ErrorEvent, ToolEvent, ToolEventStatus, WaitEvent,
)
from app.domain.services.graphs.react_graph import CancelledByEventError

pytestmark = pytest.mark.anyio


class _FakeStream:
    """In-memory stand-in for a RedisStreamMessageQueue."""
    def __init__(self):
        self._items: list[str] = []
        self._idx = 0

    async def put(self, message: str) -> str:
        self._items.append(message)
        return str(len(self._items) - 1)

    async def get(self, start_id=None, block_ms=None):
        if self._idx < len(self._items):
            i = self._idx
            self._idx += 1
            return (str(i), self._items[i])
        await asyncio.sleep(0)  # yield so a not-yet-done task can flip done
        return (None, None)


class _FakeTask:
    """Mimics RedisStreamTask's create/invoke/done/cancel contract."""
    def __init__(self, runner):
        self._runner = runner
        self.input_stream = _FakeStream()
        self.output_stream = _FakeStream()
        self.done = False
        self.cancelled_with = None

    @classmethod
    def create(cls, *, task_runner):
        return cls(task_runner)

    async def invoke(self):
        # The fake runner's script writes terminal/tool events onto output_stream
        # and flips done. Real RedisStreamTask backgrounds this; for the unit
        # test we run it inline so the drain sees events deterministically.
        await self._runner.run(self)
        self.done = True

    def cancel(self, reason="stop"):
        self.cancelled_with = reason
        self.done = True
        return True


class _ScriptedRunner:
    """Pushes a scripted event sequence onto the task's output_stream."""
    def __init__(self, events):
        self._events = events
        self.cancel_event_attr = None

    def set_coordinator_cancel_event(self, event):
        self.cancel_event_attr = event

    async def run(self, task):
        for ev in self._events:
            await task.output_stream.put(ev.model_dump_json())


async def test_adapter_returns_child_run_result_with_tool_calls_on_done():
    tool_ev = ToolEvent(
        tool_call_id="tc1", tool_name="file", function_name="file_write",
        function_args={"filepath": "a.py", "content": "x"},
        status=ToolEventStatus.CALLING,
    )
    runner = _ScriptedRunner([tool_ev, DoneEvent(metrics={"ok": 1})])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    result = await adapter.invoke_until_done(user_message="do it")
    assert isinstance(result, ChildRunResult)
    assert isinstance(result.done_event, DoneEvent)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function_name == "file_write"


async def test_adapter_raises_on_error_terminal():
    runner = _ScriptedRunner([ErrorEvent(error="boom")])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    with pytest.raises(ChildInnerRunError):
        await adapter.invoke_until_done(user_message="do it")


async def test_adapter_raises_cancelled_when_event_set():
    cancel = asyncio.Event()
    cancel.set()  # already cancelled before drain
    runner = _ScriptedRunner([DoneEvent()])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=cancel, task_cls=_FakeTask,
    )
    with pytest.raises(CancelledByEventError):
        await adapter.invoke_until_done(user_message="do it")


async def test_adapter_wires_cancel_event_into_runner_on_construction():
    cancel = asyncio.Event()
    runner = _ScriptedRunner([DoneEvent()])
    AgentTaskRunnerInvokeAdapter(runner=runner, cancel_event=cancel, task_cls=_FakeTask)
    assert runner.cancel_event_attr is cancel  # set_coordinator_cancel_event called


async def test_adapter_waits_for_task_done_after_done_event():
    """[R3 P1] The adapter captures DoneEvent but must NOT return until task.done
    (so the runner's post-DoneEvent terminal cost flush completes before
    RESULT_READY publishes — else the reducer's ledger read races → cost 0)."""

    class _SlowFlushTask(_FakeTask):
        async def invoke(self):
            # Emit DoneEvent but stay not-done; flip done shortly AFTER the drain
            # has already seen DoneEvent (simulates post-DoneEvent flush_pending).
            await self.output_stream.put(DoneEvent().model_dump_json())

            async def _flip():
                await asyncio.sleep(0.03)
                self.done = True

            asyncio.create_task(_flip())  # done stays False on return

    runner = _ScriptedRunner([])  # the task itself emits the terminal
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_SlowFlushTask,
    )
    result = await adapter.invoke_until_done(user_message="x")
    assert isinstance(result.done_event, DoneEvent)  # returned only after done flipped


async def test_adapter_raises_when_task_done_without_terminal_event():
    """Runner emits zero events, task flips done, cancel NOT set → the drain
    sees a None pull with task.done and raises (not a silent empty result)."""
    runner = _ScriptedRunner([])  # no terminal emitted
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    with pytest.raises(ChildInnerRunError, match="child task ended without a terminal event"):
        await adapter.invoke_until_done(user_message="do it")


async def test_adapter_raises_on_non_done_terminal():
    """A WaitEvent is a terminal type but NOT DoneEvent — a coordinator child
    must run to a DoneEvent, so any other terminal is an abnormal stop.

    (WaitEvent() constructs with zero args; ControlEvent requires `action` and
    a scope-when-REQUESTED validator, so WaitEvent is the clean choice here.)
    """
    runner = _ScriptedRunner([WaitEvent()])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    with pytest.raises(ChildInnerRunError, match="non-done terminal"):
        await adapter.invoke_until_done(user_message="do it")


async def test_adapter_error_event_with_empty_message_uses_fallback():
    """ErrorEvent(error="") → the `event.error or ...` fallback supplies a
    non-empty 'child runner error' message."""
    runner = _ScriptedRunner([ErrorEvent(error="")])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    with pytest.raises(ChildInnerRunError, match="child runner error"):
        await adapter.invoke_until_done(user_message="do it")


async def test_await_task_done_bounded_cancels_on_timeout():
    """[R4 P1] _await_task_done_bounded must not hang forever: on a task that
    never flips done, after grace_seconds it cancels the task and returns."""
    runner = _ScriptedRunner([])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_FakeTask,
    )
    task = _FakeTask(runner)  # done stays False; never flipped
    await adapter._await_task_done_bounded(task, grace_seconds=0.05)
    assert task.cancelled_with == "coordinator_post_done_timeout"


async def test_adapter_cancels_inner_task_when_drain_unwinds_via_cancellederror():
    """[R3 Fix-1] An outer asyncio.CancelledError unwinding the drain (e.g. the
    coordinator task is cancelled while blocked in output_stream.get) must cancel
    the backgrounded inner task — otherwise it orphans (coordinator-step terminal
    publisher is disabled, so nothing else reaps it). The adapter re-raises.

    The cancel_event is NOT set, so _drain's cooperative branch is skipped and
    the raw CancelledError propagates out of _drain into the outer guard. invoke
    is a no-op here so task.done stays False (the guard's `if not task.done`
    fires); cancel records the reason as 'adapter_unwind'.
    """
    class _UnwindOnGetTask(_FakeTask):
        async def invoke(self):
            # Do NOT run the scripted runner / flip done — the real
            # RedisStreamTask backgrounds execution, so done stays False here.
            return None

    runner = _ScriptedRunner([])  # never consulted
    task = _UnwindOnGetTask(runner)

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    task.output_stream.get = _raise_cancelled  # type: ignore[assignment]

    class _SingleTaskFactory:
        @staticmethod
        def create(*, task_runner):
            return task

    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(),  # NOT set
        task_cls=_SingleTaskFactory,
    )
    assert task.done is False  # precondition: guard's `if not task.done` will fire
    with pytest.raises(asyncio.CancelledError):
        await adapter.invoke_until_done(user_message="x")
    assert task.cancelled_with == "adapter_unwind"


async def test_adapter_cancels_inner_task_when_drain_unwinds():
    """[R3 hardening] If an outer CancelledError propagates through the drain,
    invoke_until_done cancels the backgrounded inner task (so it cannot orphan)
    and re-raises the cancellation."""
    created: list = []

    class _CancelDrainStream:
        async def put(self, message):
            return "0"

        async def get(self, start_id=None, block_ms=None):
            raise asyncio.CancelledError()

    class _CancelDrainTask:
        def __init__(self, runner):
            self.input_stream = _CancelDrainStream()
            self.output_stream = _CancelDrainStream()
            self.done = False
            self.cancelled_with = None

        @classmethod
        def create(cls, *, task_runner):
            t = cls(task_runner)
            created.append(t)
            return t

        async def invoke(self):
            return None

        def cancel(self, reason="stop"):
            self.cancelled_with = reason
            self.done = True
            return True

    runner = _ScriptedRunner([])
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=_CancelDrainTask,
    )
    with pytest.raises(asyncio.CancelledError):
        await adapter.invoke_until_done(user_message="x")
    assert created, "task was never created"
    assert created[0].cancelled_with == "adapter_unwind"  # inner task cancelled, not orphaned
    assert created[0].done is True
