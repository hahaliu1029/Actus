"""B5.5 T1 — RedisStreamTask binds the observability session context around
``runner.invoke`` / ``runner.resume`` so downstream prompt-assembly and
LLM-invocation telemetry records carry a non-null ``session_id`` (consumed by
B5.5 cache-viability analysis).

Regression lock for the ``bind_session_context()`` zero-call-site gap: before
this wiring telemetry ``session_id`` was always null in production, which made
the per-session cache-viability grouping impossible.

Binding is BEST-EFFORT: a runner that does not expose a ``str`` session_id
(test doubles, future runner types) — or whose ``session_id`` accessor raises —
simply runs unbound; telemetry must never break task execution (mirrors the
swallow-all philosophy of the observability subsystem). The AsyncExitStack also
guarantees the binding is RESET when the runner returns, raises, or is
cancelled — verified below via the post-stack ``on_done`` capture.
"""
import asyncio

import pytest

from app.infrastructure.external.task.redis_stream_task import RedisStreamTask
from app.infrastructure.observability.context import get_trace_context

pytestmark = pytest.mark.anyio


def _patch_queue(mp):
    mp.setattr(
        "app.infrastructure.external.message_queue."
        "redis_stream_message_queue.RedisStreamMessageQueue.__init__",
        lambda self, name: setattr(self, "_stream_name", name),
    )


class _SessionCapturingRunner:
    """TaskRunner-duck-typed fake that records the observability session_id
    visible at the instant ``invoke`` / ``resume`` runs, and again inside
    ``on_done`` (which runs AFTER the AsyncExitStack has closed) so a test can
    assert the binding was reset."""

    _SENTINEL = "UNSET"

    def __init__(self, session_id, *, raise_in_invoke=False):
        self.session_id = session_id
        self.invoked_session_id = self._SENTINEL
        self.resumed_session_id = self._SENTINEL
        self.on_done_session_id = self._SENTINEL
        self.on_done_event = asyncio.Event()
        self._raise_in_invoke = raise_in_invoke

    async def invoke(self, task):
        ctx = get_trace_context()
        self.invoked_session_id = ctx.session_id if ctx is not None else None
        if self._raise_in_invoke:
            raise RuntimeError("boom in invoke")

    async def resume(self, task, command):
        ctx = get_trace_context()
        self.resumed_session_id = ctx.session_id if ctx is not None else None

    async def on_done(self, task):
        ctx = get_trace_context()
        self.on_done_session_id = ctx.session_id if ctx is not None else None
        self.on_done_event.set()

    async def destroy(self):
        return None


class _RaisingSessionIdRunner:
    """A runner whose ``session_id`` accessor raises a non-AttributeError —
    the binding setup must swallow it and STILL run ``invoke``."""

    def __init__(self):
        self.ran = False
        self.on_done_event = asyncio.Event()

    @property
    def session_id(self):
        raise RuntimeError("session_id accessor boom")

    async def invoke(self, task):
        self.ran = True

    async def on_done(self, task):
        self.on_done_event.set()

    async def destroy(self):
        return None


async def _drain(task):
    for _ in range(1000):
        if task.done:
            break
        await asyncio.sleep(0)
    assert task.done


async def test_execute_task_binds_session_context():
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _SessionCapturingRunner("sess-invoke-123")
        task = RedisStreamTask(runner)
        await task.invoke()  # backgrounds _execute_task
        await _drain(task)
        assert runner.invoked_session_id == "sess-invoke-123"
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_execute_resume_binds_session_context():
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _SessionCapturingRunner("sess-resume-456")
        task = RedisStreamTask(runner)
        await task.resume(command=None)  # backgrounds _execute_resume
        await _drain(task)
        assert runner.resumed_session_id == "sess-resume-456"
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_execute_task_without_session_id_runs_unbound():
    """No str session_id → best-effort skip; execution still proceeds."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _SessionCapturingRunner(session_id=None)
        task = RedisStreamTask(runner)
        await task.invoke()
        await _drain(task)
        assert runner.invoked_session_id is None  # ran, context stayed unbound
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_session_context_reset_after_successful_run():
    """The AsyncExitStack must reset the binding once invoke returns: on_done
    (created in the `finally`, AFTER the stack closes) sees no binding."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _SessionCapturingRunner("sess-reset-ok")
        task = RedisStreamTask(runner)
        await task.invoke()
        await _drain(task)
        await asyncio.wait_for(runner.on_done_event.wait(), timeout=1.0)
        assert runner.invoked_session_id == "sess-reset-ok"  # bound during invoke
        assert runner.on_done_session_id is None  # reset after stack closed
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_session_context_reset_when_runner_raises():
    """Binding must reset even when invoke raises (the exception is swallowed
    by _execute_task); on_done still observes an unbound context."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _SessionCapturingRunner("sess-reset-raise", raise_in_invoke=True)
        task = RedisStreamTask(runner)
        await task.invoke()
        await _drain(task)  # RuntimeError swallowed → task still done
        await asyncio.wait_for(runner.on_done_event.wait(), timeout=1.0)
        assert runner.invoked_session_id == "sess-reset-raise"  # bound before raise
        assert runner.on_done_session_id is None  # reset despite the exception
        assert task.done
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_raising_session_id_accessor_does_not_skip_execution():
    """If the runner's session_id accessor raises, the best-effort binding must
    swallow it and STILL run invoke — binding failure must never silently skip
    agent execution."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _RaisingSessionIdRunner()
        task = RedisStreamTask(runner)
        await task.invoke()
        await _drain(task)
        await asyncio.wait_for(runner.on_done_event.wait(), timeout=1.0)
        assert runner.ran is True  # invoke ran despite session_id accessor boom
        RedisStreamTask._task_registry.pop(task.id, None)


class _BlockingRunner:
    """Blocks inside invoke until cancelled, so a test can cancel the task
    mid-invoke and assert the binding is reset on the cancellation path."""

    def __init__(self):
        self.session_id = "sess-cancel"
        self.invoked_session_id = "UNSET"
        self.on_done_session_id = "UNSET"
        self.started = asyncio.Event()
        self.on_done_event = asyncio.Event()

    async def invoke(self, task):
        ctx = get_trace_context()
        self.invoked_session_id = ctx.session_id if ctx is not None else None
        self.started.set()
        await asyncio.sleep(3600)  # block until cancelled

    async def on_done(self, task):
        ctx = get_trace_context()
        self.on_done_session_id = ctx.session_id if ctx is not None else None
        self.on_done_event.set()

    async def destroy(self):
        return None


async def test_session_context_reset_on_cancellation():
    """Cancelling mid-invoke must still reset the binding (AsyncExitStack
    __aexit__ runs on CancelledError); on_done (post-stack) sees no binding."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        runner = _BlockingRunner()
        task = RedisStreamTask(runner)
        await task.invoke()
        await asyncio.wait_for(runner.started.wait(), timeout=1.0)  # bound + blocked
        assert runner.invoked_session_id == "sess-cancel"
        task.cancel()  # cancels the background _execution_task mid-invoke
        await _drain(task)  # CancelledError propagates → task done
        await asyncio.wait_for(runner.on_done_event.wait(), timeout=1.0)
        assert runner.on_done_session_id is None  # reset on the cancellation path
        RedisStreamTask._task_registry.pop(task.id, None)
