"""C2b §8 test 5 — typed-propagation seam: REAL AgentTaskRunnerInvokeAdapter +
REAL RedisStreamTask (in-memory message queue, no Redis) + a fake runner that
stashes then raises ChildScopeViolation. Proves the adapter re-raises the typed
violation across the real RedisStreamTask._execute_task swallow (→
NEEDS_AUTHORIZATION end-to-end), closing the middle seam the Task 1 + Task 2
unit tests only cover separately."""
import asyncio

import pytest

from app.application.services.agent_task_runner_invoke_adapter import (
    AgentTaskRunnerInvokeAdapter,
)
from app.domain.services.permission.child_scope_gate import ScopeDecision
from app.domain.services.permission.child_scope_violation import ChildScopeViolation
from app.infrastructure.external.task.redis_stream_task import RedisStreamTask

pytestmark = pytest.mark.anyio


class _InMemoryQueue:
    """Minimal in-memory stand-in for RedisStreamMessageQueue (put/get)."""
    def __init__(self, name):
        self._name = name
        self._items: list[str] = []
        self._idx = 0

    async def put(self, message):
        self._items.append(message)
        return str(len(self._items) - 1)

    async def get(self, start_id=None, block_ms=None):
        if self._idx < len(self._items):
            i = self._idx
            self._idx += 1
            return (str(i), self._items[i])
        await asyncio.sleep(0)  # yield so the backgrounded _execute_task can run
        return (None, None)


class _StashRaiseRunner:
    """TaskRunner-shaped fake: invoke stashes the violation then raises (mirror
    AgentTaskRunner.invoke's `except ChildScopeViolation` stash-then-raise)."""
    def __init__(self, violation):
        self._violation = violation

    async def invoke(self, task):
        task.set_child_scope_violation(self._violation)
        raise self._violation

    async def on_done(self, task):
        return None

    async def destroy(self):
        return None


async def test_violation_propagates_through_real_redis_stream_task(monkeypatch):
    # Substitute the Redis-backed queue with an in-memory one (no Redis).
    monkeypatch.setattr(
        "app.infrastructure.external.task.redis_stream_task.RedisStreamMessageQueue",
        _InMemoryQueue,
    )
    viol = ChildScopeViolation(
        ScopeDecision.OUT_OF_PATH_LEASE, tool_name="file_write",
        target_path="/forbidden",
    )
    runner = _StashRaiseRunner(viol)
    adapter = AgentTaskRunnerInvokeAdapter(
        runner=runner, cancel_event=asyncio.Event(), task_cls=RedisStreamTask,
    )
    with pytest.raises(ChildScopeViolation) as ei:
        await adapter.invoke_until_done(user_message="x")
    assert ei.value is viol  # the typed violation, NOT ChildInnerRunError
    # _on_task_done removes the task from the registry; be defensive in case the
    # background on_done task hasn't drained yet.
    for tid in list(RedisStreamTask._task_registry):
        RedisStreamTask._task_registry.pop(tid, None)
