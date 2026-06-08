"""C2b §4.4 — RedisStreamTask carries a ChildScopeViolation across the
`_execute_task` exception swallow so the adapter can re-raise it."""
import asyncio

import pytest
from unittest.mock import AsyncMock

from app.domain.services.permission.child_scope_gate import ScopeDecision
from app.domain.services.permission.child_scope_violation import ChildScopeViolation
from app.infrastructure.external.task.redis_stream_task import RedisStreamTask

pytestmark = pytest.mark.anyio


def _patch_queue(mp):
    mp.setattr(
        "app.infrastructure.external.message_queue."
        "redis_stream_message_queue.RedisStreamMessageQueue.__init__",
        lambda self, name: setattr(self, "_stream_name", name),
    )


def test_child_scope_violation_defaults_none():
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        task = RedisStreamTask(AsyncMock())
        assert task.child_scope_violation is None
        RedisStreamTask._task_registry.pop(task.id, None)


def test_set_child_scope_violation_stores():
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)
        task = RedisStreamTask(AsyncMock())
        viol = ChildScopeViolation(
            ScopeDecision.OUT_OF_PATH_LEASE, tool_name="file_write",
            target_path="/x",
        )
        task.set_child_scope_violation(viol)
        assert task.child_scope_violation is viol
        RedisStreamTask._task_registry.pop(task.id, None)


async def test_stash_survives_execute_task_swallow():
    """A runner that stashes then raises ChildScopeViolation: _execute_task
    swallows the raise (except Exception) but the stash must survive."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_queue(mp)

        viol = ChildScopeViolation(
            ScopeDecision.HARD_BLOCKED, tool_name="shell_execute",
        )

        class _StashingRunner:
            async def invoke(self, task):
                task.set_child_scope_violation(viol)
                raise viol

            async def on_done(self, task):
                return None

            async def destroy(self):
                return None

        task = RedisStreamTask(_StashingRunner())
        await task.invoke()  # backgrounds _execute_task
        for _ in range(1000):
            if task.done:
                break
            await asyncio.sleep(0)
        assert task.done
        assert task.child_scope_violation is viol  # survived the swallow
        RedisStreamTask._task_registry.pop(task.id, None)
