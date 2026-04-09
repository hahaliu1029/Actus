"""Test that cancel() does NOT remove task from registry immediately."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.infrastructure.external.task.redis_stream_task import RedisStreamTask


@pytest.fixture
def mock_task_runner():
    runner = AsyncMock()
    runner.invoke = AsyncMock(side_effect=asyncio.CancelledError)
    runner.on_done = AsyncMock()
    runner.destroy = AsyncMock()
    return runner


@pytest.mark.anyio
async def test_cancel_does_not_remove_from_registry_immediately(mock_task_runner):
    """A5: After cancel(), task must still be findable via Task.get() until coroutine finishes."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "app.infrastructure.external.message_queue.redis_stream_message_queue.RedisStreamMessageQueue.__init__",
            lambda self, name: setattr(self, "_stream_name", name),
        )
        task = RedisStreamTask(mock_task_runner)
        task_id = task.id

        assert RedisStreamTask.get(task_id) is task

        await task.invoke()
        task.cancel(reason="stop")

        # Key assertion: task must STILL be in registry after cancel()
        assert RedisStreamTask.get(task_id) is task, \
            "cancel() must not remove task from registry — _on_task_done handles cleanup"

        # Cleanup
        RedisStreamTask._task_registry.pop(task_id, None)
