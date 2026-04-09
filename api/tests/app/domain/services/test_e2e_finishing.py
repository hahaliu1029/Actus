"""Behavioral integration tests for E1 FINISHING cancel path."""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.mark.anyio
async def test_cancel_via_task_cancel_propagates_to_postprocess():
    """CancelledError on _run_postprocess_or_cancel propagates to inner postprocess task.

    Scope: tests _run_postprocess_or_cancel in isolation. Does NOT cover the full
    stop_session/delete_session → task.cancel() → invoke chain (requires integration test).
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._flow = MagicMock()
    runner._memory_flusher = None

    # _do_postprocess that hangs and tracks whether it was cancelled
    postprocess_was_cancelled = asyncio.Event()
    async def slow_postprocess(task):
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            postprocess_was_cancelled.set()
            raise
    runner._do_postprocess = slow_postprocess

    # Mock task with input_stream that stays empty
    task = MagicMock()
    task.input_stream = MagicMock()
    task.input_stream.is_empty = AsyncMock(return_value=True)

    # Start _run_postprocess_or_cancel, then cancel it (simulates task.cancel → CancelledError)
    run_task = asyncio.create_task(runner._run_postprocess_or_cancel(task))
    await asyncio.sleep(0.05)  # let it start polling

    run_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run_task

    # Key assertion: the INNER postprocess asyncio.Task was actually cancelled,
    # not left running as an orphan
    assert postprocess_was_cancelled.is_set(), \
        "CancelledError must propagate to inner postprocess task (no orphan)"


@pytest.mark.anyio
async def test_new_message_cancels_inner_postprocess_task():
    """New message during postprocess: inner asyncio.Task must be cancelled, not just abandoned.

    This is the core anti-orphan guarantee: _run_postprocess_or_cancel must cancel()
    the inner task AND await it before returning True.
    """
    from app.domain.services.agent_task_runner import AgentTaskRunner

    runner = object.__new__(AgentTaskRunner)
    runner._flow = MagicMock()
    runner._memory_flusher = None

    # Track inner postprocess lifecycle
    postprocess_started = asyncio.Event()
    postprocess_was_cancelled = asyncio.Event()

    async def slow_postprocess(task):
        postprocess_started.set()
        try:
            await asyncio.sleep(999)
        except asyncio.CancelledError:
            postprocess_was_cancelled.set()
            raise
    runner._do_postprocess = slow_postprocess

    # input_stream: empty for first 2 checks, then has message
    check_count = [0]
    async def is_empty():
        check_count[0] += 1
        return check_count[0] < 3  # returns False on 3rd check → "new message"
    task = MagicMock()
    task.input_stream = MagicMock()
    task.input_stream.is_empty = is_empty

    result = await asyncio.wait_for(
        runner._run_postprocess_or_cancel(task),
        timeout=5.0,
    )

    assert result is True, "Must return True when message cancels postprocess"
    assert postprocess_started.is_set(), "Inner postprocess must have started"
    assert postprocess_was_cancelled.is_set(), \
        "Inner postprocess task must be cancelled (not orphaned) when new message arrives"
