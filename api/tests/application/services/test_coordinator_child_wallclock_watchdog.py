"""C2 PR-6 Task 6.2 — coordinator_child_wallclock_watchdog unit tests.

Pins the §14.3 #5 contract:

- After ``max_wallclock_seconds`` elapses, ``runner.request_stop(WALLCLOCK_BUDGET)``
  fires exactly once.
- The returned ``asyncio.Task`` is cancelable — caller cancels it on early
  child completion so the watchdog never trips for a child that finished
  in time.
- The internal cap (e.g. 300s) is configured by caller; the test uses
  short sleeps (sub-second) so suite runtime stays small.
"""
from __future__ import annotations

import asyncio

import pytest
from unittest.mock import MagicMock

from app.application.services.coordinator_child_runner import StopReason
from app.application.services.coordinator_child_wallclock_watchdog import (
    start_wallclock_watchdog,
)


pytestmark = pytest.mark.anyio


def _mk_runner() -> MagicMock:
    runner = MagicMock()
    runner.request_stop = MagicMock()
    return runner


async def test_returns_asyncio_task() -> None:
    """``start_wallclock_watchdog`` returns an ``asyncio.Task`` instance."""
    runner = _mk_runner()
    task = start_wallclock_watchdog(
        runner=runner, max_wallclock_seconds=60.0,
    )
    try:
        assert isinstance(task, asyncio.Task)
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


async def test_trips_on_timeout() -> None:
    """After the wallclock budget elapses, runner.request_stop is called once."""
    runner = _mk_runner()
    task = start_wallclock_watchdog(
        runner=runner, max_wallclock_seconds=0.05,
    )

    # Wait ~3x the budget so the watchdog reliably trips.
    await asyncio.sleep(0.15)

    runner.request_stop.assert_called_once_with(StopReason.WALLCLOCK_BUDGET)

    # Task should have completed naturally (not still running).
    assert task.done()
    # Reap to avoid pytest "Task was destroyed but it is pending!" warnings.
    if not task.cancelled():
        # Surface any unexpected exception from the watchdog loop.
        assert task.exception() is None


async def test_cancelable_before_trip() -> None:
    """Cancel the task before the budget elapses → request_stop NOT called."""
    runner = _mk_runner()
    task = start_wallclock_watchdog(
        runner=runner, max_wallclock_seconds=60.0,
    )

    # Brief sleep so the task is actually scheduled inside asyncio.sleep().
    await asyncio.sleep(0.02)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    runner.request_stop.assert_not_called()


async def test_cancel_after_trip_is_safe() -> None:
    """Cancelling the task after it has already tripped + completed is a no-op.

    Pins that the caller's ``finally: task.cancel()`` cleanup pattern (paired
    with the early-completion cancel) is safe to call unconditionally — even
    when the watchdog has already finished naturally.
    """
    runner = _mk_runner()
    task = start_wallclock_watchdog(
        runner=runner, max_wallclock_seconds=0.05,
    )
    await asyncio.sleep(0.15)
    assert task.done()
    assert runner.request_stop.call_count == 1

    # Idempotent cleanup — must not raise.
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, asyncio.InvalidStateError):
        pass

    # No additional request_stop calls triggered by post-trip cancel.
    assert runner.request_stop.call_count == 1
