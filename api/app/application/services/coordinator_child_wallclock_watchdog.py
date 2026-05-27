"""[C2 PR-6 §14.3 #5] Wallclock watchdog as an ``asyncio.Task``.

Started alongside the inner ReAct invocation; on timeout calls
``runner.request_stop(StopReason.WALLCLOCK_BUDGET)`` which the runner's
finalizer routes to ``RESULT_READY(NEEDS_AUTHORIZATION, reason=budget_exhausted)``.

The internal cap (typically 300s) trips BEFORE the outer supervisor's 600s
backstop so the parent sees a structured ``budget_exhausted`` outcome with
the option to raise the cap, instead of the ``TIMED_OUT`` outcome the
supervisor would emit on its hard backstop.

Caller contract:
- Capture the returned ``asyncio.Task``.
- Cancel it on early child completion (the watchdog quietly exits on
  CancelledError so the cancel is safe to call unconditionally inside a
  ``finally`` block).
- Do NOT await the task in the happy path — let it run concurrently with
  the inner runner.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.application.services.coordinator_child_runner import (
        CoordinatorChildRunner,
    )


logger = logging.getLogger(__name__)


async def _watchdog_loop(
    runner: "CoordinatorChildRunner",
    max_wallclock_seconds: float,
) -> None:
    """Sleep for the budget; on wake, request_stop(WALLCLOCK_BUDGET).

    Caller cancels this task on early child completion → ``CancelledError``
    bubbles out of ``asyncio.sleep`` and is swallowed below (the caller
    finished in time; nothing more to do).
    """
    try:
        await asyncio.sleep(max_wallclock_seconds)
    except asyncio.CancelledError:
        # Happy path: caller cancelled because the child completed before
        # the budget elapsed. Quiet exit — re-raising would propagate the
        # cancellation up to the caller, which is exactly what they DON'T
        # want when they cancelled us deliberately.
        return

    # Lazy import — avoids a top-level circular: ``coordinator_child_runner``
    # already imports from ``coordinator_child_cancel_listener``, and
    # PR-6 wiring will import this module from the runner module path.
    from app.application.services.coordinator_child_runner import StopReason

    logger.info(
        "wallclock watchdog trip after %.1fs → request_stop(WALLCLOCK_BUDGET)",
        max_wallclock_seconds,
    )
    runner.request_stop(StopReason.WALLCLOCK_BUDGET)


def start_wallclock_watchdog(
    *,
    runner: "CoordinatorChildRunner",
    max_wallclock_seconds: float,
) -> asyncio.Task:
    """Spawn the watchdog as an ``asyncio.Task`` and return it.

    Returns the live task so the caller can ``task.cancel()`` it on early
    child completion. The task is named ``coordinator-wallclock-watchdog``
    for trace-readability — when an unhandled exception escapes the loop
    (no path currently raises, but defensive naming aids future debugging),
    the asyncio default handler logs the task name.
    """
    return asyncio.create_task(
        _watchdog_loop(runner, max_wallclock_seconds),
        name="coordinator-wallclock-watchdog",
    )
