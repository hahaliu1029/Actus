"""C2 PR-3 §8.5.3 — child-side mailbox listener for coordinator child sessions.

Subscribes the root-scoped stream ``actus:child:{root_session_id}:mailbox``
with an **independent** consumer group ``coordinator:child:{child_session_id}``
(distinct from the supervisor's ``actus:mailbox-supervisor:v1`` group).

Filter predicate: ``envelope.child_session_id == self._child_session_id``
AND ``envelope.type == "CANCEL_REQUEST"``.

On match → ``runner.request_stop(StopReason.PARENT_CANCEL)`` (the runner-side
sole entry that sets ``cancel_event`` + ``_stop_reason``).

Pre-subscribe race (spec §8.5.3 race table — CLOSED by C2b budget §3-9 R4#1):
  dispatch pre-creates this listener's consumer group (same loop as the
  waiter-group hoist in parallel_execution_subgraph._first_time_dispatch), so
  a CANCEL_REQUEST published before the listener subscribes is retained as
  group backlog and consumed on start — subscribe here is BUSYGROUP-idempotent
  against that pre-creation. Independently, the child wallclock watchdog is
  LIVE (C2b budget A1) as the runaway brake; it is no longer a backstop for
  this (closed) race.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from app.domain.external.mailbox_subscriber import MailboxSubscriber

if TYPE_CHECKING:
    from app.application.services.coordinator_child_runner import CoordinatorChildRunner

logger = logging.getLogger(__name__)


class CoordinatorChildCancelListener:
    """Listens for CANCEL_REQUEST envelopes addressed to this child session."""

    def __init__(
        self,
        *,
        subscriber: MailboxSubscriber,
        root_session_id: str,
        child_session_id: str,
        runner: "CoordinatorChildRunner",
    ) -> None:
        self._subscriber = subscriber
        self._root_session_id = root_session_id
        self._child_session_id = child_session_id
        self._runner = runner
        self.ready_event: asyncio.Event = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._stream_key = f"actus:child:{root_session_id}:mailbox"
        self._consumer_group = f"coordinator:child:{child_session_id}"
        self._consumer_name = f"{child_session_id}-listener"
        # [Round 7 P2] Track whether subscribe() succeeded so shutdown() can
        # destroy the per-listener consumer group without emitting a spurious
        # NOGROUP roundtrip when subscribe never ran (e.g. start() raised
        # before subscribe, or start() was never called).
        self._subscribed: bool = False

    async def start(self) -> None:
        await self._subscriber.subscribe(
            stream_key=self._stream_key,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
        )
        self._subscribed = True
        self.ready_event.set()
        self._task = asyncio.create_task(self._listen_loop_forever())
        # r5 P1-3: attach done-callback to surface fatal listener death.
        # Without this, an unhandled exception inside consume()/predicate()/
        # xack would kill the listener task silently — subsequent CANCEL_REQUEST
        # envelopes would never be delivered to the runner, leaving the child
        # uncancelable until the wallclock watchdog trips.
        self._task.add_done_callback(self._on_task_done)

    @staticmethod
    def _on_task_done(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(
                "CoordinatorChildCancelListener task died with unhandled exception: %r",
                exc, exc_info=exc,
            )

    async def shutdown(self, timeout: float = 1.0) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        # [Round 7 P2] Destroy the per-listener consumer group so it doesn't
        # accumulate as a dead XPENDING/group-metadata entry under the
        # long-lived root stream. Skip when ``_subscribed=False`` (subscribe
        # never ran — nothing to destroy). Best-effort: log + swallow on any
        # error — the group is per-listener and idempotent destroy is the
        # adapter contract. Positioned AFTER the task drain so any in-flight
        # XACK on the listener task completes before the group is torn down.
        if self._subscribed:
            try:
                await self._subscriber.destroy_group(
                    stream_key=self._stream_key,
                    consumer_group=self._consumer_group,
                )
            except Exception:
                logger.warning(
                    "CoordinatorChildCancelListener: destroy_group failed"
                    " group=%s — dead group may accumulate in Redis",
                    self._consumer_group,
                    exc_info=True,
                )

    async def _predicate(self, env: dict[str, Any]) -> bool:
        return (
            env.get("child_session_id") == self._child_session_id
            and env.get("type") == "CANCEL_REQUEST"
        )

    async def _listen_loop_one_iteration(self) -> None:
        """Test seam — one consume cycle (uses ``max_iterations=1``)."""
        async for env in self._subscriber.consume(
            stream_key=self._stream_key,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
            predicate=self._predicate,
            max_iterations=1,
        ):
            await self._handle_cancel_request(env)

    async def _listen_loop_forever(self) -> None:
        async for env in self._subscriber.consume(
            stream_key=self._stream_key,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
            predicate=self._predicate,
        ):
            await self._handle_cancel_request(env)

    async def _handle_cancel_request(self, env: dict[str, Any]) -> None:
        # Local import avoids module-load cycle with coordinator_child_runner.
        from app.application.services.coordinator_child_runner import StopReason

        logger.info(
            "coordinator child %s received CANCEL_REQUEST (envelope_id=%s)",
            self._child_session_id, env.get("envelope_id"),
        )
        self._runner.request_stop(StopReason.PARENT_CANCEL)
