"""C2 PR-3 §8.5.3 — child-side mailbox listener for coordinator child sessions.

Subscribes the root-scoped stream ``actus:child:{root_session_id}:mailbox``
with an **independent** consumer group ``coordinator:child:{child_session_id}``
(distinct from the supervisor's ``actus:mailbox-supervisor:v1`` group).

Filter predicate: ``envelope.child_session_id == self._child_session_id``
AND ``envelope.type == "CANCEL_REQUEST"``.

On match → ``runner.request_stop(StopReason.PARENT_CANCEL)`` (the runner-side
sole entry that sets ``cancel_event`` + ``_stop_reason``).

Pre-subscribe race (spec §8.5.3 race table):
  If a CANCEL_REQUEST is published BEFORE the listener finishes subscribing,
  XREADGROUP with ``id="$"`` will not deliver it. The supervisor's pre-existing
  cancel handling + child wallclock budget watchdog (PR-6) act as backstop —
  ``TIMED_OUT`` outcome is the accepted v1 path. PR-4 hardens with a startup
  fence (CANCEL_ACK on graceful shutdown) but cannot fully eliminate the race.
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

    async def start(self) -> None:
        await self._subscriber.subscribe(
            stream_key=self._stream_key,
            consumer_group=self._consumer_group,
            consumer_name=self._consumer_name,
        )
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
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

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
