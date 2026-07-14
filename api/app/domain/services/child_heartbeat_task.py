"""C3 ChildHeartbeatTask — independent 15s heartbeat loop (spec §9.1 + M5).

Runs concurrently with the child agent's main loop. Even when the child is
awaiting a long-running tool (e.g. 600s ``web_search``), this task keeps
publishing ``PROGRESS_UPDATE(kind=HEARTBEAT, visibility=HIDDEN)`` envelopes
so the MailboxSupervisor's stale-detection clock (spec §9.2 + §9.4) does
not classify the child as ORPHAN_TIMEOUT.

Stale judgment is supervisor-side via ``time.monotonic()`` on receipt — the
heartbeat task is a fire-and-forget publisher; consumers infer liveness from
arrival cadence.

Domain-layer constraint: only stdlib + ``app.domain.*`` imports. No
infrastructure, no Redis, no FastAPI.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Awaitable, Callable, Literal, Optional, TypeVar


logger = logging.getLogger(__name__)

from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ProgressKind,
    ProgressUpdatePayload,
    ProgressVisibility,
    MAILBOX_PUBLISH_OPERATION_TIMEOUT_SECONDS,
    SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
)


_Phase = Literal["idle", "in_tool", "finalizing"]
_T = TypeVar("_T")


class ChildHeartbeatTask:
    """Heartbeat publisher for a single mailbox-plane child session.

    Usage::

        task = ChildHeartbeatTask(publisher, parent_id, child_id)
        handle = asyncio.create_task(task.run())
        ...
        task.set_phase("in_tool", tool_call_id="tc-1")
        ...
        await task.stop()
        await handle

    ``stop()`` is idempotent. The asyncio wait pattern (``wait_for`` on the
    stop event) ensures cancel takes effect within microseconds rather than
    the full interval — important for clean shutdown on terminal transitions.
    """

    def __init__(
        self,
        publisher: MailboxPublisher,
        parent_session_id: str,
        child_session_id: str,
        *,
        interval_seconds: float = SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS,
        publish_timeout_seconds: float = MAILBOX_PUBLISH_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        if not math.isfinite(publish_timeout_seconds) or publish_timeout_seconds <= 0:
            raise ValueError("publish_timeout_seconds must be finite and positive")
        self._publisher = publisher
        self._parent = parent_session_id
        self._child = child_session_id
        self._interval = interval_seconds
        self._publish_timeout = publish_timeout_seconds
        self._stopping = asyncio.Event()
        # Serialize heartbeat and terminal publishes.  The terminal owner uses
        # ``publish_terminal`` to publish-and-close atomically: a heartbeat
        # already queued on this lock re-checks ``_stopping`` and therefore can
        # never land after the terminal envelope.
        self._publish_lock = asyncio.Lock()
        self._phase: Optional[_Phase] = "idle"
        self._tool_call_id: Optional[str] = None
        # Stable per-child correlation id so the supervisor / audit table can
        # join all heartbeat envelopes from a single child without storing
        # extra metadata. ``hb:`` prefix distinguishes from spawn/result
        # correlation streams.
        self._correlation_id = f"hb:{child_session_id}"

    def set_phase(
        self,
        phase: _Phase,
        *,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """Update the phase reported in the next heartbeat.

        Thread-safety: ChildHeartbeatTask runs in a single asyncio task; this
        setter is intended to be called from the same event loop only.
        """
        self._phase = phase
        self._tool_call_id = tool_call_id

    async def run(self) -> None:
        """Publish a heartbeat envelope every ``interval_seconds`` until
        ``stop()`` is called.

        Uses ``asyncio.wait_for(stop_event.wait(), timeout=interval)`` rather
        than a bare ``asyncio.sleep`` so the loop exits within microseconds
        on stop. The ``TimeoutError`` is caught silently — it is the normal
        "interval elapsed without stop" path.

        codex r1 [R1-5, HIGH PERF] — a single transient publisher failure
        (Redis blip, connection reset, etc.) must NOT kill the loop. If
        we propagate, the asyncio task dies, no more heartbeats reach the
        supervisor, and the child is misclassified as ORPHAN_TIMEOUT even
        though it's still alive. Catch non-cancel exceptions per
        iteration, log them, and continue waiting for the next interval;
        the supervisor's stale-detection clock will reset on the next
        successful publish. ``asyncio.CancelledError`` is the one
        exception that still propagates — caller is responsible for the
        task lifecycle.
        """
        while not self._stopping.is_set():
            try:
                await self._publish_heartbeat()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                # codex r17 [R17-5, LOW PERF] — heartbeat publishes
                # at a short interval (15s in prod); a Redis blip can
                # produce many consecutive failures. Log at warning
                # WITHOUT ``exc_info`` to keep log volume bounded;
                # the supervisor's stale-detection telemetry already
                # surfaces extended outages.
                logger.warning(
                    "heartbeat publish failed for child=%s: %s — continuing loop",
                    self._child,
                    e,
                )
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._interval
                )
            except asyncio.TimeoutError:
                # Normal: interval elapsed without stop. Continue loop.
                continue

    async def stop(self) -> None:
        """Signal the run loop to exit. Idempotent."""
        self._stopping.set()

    async def publish_terminal(
        self,
        publish: Callable[[], Awaitable[_T]],
    ) -> _T:
        """Run one terminal publish as the final serialized lifecycle write.

        Liveness remains active while a terminal publisher waits for the lock.
        Once it owns the boundary, success, failure, and cancellation all close
        the heartbeat loop before releasing the lock, so no queued heartbeat can
        appear after the terminal attempt.
        """
        async with self._publish_lock:
            try:
                return await asyncio.wait_for(
                    publish(), timeout=self._publish_timeout
                )
            finally:
                self._stopping.set()

    async def _publish_heartbeat(self) -> None:
        async with self._publish_lock:
            # A terminal publish may have closed liveness while this iteration
            # was queued on the publisher lock.
            if self._stopping.is_set():
                return
            envelope = MailboxEnvelope(
                envelope_id=str(uuid.uuid4()),
                type=MailboxEnvelopeType.PROGRESS_UPDATE,
                parent_session_id=self._parent,
                child_session_id=self._child,
                correlation_id=self._correlation_id,
                emitted_at=datetime.now(tz=timezone.utc),
                producer_role=ProducerRole.CHILD_AGENT,
                payload=ProgressUpdatePayload(
                    kind=ProgressKind.HEARTBEAT,
                    visibility=ProgressVisibility.HIDDEN,
                    phase=self._phase,
                    tool_call_id=self._tool_call_id,
                ).model_dump(mode="json"),
            )
            await asyncio.wait_for(
                self._publisher.publish(envelope),
                timeout=self._publish_timeout,
            )
