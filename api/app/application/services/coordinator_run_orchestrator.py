"""C2 PR-3 §11.1 + §11.5 — CoordinatorRunOrchestrator skeleton (parent-cancel only).

PR-3 minimal scope:
  - parent-cancel path: wait for ``cancel_event``; on set publish CANCEL_REQUEST
    × all pending work units.
  - timeout: log + return (orchestrator does not raise — child runner finalizers
    own outcome resolution).

PR-6 fleshes out:
  - sibling cancel predicate
  - RESULT_READY observation
  - cost aggregation
  - quota enforcement
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.domain.external.mailbox_publisher import MailboxPublisher

logger = logging.getLogger(__name__)


class CoordinatorRunOrchestrator:
    """Skeleton — PR-6 adds sibling-cancel predicate + cost aggregation."""

    def __init__(
        self,
        *,
        publisher: MailboxPublisher,
        parent_session_id: str,
        coordinator_run_id: str,
        envelope_factory: Optional[CoordinatorEnvelopeFactory] = None,
    ) -> None:
        """``publisher``: live MailboxPublisher port (single-arg ``publish``).

        r3 P1-2: ``parent_session_id`` and ``coordinator_run_id`` are REQUIRED
        (no default empty string). Live publisher derives the Redis stream key
        from ``envelope.parent_session_id`` (see
        ``RedisMailboxPublisher.publish``); an empty string would publish
        CANCEL_REQUEST to ``actus:child::mailbox`` and never reach the child.
        """
        if not parent_session_id:
            raise ValueError(
                "CoordinatorRunOrchestrator: parent_session_id must be non-empty"
            )
        if not coordinator_run_id:
            raise ValueError(
                "CoordinatorRunOrchestrator: coordinator_run_id must be non-empty"
            )
        self._publisher = publisher
        self._published: set[str] = set()
        self._envelope_factory = envelope_factory or CoordinatorEnvelopeFactory()
        self._parent_session_id = parent_session_id
        self._coordinator_run_id = coordinator_run_id

    async def run(
        self,
        *,
        coordinator_run_id: str,
        root_session_id: str,
        work_units_pending: list[str],
        child_session_ids: dict[str, str],
        cancel_event: asyncio.Event,
        timeout_seconds: float = 600.0,
    ) -> None:
        """PR-3 parent-cancel loop.

        Wait for ``cancel_event``; on set, publish CANCEL_REQUEST × all pending.
        Timeout → log + return (orchestrator finishes; child finalizers report
        outcome). Per-envelope publish failure is logged + swallowed so a single
        broken child does not stop CANCEL fan-out to the rest.
        """
        try:
            await asyncio.wait_for(cancel_event.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.info(
                "CoordinatorRunOrchestrator: timeout (no cancel) for %s",
                coordinator_run_id,
            )
            return

        # r5 P1-2: track attempted vs succeeded. Per-envelope failure is
        # logged + swallowed so a single broken child doesn't stop CANCEL
        # fan-out to the rest, BUT all-failed must raise so the caller
        # (orchestrator_task done-callback in PR-6) sees the failure instead
        # of silently losing the parent cancel.
        attempted = 0
        succeeded = 0
        last_exc: Optional[BaseException] = None
        for wu_id in work_units_pending:
            child_sid = child_session_ids.get(wu_id)
            if child_sid is None or wu_id in self._published:
                continue
            attempted += 1
            try:
                envelope = self._envelope_factory.make_cancel_request(
                    parent_session_id=self._parent_session_id,
                    child_session_id=child_sid,
                    correlation_id=coordinator_run_id,
                    reason="parent_cancel",
                )
                await self._publisher.publish(envelope)
                self._published.add(wu_id)
                succeeded += 1
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "CoordinatorRunOrchestrator: cancel publish failed wu=%s child=%s: %s",
                    wu_id, child_sid, exc,
                )
        if attempted > 0 and succeeded == 0:
            raise RuntimeError(
                f"CoordinatorRunOrchestrator: all {attempted} CANCEL_REQUEST "
                f"publishes failed for run {coordinator_run_id}; parent cancel lost. "
                f"Last error: {last_exc!r}"
            )

    async def shutdown(self, timeout: float = 5.0) -> None:
        """PR-6 will manage background tasks here; PR-3 is a no-op."""
        return None
