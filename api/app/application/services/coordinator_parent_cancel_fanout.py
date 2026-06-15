"""C2 coordinator-cancel — parent-stop child cancellation fanout (spec §3.2).

When a user stops a parent coordinator session, this service enumerates the
parent's still-RUNNING coordinator children and fans out a ``CANCEL_REQUEST``
to each (plus a same-pod ``request_stop_started`` fast-path), reusing the tested
``CANCEL_REQUEST -> child-listener -> request_stop -> CANCEL_ACK`` path so children
terminalize immediately instead of waiting for the <=300s watchdog.

INV-C2: ``cancel_children`` NEVER raises — enumeration failure and per-child
publish failure are both contained so the parent always reaches its own
terminalization. INV-C4: null deps -> no-op.

Application-layer pure orchestration — no FastAPI / SQLAlchemy import.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CancelFanoutResult:
    """Outcome counters for one parent-cancel fanout (returned + logged)."""

    enumerated: int
    published: int
    failed: int


class CoordinatorParentCancelFanout:
    """Out-of-band CANCEL_REQUEST fanout from the user-stop path.

    Deps are duck-typed (Protocol-free) so the legacy/test path can pass
    ``None`` for any of them; ``cancel_children`` short-circuits to a no-op
    when the correctness-critical trio (repository / factory / publisher) is
    absent (INV-C4). The optional ``child_runner_starter`` only adds a same-pod
    fast-path.
    """

    def __init__(
        self,
        *,
        session_repository: Any = None,
        envelope_factory: Any = None,
        mailbox_publisher: Any = None,
        child_runner_starter: Any = None,
    ) -> None:
        self._session_repository = session_repository
        self._envelope_factory = envelope_factory
        self._mailbox_publisher = mailbox_publisher
        self._child_runner_starter = child_runner_starter

    async def cancel_children(
        self, *, parent_session_id: str, reason: str = "parent_cancel"
    ) -> CancelFanoutResult:
        if (
            self._session_repository is None
            or self._envelope_factory is None
            or self._mailbox_publisher is None
        ):
            return CancelFanoutResult(0, 0, 0)  # null deps -> no-op (INV-C4)
        try:
            children = (
                await self._session_repository.find_running_mailbox_children_for_parent(
                    parent_session_id
                )
            )
        except Exception:  # enumeration failure -> never raise (INV-C2)
            logger.warning(
                "parent-cancel enumeration failed parent=%s",
                parent_session_id,
                exc_info=True,
            )
            return CancelFanoutResult(0, 0, 0)
        if not children:
            return CancelFanoutResult(0, 0, 0)
        # Same-pod fast-path FIRST: stop the in-process runners synchronously so
        # a child cancels even before its listener consumes the envelope. A
        # cross-pod child is a no-op here and is covered by the envelope below.
        if self._child_runner_starter is not None:
            try:
                self._child_runner_starter.request_stop_started(
                    [c.session_id for c in children]
                )
            except Exception:
                logger.warning(
                    "parent-cancel fast-path failed parent=%s",
                    parent_session_id,
                    exc_info=True,
                )
        published = 0
        failed = 0
        for c in children:  # cross-pod CANCEL_REQUEST (the correctness source)
            try:
                env = self._envelope_factory.make_cancel_request(
                    parent_session_id=parent_session_id,
                    child_session_id=c.session_id,
                    correlation_id=(c.coordinator_run_id or parent_session_id),
                    reason=reason,
                )
                await self._mailbox_publisher.publish(env)
                published += 1
            except Exception:
                failed += 1
                logger.warning(
                    "parent-cancel publish failed parent=%s child=%s",
                    parent_session_id,
                    c.session_id,
                    exc_info=True,
                )
        logger.info(
            "parent-cancel fanout parent=%s enumerated=%d published=%d failed=%d",
            parent_session_id,
            len(children),
            published,
            failed,
        )
        return CancelFanoutResult(len(children), published, failed)
