"""Concrete ``MemoryNotificationEmitter`` bound to a SQLAlchemy session
factory. Used by the LLM quality gate to surface ``memory_gate_paused``
/ ``quota_exceeded`` events to the user's notification tray.
"""
from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Callable, Literal, get_args

from app.infrastructure.repositories.db_memory_system_notification_repository import (
    DBMemorySystemNotificationRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

logger = logging.getLogger(__name__)


B3CoreNotificationEventType = Literal[
    "bg_completed",
    "bg_cancelled",
    "bg_failed_resume",
    "bg_failed_watchdog",
    "bg_terminal_server_restart",
    "bg_suspended_timeout",
    "bg_suspended_server_restart",
    "bg_retry_exhausted",
]

M1NotificationEventType = Literal[
    "memory_gate_paused",
    "quota_exceeded",
    "fs_permanent_failure",
]

_B3_CORE_EVENT_TYPES: frozenset[str] = frozenset(
    get_args(B3CoreNotificationEventType)
)
_M1_EVENT_TYPES: frozenset[str] = frozenset(get_args(M1NotificationEventType))
ALL_VALID_EVENT_TYPES: frozenset[str] = (
    _M1_EVENT_TYPES | _B3_CORE_EVENT_TYPES
)


class DBMemoryNotificationEmitter:
    """Persists notifications via a short-lived session; errors swallowed."""

    def __init__(
        self,
        session_factory: "async_sessionmaker[AsyncSession]",
        *,
        repo_factory: Callable[
            ["AsyncSession"], DBMemorySystemNotificationRepository
        ] = DBMemorySystemNotificationRepository,
    ) -> None:
        self._session_factory = session_factory
        self._repo_factory = repo_factory

    async def emit(
        self,
        *,
        user_id: str,
        event_type: str,
        payload: dict,
    ) -> None:
        if event_type not in ALL_VALID_EVENT_TYPES:
            logger.warning(
                "memory notification emit: unknown event_type=%s dropping",
                event_type,
            )
            return

        # Swallow errors — notification is advisory. The flush path can't
        # afford to fail because we couldn't tell the user we degraded.
        try:
            async with self._session_factory() as session:
                repo = self._repo_factory(session)
                await repo.create(
                    notification_id=str(uuid.uuid4()),
                    user_id=user_id,
                    event_type=event_type,
                    payload=payload,
                )
                await session.commit()
        except Exception as exc:
            logger.warning(
                "memory notification emit failed: user=%s event=%s err=%s",
                user_id, event_type, exc,
            )
