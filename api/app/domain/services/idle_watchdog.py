"""Supervisor hot-hash activity tracking."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Callable

from app.domain.models.session import SessionStatus
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork

_HOT_TTL_SECONDS = 300
_DEFAULT_IDLE_TIMEOUT_SECONDS = 180.0
_DEFAULT_SCAN_INTERVAL_SECONDS = 60.0

logger = logging.getLogger(__name__)


class IdleWatchdog:
    """Refreshes supervisor hot-hash activity state."""

    def __init__(
        self,
        *,
        redis_client,
        supervisor=None,
        session_repository: SessionRepository | None = None,
        uow_factory: Callable[[], IUnitOfWork] | None = None,
        notification_emitter=None,
        idle_timeout_seconds: float = _DEFAULT_IDLE_TIMEOUT_SECONDS,
        scan_interval_seconds: float = _DEFAULT_SCAN_INTERVAL_SECONDS,
    ) -> None:
        self._redis = (
            redis_client.client if hasattr(redis_client, "client") else redis_client
        )
        self._supervisor = supervisor
        self._repo = session_repository
        self._uow_factory = uow_factory
        self._notification_emitter = notification_emitter
        self._idle_timeout_seconds = idle_timeout_seconds
        self._scan_interval_seconds = scan_interval_seconds
        self._task: asyncio.Task[None] | None = None

    async def touch_activity(self, *, session_id: str) -> None:
        key = f"supervisor:hot:{session_id}"
        await self._redis.hset(
            key,
            mapping={
                "last_activity_at": f"{datetime.now(timezone.utc).timestamp():.6f}",
            },
        )
        await self._redis.expire(key, _HOT_TTL_SECONDS)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(
            self._run_loop(),
            name="supervisor-idle-watchdog",
        )

    async def stop(self) -> None:
        if self._task is None or self._task.done():
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run_loop(self) -> None:
        while True:
            try:
                await self._scan_once()
                await self._sweep_expired_once()
            except Exception:
                logger.exception("idle watchdog scan failed")
            await asyncio.sleep(self._scan_interval_seconds)

    async def _scan_once(self) -> None:
        if self._supervisor is None:
            return
        now = datetime.now(timezone.utc).timestamp()
        async with self._repo_context() as repo:
            rows = await repo.find_running_background()
        for row in rows:
            if row.status != SessionStatus.RUNNING:
                continue
            raw_last_activity = await self._redis.hget(
                f"supervisor:hot:{row.session_id}",
                "last_activity_at",
            )
            if raw_last_activity is None:
                continue
            try:
                last_activity = float(raw_last_activity)
            except (TypeError, ValueError):
                logger.debug(
                    "invalid supervisor hot last_activity_at for %s: %r",
                    row.session_id,
                    raw_last_activity,
                )
                continue
            if now - last_activity <= self._idle_timeout_seconds:
                continue
            if await self._has_inflight_work(row.session_id):
                continue
            await self._supervisor.suspend_idle(
                session_id=row.session_id,
                user_id=row.user_id,
            )

    async def _has_inflight_work(self, session_id: str) -> bool:
        try:
            llm_count, tool_count = await self._supervisor.get_inflight_counts(
                session_id=session_id
            )
        except Exception:
            logger.warning(
                "idle watchdog failed to read inflight counts for %s",
                session_id,
                exc_info=True,
            )
            return True
        return max(llm_count, 0) + max(tool_count, 0) > 0

    async def _sweep_expired_once(self) -> None:
        if self._supervisor is None:
            return
        async with self._repo_context() as repo:
            user_ids = set(await repo.distinct_user_ids_with_running_bg())
        user_ids.update(await self._supervisor.list_background_slot_user_ids())
        for user_id in user_ids:
            expired_session_ids = await self._supervisor.sweep_expired(user_id=user_id)
            for session_id in expired_session_ids:
                try:
                    await self._supervisor.terminate(
                        session_id=session_id,
                        user_id=user_id,
                        terminal_reason="watchdog_timeout",
                        status=SessionStatus.TIMED_OUT,
                        notification_emitter=self._notification_emitter,
                    )
                except Exception:
                    logger.exception(
                        "expired background terminate failed for %s",
                        session_id,
                    )

    @asynccontextmanager
    async def _repo_context(self) -> AsyncIterator[SessionRepository]:
        if self._repo is not None:
            yield self._repo
            return
        if self._uow_factory is None:
            raise RuntimeError("idle watchdog has no repository source")
        async with self._uow_factory() as uow:
            yield uow.session
