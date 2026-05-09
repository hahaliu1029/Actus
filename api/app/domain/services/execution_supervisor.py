"""Execution supervisor FSM and Redis slot accounting."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal

from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.session import Session, SessionStatus
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services._lua_scripts import (
    LUA_ADMIT,
    LUA_ADMIT_SHA,
    LUA_REVOKE,
    LUA_REVOKE_SHA,
    LUA_SWEEP_EXPIRED,
    LUA_SWEEP_EXPIRED_SHA,
    run_lua_with_fallback,
)

logger = logging.getLogger(__name__)

_HOT_TTL_SECONDS = 300
_OWNER_TTL_SECONDS = 10
_OWNER_RENEW_SECONDS = 5
_LUA_RELEASE_OWNER_IF_EQUAL = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('DEL', KEYS[1])
end
return 0
"""
_LUA_RENEW_OWNER_IF_EQUAL = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('EXPIRE', KEYS[1], tonumber(ARGV[2]))
end
return 0
"""
_LUA_DECREMENT_SUBSCRIBER_COUNT_IF_PRESENT = """
if redis.call('EXISTS', KEYS[1]) == 0 then
    return 0
end
local count = redis.call('HINCRBY', KEYS[1], 'subscriber_count', -1)
if count < 0 then
    redis.call('HSET', KEYS[1], 'subscriber_count', 0)
    count = 0
end
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
return count
"""


@dataclass(frozen=True)
class SubscriberScopeContext:
    is_conflict: bool
    current_owner: str | None


class ExecutionSupervisor:
    """Owns supervisor Redis keys and session supervisor columns."""

    def __init__(
        self,
        *,
        redis_client,
        session_repository: SessionRepository | None = None,
        uow_factory: Callable[[], IUnitOfWork] | None = None,
        meter=None,
        max_system_bg: int = 100,
        max_user_bg: int = 5,
    ) -> None:
        if session_repository is None and uow_factory is None:
            raise ValueError("session_repository or uow_factory is required")
        self._redis = (
            redis_client.client if hasattr(redis_client, "client") else redis_client
        )
        self._repo = session_repository
        self._uow_factory = uow_factory
        self._meter = meter
        self._max_system_bg = max_system_bg
        self._max_user_bg = max_user_bg
        self._sha_admit: str | None = None
        self._sha_revoke: str | None = None
        self._sha_sweep: str | None = None
        self._runners: dict[str, object] = {}
        self._init_metrics()

    async def script_load_all(self) -> None:
        self._sha_admit = await self._redis.script_load(LUA_ADMIT)
        self._sha_revoke = await self._redis.script_load(LUA_REVOKE)
        self._sha_sweep = await self._redis.script_load(LUA_SWEEP_EXPIRED)

    async def admit(
        self,
        *,
        session_id: str,
        user_id: str,
        execution_mode: Literal["foreground", "background"] | None = None,
        mode: Literal["foreground", "background"] | None = None,
        background_reason: Literal["explicit", "auto_degrade"] | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        target_mode = execution_mode or mode
        if target_mode not in ("foreground", "background"):
            raise ValueError("execution_mode must be foreground or background")

        if target_mode == "foreground":
            async with self._repo_context() as repo:
                await self._ensure_session(
                    repo,
                    session_id=session_id,
                    user_id=user_id,
                    execution_mode="foreground",
                    background_reason=None,
                    expires_at=None,
                    was_background=False,
                )
            self._meter_inc("admit", result="success")
            return

        if expires_at is None:
            raise ValueError("background admission requires expires_at")

        reason = background_reason or "explicit"
        await self._admit_background_slot(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            from_phase="running",
        )
        try:
            async with self._repo_context() as repo:
                await self._ensure_session(
                    repo,
                    session_id=session_id,
                    user_id=user_id,
                    execution_mode="background",
                    background_reason=reason,
                    expires_at=expires_at,
                    was_background=True,
                )
        except Exception:
            logger.exception("background admission PG write failed; revoking slot")
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="admit_pg_fail",
            )
            raise
        self._meter_inc("admit", result="success")

    async def promote(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
    ) -> None:
        await self._admit_background_slot(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            from_phase="foreground",
        )
        try:
            async with self._repo_context() as repo:
                await repo.update_supervisor_fields(
                    session_id,
                    execution_mode="background",
                    background_reason="auto_degrade",
                    expires_at=expires_at,
                    execution_phase="running",
                    suspended_reason=None,
                    was_background=True,
                )
        except Exception:
            logger.exception("promote PG write failed; revoking slot")
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="promote_pg_fail",
            )
            raise
        self._meter_inc("admit", result="success")
        self._meter_inc("auto_degrade")

    async def suspend_idle(self, *, session_id: str, user_id: str) -> None:
        async with self._repo_context() as repo:
            await repo.update_supervisor_fields(
                session_id,
                execution_phase="suspended",
                suspended_reason="bg_idle_timeout",
            )
        if self._cancel_registered_runner(
            session_id,
            reason="supervisor_suspend",
        ):
            await self._on_runner_session_complete(
                session_id=session_id,
                user_id=user_id,
                cancel_reason="supervisor_suspend",
            )
        self._meter_inc("idle_suspend")

    async def sweep_expired(self, *, user_id: str) -> list[str]:
        expired = await run_lua_with_fallback(
            self._redis,
            source=LUA_SWEEP_EXPIRED,
            sha=self._sha_sweep or LUA_SWEEP_EXPIRED_SHA,
            keys=[self._bg_key(user_id)],
            args=[f"{datetime.now(timezone.utc).timestamp():.6f}"],
            meter=self._meter,
        )
        return [
            item.decode() if isinstance(item, (bytes, bytearray)) else str(item)
            for item in (expired or [])
        ]

    async def list_background_slot_user_ids(self) -> list[str]:
        prefix = "supervisor:bg:"
        user_ids: set[str] = set()
        async for raw_key in self._redis.scan_iter(match=f"{prefix}*"):
            key = (
                raw_key.decode()
                if isinstance(raw_key, (bytes, bytearray))
                else str(raw_key)
            )
            if key.startswith(prefix) and len(key) > len(prefix):
                user_ids.add(key[len(prefix):])
        return sorted(user_ids)

    async def resume(
        self,
        *,
        session_id: str,
        user_id: str,
        execution_mode: Literal["foreground", "background"] | None = None,
        mode: Literal["foreground", "background"] | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        target_mode = execution_mode or mode
        if target_mode == "foreground":
            async with self._repo_context() as repo:
                await repo.update_supervisor_fields(
                    session_id,
                    execution_mode="foreground",
                    background_reason=None,
                    expires_at=None,
                    execution_phase="running",
                    suspended_reason=None,
                )
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="t7_reconnect",
            )
            return

        if target_mode != "background":
            raise ValueError("execution_mode must be foreground or background")
        if expires_at is None:
            raise ValueError("background resume requires expires_at")

        rc = await self._run_lua_admit(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
        )
        if rc == 1:
            raise SupervisorContractError(
                "R1",
                "suspended",
                "background",
                "system bg slots exhausted",
            )
        if rc == 2:
            raise SupervisorContractError(
                "R2",
                "suspended",
                "background",
                "user bg slots exhausted",
            )
        async with self._repo_context() as repo:
            await repo.update_supervisor_fields(
                session_id,
                execution_phase="running",
                suspended_reason=None,
                expires_at=expires_at,
            )

    async def terminate(
        self,
        *,
        session_id: str,
        user_id: str,
        terminal_reason: Literal[
            "natural",
            "user_cancel",
            "server_restart",
            "resume_state_lost",
            "watchdog_timeout",
        ],
        status: SessionStatus = SessionStatus.COMPLETED,
    ) -> None:
        async with self._repo_context() as repo:
            session = await repo.get_by_id(session_id)
            if (
                session is not None
                and session.status
                not in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)
                and session.execution_mode == "background"
                and session.execution_phase in ("running", "suspended")
            ):
                await repo.update_to_terminal(session_id, status, terminal_reason)
        await self._lua_revoke(
            session_id=session_id,
            user_id=user_id,
            reason=terminal_reason,
        )

    async def reconcile_running_background_at_boot(
        self,
        *,
        notification_emitter=None,
    ) -> dict[str, int]:
        finishing = 0
        suspended = 0
        async with self._repo_context() as repo:
            rows = await repo.find_running_background()
            for row in rows:
                try:
                    if row.status == SessionStatus.FINISHING:
                        await repo.update_to_terminal(
                            row.session_id,
                            SessionStatus.TIMED_OUT,
                            "server_restart",
                        )
                        await self._lua_revoke(
                            session_id=row.session_id,
                            user_id=row.user_id,
                            reason="server_restart",
                        )
                        if notification_emitter is not None:
                            await notification_emitter.emit(
                                user_id=row.user_id,
                                event_type="bg_terminal_server_restart",
                                payload={"session_id": row.session_id},
                            )
                        finishing += 1
                    else:
                        await repo.update_supervisor_fields(
                            row.session_id,
                            execution_phase="suspended",
                            suspended_reason="server_restart",
                        )
                        if notification_emitter is not None:
                            await notification_emitter.emit(
                                user_id=row.user_id,
                                event_type="bg_suspended_server_restart",
                                payload={"session_id": row.session_id},
                            )
                        suspended += 1
                except Exception:
                    logger.exception(
                        "supervisor boot reconcile failed for %s",
                        row.session_id,
                    )
        return {
            "finishing": finishing,
            "suspended": suspended,
            "total": finishing + suspended,
        }

    async def inflight_inc(
        self,
        *,
        session_id: str,
        kind: Literal["llm", "tool"],
    ) -> int:
        field = f"inflight_{kind}_count"
        try:
            value = await self._redis.hincrby(self._hot_key(session_id), field, 1)
            await self._redis.expire(self._hot_key(session_id), _HOT_TTL_SECONDS)
            return int(value)
        except Exception:
            logger.warning(
                "inflight_inc failed for %s/%s",
                session_id,
                kind,
                exc_info=True,
            )
            return 0

    async def inflight_dec(
        self,
        *,
        session_id: str,
        kind: Literal["llm", "tool"],
    ) -> int:
        field = f"inflight_{kind}_count"
        key = self._hot_key(session_id)
        try:
            value = int(await self._redis.hincrby(key, field, -1))
            if value < 0:
                self._meter_inc("inflight_negative", kind=kind)
            return value
        except Exception:
            logger.warning(
                "inflight_dec failed for %s/%s",
                session_id,
                kind,
                exc_info=True,
            )
            return 0

    async def get_inflight_counts(self, *, session_id: str) -> tuple[int, int]:
        values = await self._redis.hmget(
            self._hot_key(session_id),
            "inflight_llm_count",
            "inflight_tool_count",
        )
        return (int(values[0] or 0), int(values[1] or 0))

    @asynccontextmanager
    async def subscriber_scope(
        self,
        session_id: str,
        connection_id: str,
    ) -> AsyncIterator[SubscriberScopeContext]:
        hot_key = self._hot_key(session_id)
        owner_key = self._owner_key(session_id)
        renew_task: asyncio.Task[None] | None = None
        lease_acquired = False
        entered_count = False

        try:
            count_task = asyncio.create_task(self._enter_subscriber_count(hot_key))
            try:
                await asyncio.shield(count_task)
            except asyncio.CancelledError:
                count_task.add_done_callback(
                    lambda task: self._cleanup_subscriber_count_after_enter(
                        task,
                        hot_key=hot_key,
                    )
                )
                raise
            entered_count = True

            acquired = await self._redis.set(
                owner_key,
                connection_id,
                nx=True,
                ex=_OWNER_TTL_SECONDS,
            )
            lease_acquired = bool(acquired)
            if lease_acquired:
                renew_task = asyncio.create_task(
                    self._renew_owner_lease(
                        owner_key=owner_key,
                        connection_id=connection_id,
                    )
                )
                context = SubscriberScopeContext(
                    is_conflict=False,
                    current_owner=connection_id,
                )
            else:
                current_owner = self._decode_redis_value(
                    await self._redis.get(owner_key)
                )
                context = SubscriberScopeContext(
                    is_conflict=True,
                    current_owner=current_owner,
                )
        except BaseException:
            if entered_count:
                await self._await_or_detach_subscriber_cleanup(
                    hot_key=hot_key,
                    owner_key=owner_key,
                    connection_id=connection_id,
                    lease_acquired=lease_acquired,
                    renew_task=renew_task,
                )
            raise

        try:
            yield context
        finally:
            await self._await_or_detach_subscriber_cleanup(
                hot_key=hot_key,
                owner_key=owner_key,
                connection_id=connection_id,
                lease_acquired=lease_acquired,
                renew_task=renew_task,
            )

    async def request_cancel(
        self,
        *,
        session_id: str,
        user_id: str,
        reason: str = "user_cancel",
        stop_session: Callable[..., Awaitable[None]] | None = None,
    ) -> None:
        hot_key = self._hot_key(session_id)
        await self._redis.hset(
            hot_key,
            mapping={
                "cancellation_pending": "1",
                "pending_terminal_reason": reason,
            },
        )
        await self._redis.expire(hot_key, _HOT_TTL_SECONDS)
        if stop_session is not None:
            await stop_session(session_id=session_id, user_id=user_id)

    def _register_runner(self, session_id: str, runner: object) -> None:
        self._runners[session_id] = runner

    def _unregister_runner(self, session_id: str) -> None:
        self._runners.pop(session_id, None)

    def _cancel_registered_runner(self, session_id: str, *, reason: str) -> bool:
        runner = self._runners.get(session_id)
        if runner is None:
            return False
        cancel = getattr(runner, "cancel", None)
        if cancel is None:
            logger.warning(
                "supervisor suspend found non-cancelable runner for session=%s",
                session_id,
            )
            return False
        try:
            return bool(cancel(reason=reason))
        except Exception:
            logger.exception(
                "supervisor failed to cancel live runner for session=%s",
                session_id,
            )
            return False

    async def _on_runner_session_complete(
        self,
        *,
        session_id: str,
        user_id: str,
        cancel_reason: str | None = None,
    ) -> None:
        try:
            self._unregister_runner(session_id)
            if cancel_reason == "supervisor_suspend":
                logger.info(
                    "supervisor cleanup: skip LUA_REVOKE for supervisor_suspend session=%s",
                    session_id,
                )
                self._meter_inc("revoke", reason="supervisor_suspend_skip")
                return

            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason=cancel_reason or "natural",
            )
        except Exception:
            logger.exception(
                "supervisor cleanup hook failed for session=%s reason=%s",
                session_id,
                cancel_reason,
            )

    async def _renew_owner_lease(
        self,
        *,
        owner_key: str,
        connection_id: str,
    ) -> None:
        while True:
            await asyncio.sleep(_OWNER_RENEW_SECONDS)
            if not await self._renew_owner_if_equal(
                owner_key=owner_key,
                connection_id=connection_id,
            ):
                return

    async def _enter_subscriber_count(self, hot_key: str) -> None:
        entered_count = False
        try:
            await self._redis.hincrby(hot_key, "subscriber_count", 1)
            entered_count = True
            await self._redis.expire(hot_key, _HOT_TTL_SECONDS)
        except BaseException:
            if entered_count:
                await self._await_or_detach_subscriber_cleanup(
                    hot_key=hot_key,
                    owner_key="",
                    connection_id="",
                    lease_acquired=False,
                    renew_task=None,
                )
            raise

    def _cleanup_subscriber_count_after_enter(
        self,
        task: asyncio.Task[None],
        *,
        hot_key: str,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning(
                "subscriber scope enter count task failed for %s",
                hot_key,
                exc_info=True,
            )
            return

        cleanup_task = asyncio.create_task(
            self._cleanup_subscriber_scope(
                hot_key=hot_key,
                owner_key="",
                connection_id="",
                lease_acquired=False,
                renew_task=None,
            )
        )
        cleanup_task.add_done_callback(self._log_subscriber_cleanup_result)

    async def _await_or_detach_subscriber_cleanup(
        self,
        *,
        hot_key: str,
        owner_key: str,
        connection_id: str,
        lease_acquired: bool,
        renew_task: asyncio.Task[None] | None,
    ) -> None:
        cleanup_task = asyncio.create_task(
            self._cleanup_subscriber_scope(
                hot_key=hot_key,
                owner_key=owner_key,
                connection_id=connection_id,
                lease_acquired=lease_acquired,
                renew_task=renew_task,
            )
        )
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            cleanup_task.add_done_callback(self._log_subscriber_cleanup_result)
            raise

    async def _cleanup_subscriber_scope(
        self,
        *,
        hot_key: str,
        owner_key: str,
        connection_id: str,
        lease_acquired: bool,
        renew_task: asyncio.Task[None] | None,
    ) -> None:
        if renew_task is not None:
            renew_task.cancel()
            try:
                await renew_task
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.warning(
                    "subscriber scope owner renew task failed before cleanup",
                    exc_info=True,
                )

        try:
            await self._decrement_subscriber_count_if_present(hot_key)
        except Exception:
            logger.warning(
                "subscriber scope decrement failed for %s",
                hot_key,
                exc_info=True,
            )

        if lease_acquired:
            try:
                await self._release_owner_if_equal(
                    owner_key=owner_key,
                    connection_id=connection_id,
                )
            except Exception:
                logger.warning(
                    "subscriber scope owner release failed for %s",
                    owner_key,
                    exc_info=True,
                )

    @staticmethod
    def _log_subscriber_cleanup_result(task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.warning(
                "detached subscriber scope cleanup failed",
                exc_info=True,
            )

    async def _renew_owner_if_equal(
        self,
        *,
        owner_key: str,
        connection_id: str,
    ) -> bool:
        renewed = await self._redis.eval(
            _LUA_RENEW_OWNER_IF_EQUAL,
            1,
            owner_key,
            connection_id,
            _OWNER_TTL_SECONDS,
        )
        return int(renewed or 0) == 1

    async def _decrement_subscriber_count_if_present(self, hot_key: str) -> None:
        await self._redis.eval(
            _LUA_DECREMENT_SUBSCRIBER_COUNT_IF_PRESENT,
            1,
            hot_key,
            _HOT_TTL_SECONDS,
        )

    async def _release_owner_if_equal(
        self,
        *,
        owner_key: str,
        connection_id: str,
    ) -> None:
        await self._redis.eval(
            _LUA_RELEASE_OWNER_IF_EQUAL,
            1,
            owner_key,
            connection_id,
        )

    async def _lua_revoke(self, *, session_id: str, user_id: str, reason: str) -> int:
        rc = await run_lua_with_fallback(
            self._redis,
            source=LUA_REVOKE,
            sha=self._sha_revoke or LUA_REVOKE_SHA,
            keys=[self._system_key(), self._user_key(user_id), self._bg_key(user_id)],
            args=[session_id],
            meter=self._meter,
        )
        self._meter_inc("revoke", reason=reason)
        return int(rc)

    async def _admit_background_slot(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        from_phase: str,
    ) -> None:
        rc = await self._run_lua_admit(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
        )
        if rc == 1:
            self._meter_inc("admit", result="system_full")
            raise SupervisorContractError(
                "R1",
                from_phase,
                "background",
                "system bg slots exhausted",
            )
        if rc == 2:
            self._meter_inc("admit", result="user_full")
            raise SupervisorContractError(
                "R2",
                from_phase,
                "background",
                "user bg slots exhausted",
            )
        if rc == 3:
            self._meter_inc("admit", result="already_bg")
            raise SupervisorContractError(
                "R3",
                "background",
                "background",
                "session already in BG scope",
            )

    async def _run_lua_admit(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
    ) -> int:
        expires_at_unix = expires_at.astimezone(timezone.utc).timestamp()
        rc = await run_lua_with_fallback(
            self._redis,
            source=LUA_ADMIT,
            sha=self._sha_admit or LUA_ADMIT_SHA,
            keys=[
                self._system_key(),
                self._user_key(user_id),
                self._hot_key(session_id),
                self._bg_key(user_id),
            ],
            args=[
                session_id,
                f"{expires_at_unix:.6f}",
                str(self._max_system_bg),
                str(self._max_user_bg),
            ],
            meter=self._meter,
        )
        return int(rc)

    async def _ensure_session(
        self,
        repo: SessionRepository,
        *,
        session_id: str,
        user_id: str,
        execution_mode: Literal["foreground", "background"],
        background_reason: Literal["explicit", "auto_degrade"] | None,
        expires_at: datetime | None,
        was_background: bool,
    ) -> None:
        existing = await repo.get_by_id(session_id)
        if existing is None:
            await repo.save(
                Session(
                    id=session_id,
                    user_id=user_id,
                    status=SessionStatus.RUNNING,
                    execution_mode=execution_mode,
                    background_reason=background_reason,
                    expires_at=expires_at,
                    execution_phase="running",
                    retry_budget_remaining=3,
                    was_background=was_background,
                    last_activity_at=datetime.now(timezone.utc),
                )
            )
            return

        await repo.update_supervisor_fields(
            session_id,
            execution_mode=execution_mode,
            background_reason=background_reason,
            expires_at=expires_at,
            execution_phase="running",
            retry_budget_remaining=3,
            was_background=True if was_background else existing.was_background,
        )

    @asynccontextmanager
    async def _repo_context(self) -> AsyncIterator[SessionRepository]:
        if self._repo is not None:
            yield self._repo
            return

        if self._uow_factory is None:
            raise RuntimeError("supervisor has no repository source")
        async with self._uow_factory() as uow:
            yield uow.session

    def _init_metrics(self) -> None:
        self._counters = {}
        if self._meter is None:
            return
        try:
            for name, metric in {
                "admit": "actus_supervisor_admit_total",
                "revoke": "actus_supervisor_revoke_total",
                "idle_suspend": "actus_supervisor_idle_suspend_total",
                "auto_degrade": "actus_supervisor_auto_degrade_total",
                "inflight_negative": "actus_supervisor_inflight_negative_total",
            }.items():
                self._counters[name] = self._meter.create_counter(metric)
        except Exception:
            logger.warning("supervisor metric init failed; metrics disabled", exc_info=True)
            self._counters = {}

    def _meter_inc(self, name: str, /, **labels: str) -> None:
        counter = getattr(self, "_counters", {}).get(name)
        if counter is None:
            return
        try:
            counter.add(1, attributes=labels) if labels else counter.add(1)
        except Exception:
            logger.debug("supervisor metric increment failed: %s", name, exc_info=True)

    @staticmethod
    def _system_key() -> str:
        return "supervisor:system:bg_count"

    @staticmethod
    def _user_key(user_id: str) -> str:
        return f"supervisor:user:{user_id}"

    @staticmethod
    def _hot_key(session_id: str) -> str:
        return f"supervisor:hot:{session_id}"

    @staticmethod
    def _owner_key(session_id: str) -> str:
        return f"supervisor:owner:{session_id}"

    @staticmethod
    def _bg_key(user_id: str) -> str:
        return f"supervisor:bg:{user_id}"

    @staticmethod
    def _decode_redis_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return value.decode()
        return str(value)
