"""Execution supervisor FSM and Redis slot accounting."""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Awaitable, Callable, Literal

from app.domain.errors.supervisor import SupervisorContractError
from app.domain.external.supervisor_registry import SupervisorRegistryPort
from app.domain.models.event import ExecutionStatePayload, PendingExecutionEvent
from app.domain.models.session import Session, SessionStatus
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.uow import IUnitOfWork
from app.domain.services.session.session_state_machine import SessionStateMachine
from app.domain.services._lua_scripts import (
    LUA_ADMIT,
    LUA_ADMIT_SHA,
    LUA_GC_BACKGROUND_RECONCILE_MARKER,
    LUA_GC_BACKGROUND_RECONCILE_MARKER_SHA,
    LUA_MARK_BACKGROUND_RECONCILE_HELD,
    LUA_MARK_BACKGROUND_RECONCILE_HELD_SHA,
    LUA_RELEASE_HELD_BACKGROUND_RECONCILE,
    LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SHA,
    LUA_REVOKE,
    LUA_REVOKE_SHA,
    LUA_SWEEP_EXPIRED,
    LUA_SWEEP_EXPIRED_SHA,
    LUA_RESTORE_BACKGROUND_FROM_MARKER,
    LUA_RESTORE_BACKGROUND_FROM_MARKER_SHA,
    LUA_SYNC_BACKGROUND_EXPIRY,
    LUA_SYNC_BACKGROUND_EXPIRY_SHA,
    run_lua_with_fallback,
)

logger = logging.getLogger(__name__)


async def _commit_uow_if_real(uow) -> None:
    """C3 PR-3c (codex r11 [HIGH CONTRACT] fix) — same contract as
    ``app.application.services.agent_service._commit_uow_if_real``.

    Explicit commit so DBUnitOfWork's CancelledError-swallowing
    ``__aexit__`` (db_uow.py:71) cannot let a terminal write appear
    durable when it isn't, before this module's mailbox-supervisor stop
    side-effect runs. Test stubs without ``db_session`` get a no-op.
    """
    db_session = getattr(uow, "db_session", None)
    if db_session is None:
        return
    commit = getattr(db_session, "commit", None)
    if commit is None:
        return
    await commit()


# C3 PR-3c (codex r9 [HIGH CONTRACT] fix) — GC anchor + observability for
# shielded mailbox-stop tasks. Same pattern as runner's
# ``_PENDING_TERMINAL_TASKS`` in ``agent_task_runner.py:119`` and
# AgentService's ``_PENDING_MAILBOX_STOP_TASKS``: a strong reference
# prevents GC, and the done callback surfaces any exception.
_PENDING_MAILBOX_STOP_TASKS: set[asyncio.Task] = set()


def _on_mailbox_stop_task_done(task: asyncio.Task) -> None:
    _PENDING_MAILBOX_STOP_TASKS.discard(task)
    if task.cancelled():
        logger.warning(
            "mailbox stop task %s was cancelled unexpectedly", task.get_name()
        )
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "mailbox stop task %s raised: %s",
            task.get_name(),
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

_BACKGROUND_RETRY_BUDGET = 3
_BG_SLOT_CLEANUP_GRACE_SECONDS = 86400
_BG_RECONCILE_SETTLE_SECONDS = 60
_BG_RECONCILE_RETENTION_SECONDS = 86400
_HOT_TTL_SECONDS = 300
AUTO_DEGRADE_CLEANUP_WINDOW = timedelta(hours=2)
AUTO_DEGRADE_RENEW_WINDOW = timedelta(minutes=30)
_OWNER_TTL_SECONDS = 10
_OWNER_RENEW_SECONDS = 5
_MODE_TRANSITION_FENCE_TTL_SECONDS = 15
_MODE_TRANSITION_FENCE_RENEW_SECONDS = 5
_MODE_TRANSITION_FENCE_WAIT_SECONDS = 5
_MODE_TRANSITION_FENCE_RETRY_SECONDS = 0.01
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

_PENDING_MODE_FENCE_CLEANUPS: set[asyncio.Task[None]] = set()
_MODE_FENCE_SHUTDOWN_WAIT_SECONDS = 5.0


async def drain_mode_transition_fence_cleanups() -> None:
    """Finish Redis lease releases before the Redis client is closed."""
    tasks = list(_PENDING_MODE_FENCE_CLEANUPS)
    if not tasks:
        return
    _, pending = await asyncio.wait(
        tasks,
        timeout=_MODE_FENCE_SHUTDOWN_WAIT_SECONDS,
    )
    for task in pending:
        task.cancel()
    # Consume every task in the original snapshot. A task may finish with an
    # exception while ``asyncio.wait`` is returning and otherwise produce an
    # un-retrieved-task warning during loop teardown.
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for task, result in zip(tasks, results, strict=True):
        if isinstance(result, BaseException) and not isinstance(
            result, asyncio.CancelledError
        ):
            logger.warning(
                "mode transition fence cleanup failed during shutdown: task=%s",
                task.get_name(),
                exc_info=(type(result), result, result.__traceback__),
            )


class ModeTransitionFenceLostError(RuntimeError):
    """The owner must stop because its cross-pod transition lease was lost."""


@dataclass(frozen=True)
class _BackgroundReconcileMarker:
    phase: Literal["held", "released"]
    generation: int
    next_check_at: datetime
    gc_at: datetime
    token: str
    require_generation: bool
    user_id: str | None


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
        supervisor_registry: SupervisorRegistryPort | None = None,
        session_state_machine: SessionStateMachine | None = None,
        utcnow: Callable[[], datetime] | None = None,
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
        self._sha_gc_background_reconcile_marker: str | None = None
        self._sha_mark_background_reconcile_held: str | None = None
        self._sha_release_held_background_reconcile: str | None = None
        self._sha_revoke: str | None = None
        self._sha_sweep: str | None = None
        self._sha_restore_background_from_marker: str | None = None
        self._sha_sync_background_expiry: str | None = None
        self._runners: dict[str, object] = {}
        self._auto_degrade_redis_sync_pending: set[str] = set()
        # C3 PR-3c (codex r6 [HIGH CONTRACT] fix) — ExecutionSupervisor owns
        # two non-runner terminal-write paths (`terminate` for
        # idle_watchdog + admin cancel, and `reconcile_running_background_at_boot`
        # for FINISHING cleanup at pod start). Both used to bypass the
        # MailboxSupervisor stop hook → if mailbox plane was enabled, the
        # supervisor task would outlive its root session. Injecting the
        # SupervisorRegistry port here (domain Protocol → application impl
        # via duck typing) keeps Clean Architecture clean and gives both
        # paths a one-line stop call below. Default None preserves
        # backwards compat for tests / pre-mailbox deployments.
        self._supervisor_registry = supervisor_registry
        self._session_state_machine = session_state_machine
        self._utcnow = utcnow or (lambda: datetime.now(timezone.utc))
        self._init_metrics()

    def _require_state_machine(self) -> SessionStateMachine:
        # A4-1 §6: production wiring (service_dependencies.py) always injects an
        # SSM. Optional ctor param + this guard => a missing production injection
        # fails loud at the first terminal write, not as a None AttributeError.
        ssm = self._session_state_machine
        if ssm is None:
            raise RuntimeError(
                "ExecutionSupervisor status write requires a SessionStateMachine "
                "but none was injected (INV-4: SSM is the sole status writer)"
            )
        return ssm

    async def _maybe_stop_supervisor_for_session(self, session_id: str) -> None:
        """C3 PR-3c (codex r6) — stop the per-pod MailboxSupervisor on
        non-runner terminal writes owned by ExecutionSupervisor.

        Mirrors ``AgentService._maybe_stop_supervisor_for_session``: safe
        on missing registry, safe on unknown session id (the registry's
        ``stop`` is a no-op on roots it doesn't track), swallows
        exceptions so a transient registry hiccup cannot fail a terminal
        write that already committed.
        """
        registry = self._supervisor_registry
        if registry is None:
            return
        # C3 PR-3c (codex r7→r12) — fire-and-forget the stop. See
        # AgentService._maybe_stop_supervisor_for_session for the full
        # rationale: awaiting the stop introduces a cancellation seam
        # that can skip downstream caller cleanup (lua_revoke, control
        # events). Spawn + anchor + done callback gives us GC safety +
        # exception observability without blocking the caller.
        stop_task = asyncio.create_task(
            registry.stop(session_id),
            name=f"mailbox-stop-{session_id}",
        )
        _PENDING_MAILBOX_STOP_TASKS.add(stop_task)
        stop_task.add_done_callback(_on_mailbox_stop_task_done)

    async def script_load_all(self) -> None:
        self._sha_admit = await self._redis.script_load(LUA_ADMIT)
        self._sha_gc_background_reconcile_marker = await self._redis.script_load(
            LUA_GC_BACKGROUND_RECONCILE_MARKER
        )
        self._sha_mark_background_reconcile_held = await self._redis.script_load(
            LUA_MARK_BACKGROUND_RECONCILE_HELD
        )
        self._sha_release_held_background_reconcile = (
            await self._redis.script_load(LUA_RELEASE_HELD_BACKGROUND_RECONCILE)
        )
        self._sha_revoke = await self._redis.script_load(LUA_REVOKE)
        self._sha_sweep = await self._redis.script_load(LUA_SWEEP_EXPIRED)
        self._sha_restore_background_from_marker = await self._redis.script_load(
            LUA_RESTORE_BACKGROUND_FROM_MARKER
        )
        self._sha_sync_background_expiry = await self._redis.script_load(
            LUA_SYNC_BACKGROUND_EXPIRY
        )

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
        async with self.mode_transition_fence(session_id=session_id):
            reservation: tuple[str, int] | None = None
            try:
                async with self._transition_repo_context() as (repo, uow):
                    current = await repo.get_by_id(session_id)
                    if current is None:
                        await self._ensure_session(
                            repo,
                            session_id=session_id,
                            user_id=user_id,
                            execution_mode="foreground",
                            background_reason=None,
                            expires_at=None,
                            was_background=False,
                        )
                        current = await repo.get_by_id(session_id)
                    if current is None:
                        raise RuntimeError(
                            "background admission session was not persisted"
                        )
                    if (
                        current.status != SessionStatus.RUNNING
                        or current.execution_mode != "foreground"
                        or current.execution_phase
                        not in ("running", "recovering", "idle")
                    ):
                        raise SupervisorContractError(
                            "R3",
                            current.execution_phase,
                            "background",
                            "session is not foreground",
                        )
                    expected_revision = current.execution_revision
                    generation = expected_revision + 1
                    await self._admit_background_slot(
                        session_id=session_id,
                        user_id=user_id,
                        expires_at=expires_at,
                        from_phase="running",
                        generation=generation,
                    )
                    reservation = (user_id, generation)
                    pending = self._pending_execution_event(
                        execution_revision=generation,
                        execution_mode="background",
                        execution_phase="running",
                        transition_reason="explicit_background_admission",
                        background_reason=reason,
                        expires_at=expires_at,
                        retry_budget_remaining=_BACKGROUND_RETRY_BUDGET,
                    )
                    transitioned = await repo.promote_foreground_to_background(
                        session_id,
                        expires_at=expires_at,
                        retry_budget_remaining=_BACKGROUND_RETRY_BUDGET,
                        expected_execution_revision=expected_revision,
                        background_reason=reason,
                        pending_event=pending,
                    )
                    if transitioned is not None and uow is not None:
                        await _commit_uow_if_real(uow)
            except BaseException:
                if reservation is not None:
                    logger.exception(
                        "background admission PG write/commit failed; revoking slot"
                    )
                    await self._cleanup_failed_background_admission(
                        session_id=session_id,
                        user_id=reservation[0],
                        reason="admit_pg_fail",
                        generation=reservation[1],
                    )
                raise
            if transitioned is None:
                await self._cleanup_failed_background_admission(
                    session_id=session_id,
                    user_id=user_id,
                    reason="admit_stale",
                    generation=generation,
                )
                raise SupervisorContractError(
                    "R3", "background", "background", "session already in BG scope"
                )
            await self._ensure_authoritative_background_projection(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                generation=generation,
            )
        self._meter_inc("admit", result="success")

    async def promote(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
    ) -> int | None:
        current = await self._read_session_snapshot(session_id)
        if current is None:
            return None
        if (
            current.status != SessionStatus.RUNNING
            or current.execution_mode != "foreground"
            or current.execution_phase not in ("running", "recovering", "idle")
        ):
            return None
        expected_revision = current.execution_revision
        generation = expected_revision + 1
        await self._admit_background_slot(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            from_phase="foreground",
            generation=generation,
        )
        pending = self._pending_execution_event(
            execution_revision=generation,
            execution_mode="background",
            execution_phase="running",
            transition_reason="auto_degrade_sse_disconnect",
            background_reason="auto_degrade",
            expires_at=expires_at,
            retry_budget_remaining=_BACKGROUND_RETRY_BUDGET,
        )
        try:
            async with self._transition_repo_context() as (repo, uow):
                promoted = await repo.promote_foreground_to_background(
                    session_id,
                    expires_at=expires_at,
                    retry_budget_remaining=_BACKGROUND_RETRY_BUDGET,
                    expected_execution_revision=expected_revision,
                    background_reason="auto_degrade",
                    pending_event=pending,
                )
                if promoted is not None and uow is not None:
                    await _commit_uow_if_real(uow)
        except BaseException:
            logger.exception("promote PG write/commit failed; revoking slot")
            await self._cleanup_failed_background_admission(
                session_id=session_id,
                user_id=user_id,
                reason="promote_pg_fail",
                generation=generation,
            )
            raise
        if not promoted:
            await self._cleanup_failed_background_admission(
                session_id=session_id,
                user_id=user_id,
                reason="promote_stale",
                generation=generation,
            )
            return None
        await self._ensure_authoritative_background_projection(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            generation=generation,
        )
        self._meter_inc("admit", result="success")
        self._meter_inc("auto_degrade")
        return _BACKGROUND_RETRY_BUDGET

    def new_auto_degrade_cleanup_expiry(self) -> datetime:
        """Return the end of the next crash-cleanup window.

        The two-hour value is deliberately a rolling lease window, not a task
        wall-clock deadline.
        """
        return self._now_utc() + AUTO_DEGRADE_CLEANUP_WINDOW

    async def renew_auto_degrade_expiry_if_running(
        self,
        *,
        session_id: str,
    ) -> bool:
        """Renew one live auto-degrade cleanup lease when its window is due.

        PostgreSQL is the authority. Redis is updated only after the
        conditional PG write has completed successfully. If PG succeeded but
        a Redis write failed, a later tick observes the durable PG expiry and
        retries the Redis synchronization instead of being hidden by the
        renewal throttle.
        """
        now = self._now_utc()
        if self._uow_factory is not None:
            async with self._uow_factory() as uow:
                user_id, expires_at, execution_revision, pg_renewed = (
                    await self._prepare_auto_degrade_renewal(
                        uow.session,
                        session_id=session_id,
                        now=now,
                    )
                )
                if pg_renewed:
                    # Redis must never move ahead of a PG value that only
                    # exists in an uncommitted transaction.
                    await _commit_uow_if_real(uow)
        else:
            if self._repo is None:
                raise RuntimeError("supervisor has no repository source")
            user_id, expires_at, execution_revision, pg_renewed = (
                await self._prepare_auto_degrade_renewal(
                    self._repo,
                    session_id=session_id,
                    now=now,
                )
            )

        if expires_at is None or user_id is None or execution_revision is None:
            return False
        if (
            not pg_renewed
            and session_id not in self._auto_degrade_redis_sync_pending
            and await self._redis_expiry_matches(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                execution_revision=execution_revision,
            )
        ):
            return False
        try:
            await self._ensure_authoritative_background_projection(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                generation=execution_revision,
            )
        except BaseException:
            self._auto_degrade_redis_sync_pending.add(session_id)
            raise
        self._auto_degrade_redis_sync_pending.discard(session_id)
        current = await self._read_session_snapshot(session_id)
        if not self._is_running_auto_degrade(current):
            # A reconnect may commit foreground after our PG renewal but before
            # this projection. Remove the stale slot we may just have written.
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="renew_post_projection_stale",
                expected_generation=execution_revision,
                allow_legacy=True,
            )
            return False
        if current.execution_revision != execution_revision:
            current_expiry = self._as_utc(current.expires_at)
            if current_expiry is not None:
                await self._ensure_authoritative_background_projection(
                    session_id=session_id,
                    user_id=user_id,
                    expires_at=current_expiry,
                    generation=current.execution_revision,
                )
            return False
        return True

    async def _read_session_snapshot(self, session_id: str) -> Session | None:
        async with self._repo_context() as repo:
            return await repo.get_by_id(session_id)

    async def clear_pending_execution_event(
        self,
        *,
        session_id: str,
        execution_revision: int,
    ) -> bool:
        async with self._repo_context() as repo:
            return await repo.clear_pending_execution_event(
                session_id,
                execution_revision=execution_revision,
            )

    async def _required_execution_revision(self, session_id: str) -> int:
        current = await self._read_session_snapshot(session_id)
        if current is None:
            raise ValueError(f"session {session_id} does not exist")
        return current.execution_revision

    async def _prepare_auto_degrade_renewal(
        self,
        repo: SessionRepository,
        *,
        session_id: str,
        now: datetime,
    ) -> tuple[str | None, datetime | None, int | None, bool]:
        session = await repo.get_by_id(session_id)
        if not self._is_running_auto_degrade(session):
            return (None, None, None, False)
        user_id = str(session.user_id) if session.user_id is not None else None
        if not user_id:
            logger.warning(
                "auto-degrade lease renewal skipped session without owner: %s",
                session_id,
            )
            return (None, None, None, False)

        current_expiry = self._as_utc(session.expires_at)
        if (
            current_expiry is not None
            and current_expiry > now + AUTO_DEGRADE_RENEW_WINDOW
        ):
            return (user_id, current_expiry, session.execution_revision, False)

        expires_at = now + AUTO_DEGRADE_CLEANUP_WINDOW
        renewed = await repo.renew_auto_degrade_expiry_if_running(
            session_id,
            expires_at=expires_at,
        )
        if renewed is None:
            return (None, None, None, False)
        authoritative_expiry, execution_revision = renewed
        return (user_id, authoritative_expiry, execution_revision, True)

    async def can_auto_degrade_after_disconnect(self, *, session_id: str) -> bool:
        """Fail closed unless both owner and subscriber count are absent."""
        try:
            owner = self._decode_redis_value(
                await self._redis.get(self._owner_key(session_id))
            )
            raw_count = await self._redis.hget(
                self._hot_key(session_id),
                "subscriber_count",
            )
            count = int(self._decode_redis_value(raw_count) or "0")
        except Exception:
            logger.warning(
                "auto-degrade subscriber precheck failed for %s",
                session_id,
                exc_info=True,
            )
            return False
        return owner is None and count <= 0

    @asynccontextmanager
    async def mode_transition_fence(
        self,
        *,
        session_id: str,
    ) -> AsyncIterator[None]:
        """Short cross-pod fence for one mode transition plus its event emit."""
        key = f"supervisor:mode-transition:{session_id}"
        token = uuid.uuid4().hex
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _MODE_TRANSITION_FENCE_WAIT_SECONDS
        while not await self._redis.set(
            key,
            token,
            nx=True,
            ex=_MODE_TRANSITION_FENCE_TTL_SECONDS,
        ):
            if loop.time() >= deadline:
                raise TimeoutError(f"mode transition fence busy for {session_id}")
            await asyncio.sleep(_MODE_TRANSITION_FENCE_RETRY_SECONDS)

        owner_task = asyncio.current_task()
        if owner_task is None:
            raise RuntimeError("mode transition fence requires an asyncio task")
        owner_cancel_marker = f"mode-transition-fence-lost:{token}"
        lease_lost_errors: list[BaseException] = []
        body_active = True
        renew_task = asyncio.create_task(
            self._renew_mode_transition_fence(key=key, token=token),
            name=f"mode-transition-fence-renew-{session_id}",
        )

        def _stop_owner_on_lease_lost(task: asyncio.Task[None]) -> None:
            if task.cancelled():
                return
            error = task.exception()
            if error is None:
                error = ModeTransitionFenceLostError(
                    f"mode transition fence renewal stopped for {session_id}"
                )
            lease_lost_errors.append(error)
            logger.error(
                "mode transition fence lease lost: session=%s error=%s",
                session_id,
                error,
            )
            if body_active and not owner_task.done():
                owner_task.cancel(owner_cancel_marker)

        renew_task.add_done_callback(_stop_owner_on_lease_lost)
        try:
            try:
                yield
            finally:
                body_active = False
                cleanup = asyncio.create_task(
                    self._cleanup_mode_transition_fence(
                        key=key,
                        token=token,
                        renew_task=renew_task,
                    ),
                    name=f"mode-transition-fence-cleanup-{session_id}",
                )
                _PENDING_MODE_FENCE_CLEANUPS.add(cleanup)
                cleanup.add_done_callback(_PENDING_MODE_FENCE_CLEANUPS.discard)
                cleanup.add_done_callback(self._log_subscriber_cleanup_result)
                await asyncio.shield(cleanup)
            if lease_lost_errors:
                remaining_cancellations = owner_task.uncancel()
                if remaining_cancellations > 0:
                    raise asyncio.CancelledError
                error = lease_lost_errors[-1]
                if isinstance(error, ModeTransitionFenceLostError):
                    raise error
                raise ModeTransitionFenceLostError(
                    f"mode transition fence lease lost for {session_id}"
                ) from error
        except asyncio.CancelledError as exc:
            if lease_lost_errors and exc.args == (owner_cancel_marker,):
                remaining_cancellations = owner_task.uncancel()
                if remaining_cancellations == 0:
                    error = lease_lost_errors[-1]
                    if isinstance(error, ModeTransitionFenceLostError):
                        raise error
                    raise ModeTransitionFenceLostError(
                        f"mode transition fence lease lost for {session_id}"
                    ) from error
            raise

    async def _renew_mode_transition_fence(self, *, key: str, token: str) -> None:
        while True:
            await asyncio.sleep(_MODE_TRANSITION_FENCE_RENEW_SECONDS)
            try:
                renewed = await self._redis.eval(
                    _LUA_RENEW_OWNER_IF_EQUAL,
                    1,
                    key,
                    token,
                    _MODE_TRANSITION_FENCE_TTL_SECONDS,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ModeTransitionFenceLostError(
                    f"mode transition fence lease lost: renewal failed for {key}"
                ) from exc
            if int(renewed or 0) != 1:
                raise ModeTransitionFenceLostError(
                    f"mode transition fence lease lost for {key}"
                )

    async def _cleanup_mode_transition_fence(
        self,
        *,
        key: str,
        token: str,
        renew_task: asyncio.Task[None],
    ) -> None:
        renew_task.cancel()
        try:
            await renew_task
        except asyncio.CancelledError:
            pass
        except ModeTransitionFenceLostError:
            # The done callback already logged the loss and stopped the owner.
            pass
        except Exception:
            logger.warning(
                "mode transition fence renewal failed before cleanup: key=%s",
                key,
                exc_info=True,
            )
        await self._release_owner_if_equal(
            owner_key=key,
            connection_id=token,
        )

    async def suspend_idle(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> None:
        async with self._repo_context() as repo:
            transitioned = await repo.suspend_running_background_if_active(session_id)
        if not transitioned:
            logger.info("idle suspend skipped stale session=%s", session_id)
            return
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

    async def sweep_expired(
        self,
        *,
        user_id: str,
        sweep_now: datetime | None = None,
    ) -> list[str]:
        effective_now = self._as_utc(sweep_now) or self._now_utc()
        expired = await run_lua_with_fallback(
            self._redis,
            source=LUA_SWEEP_EXPIRED,
            sha=self._sha_sweep or LUA_SWEEP_EXPIRED_SHA,
            keys=[self._bg_key(user_id)],
            args=[f"{effective_now.timestamp():.6f}"],
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

    async def reconcile_global_background_memberships(
        self,
        *,
        reconcile_now: datetime | None = None,
    ) -> dict[str, int]:
        """Reconcile counted slots through a durable held/released quarantine."""
        now = self._as_utc(reconcile_now) or self._now_utc()
        cleaned = 0
        repaired = 0
        skipped = 0
        for session_id, member_user_id, member_generation in (
            await self._scan_global_background_memberships()
        ):
            try:
                # Admission, reconnect and retry use this same fence. Waiting
                # for it prevents reconciliation from deleting an in-flight
                # Redis reservation before its PG CAS commits.
                async with self.mode_transition_fence(session_id=session_id):
                    refreshed_membership = self._decode_system_membership(
                        await self._redis.hget(
                            self._system_members_key(), session_id
                        )
                    )
                    if (
                        refreshed_membership is None
                        or refreshed_membership[0] != member_generation
                    ):
                        skipped += 1
                        continue
                    member_generation, refreshed_user_id = refreshed_membership
                    member_user_id = refreshed_user_id or member_user_id
                    current = await self._read_session_snapshot(session_id)
                    if (
                        current is not None
                        and current.execution_revision < member_generation
                    ):
                        skipped += 1
                        continue
                    is_live_background = bool(
                        current is not None
                        and current.status == SessionStatus.RUNNING
                        and current.execution_mode == "background"
                        and current.execution_phase in ("running", "suspended")
                    )
                    if is_live_background:
                        assert current is not None
                        expiry = self._as_utc(current.expires_at)
                        projection_user_id = str(
                            current.user_id or member_user_id or ""
                        )
                        if expiry is None or not projection_user_id:
                            skipped += 1
                            logger.warning(
                                "cannot repair global background membership "
                                "without user/expiry: session=%s",
                                session_id,
                            )
                            continue
                        if current.execution_revision >= member_generation:
                            await self._ensure_authoritative_background_projection(
                                session_id=session_id,
                                user_id=projection_user_id,
                                expires_at=expiry,
                                generation=current.execution_revision,
                            )
                            repaired += 1
                            continue

                    require_generation = False
                    if member_user_id is None:
                        member_user_id = await self._find_legacy_projection_user(
                            session_id=session_id,
                            expected_generation=member_generation,
                        )
                        require_generation = member_user_id is not None
                    marked = await self._mark_background_reconcile_held(
                        session_id=session_id,
                        generation=member_generation,
                        user_id=member_user_id,
                        require_generation=require_generation,
                        now=now,
                    )
                    if marked not in (1, 2):
                        skipped += 1
            except Exception:
                skipped += 1
                logger.exception(
                    "global background membership reconcile failed: session=%s",
                    session_id,
                )
        marker_result = await self._reconcile_background_markers(now=now)
        cleaned += marker_result["cleaned"]
        repaired += marker_result["repaired"]
        skipped += marker_result["skipped"]
        return {"cleaned": cleaned, "repaired": repaired, "skipped": skipped}

    async def get_background_quota(self, user_id: str) -> dict[str, int]:
        raw_system_used = await self._redis.get(self._system_key())
        raw_user_used = await self._redis.hlen(self._user_key(user_id))
        return {
            "system_used": self._parse_count(raw_system_used),
            "system_limit": self._max_system_bg,
            "user_used": self._parse_count(raw_user_used),
            "user_limit": self._max_user_bg,
        }

    async def resume(
        self,
        *,
        session_id: str,
        user_id: str,
        execution_mode: Literal["foreground", "background"] | None = None,
        mode: Literal["foreground", "background"] | None = None,
        expires_at: datetime | None = None,
        previous_expires_at: datetime | None = None,
        retry_budget_remaining: int | None = None,
        expected_execution_revision: int | None = None,
    ) -> int | None:
        target_mode = execution_mode or mode
        if target_mode == "foreground":
            before = await self._read_session_snapshot(session_id)
            if before is None or before.status != SessionStatus.RUNNING:
                return None
            resumed_revision: int | None = None
            expected_revision = before.execution_revision
            if self._is_running_auto_degrade(before):
                pending = self._pending_execution_event(
                    execution_revision=expected_revision + 1,
                    execution_mode="foreground",
                    execution_phase="running",
                    transition_reason="auto_degrade_sse_reconnect",
                    retry_budget_remaining=before.retry_budget_remaining,
                )
            if self._uow_factory is not None:
                async with self._uow_factory() as uow:
                    if self._is_running_auto_degrade(before):
                        resumed_revision = (
                        await uow.session.resume_auto_degrade_to_foreground_if_running(
                            session_id,
                            expected_execution_revision=expected_revision,
                            pending_event=pending,
                        )
                        )
                    if resumed_revision is not None:
                        await _commit_uow_if_real(uow)
            else:
                if self._repo is None:
                    raise RuntimeError("supervisor has no repository source")
                if self._is_running_auto_degrade(before):
                    resumed_revision = (
                    await self._repo.resume_auto_degrade_to_foreground_if_running(
                        session_id,
                        expected_execution_revision=expected_revision,
                        pending_event=pending,
                    )
                    )
            current = await self._read_session_snapshot(session_id)
            if not (
                current is not None
                and current.status == SessionStatus.RUNNING
                and current.execution_mode == "foreground"
            ):
                # CAS loss to an explicit/background or terminal transition is
                # not proof that this reconnect owns the Redis slot.
                return None
            stale_generation = (
                expected_revision
                if resumed_revision is not None
                else max(current.execution_revision - 1, 0)
            )
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="t7_reconnect",
                expected_generation=stale_generation,
                allow_legacy=True,
            )
            return resumed_revision

        if target_mode != "background":
            raise ValueError("execution_mode must be foreground or background")
        if expires_at is None:
            raise ValueError("background resume requires expires_at")
        if expected_execution_revision is None:
            raise ValueError(
                "background resume requires expected_execution_revision"
            )

        before = await self._read_session_snapshot(session_id)
        if not (
            before is not None
            and before.status == SessionStatus.RUNNING
            and before.execution_mode == "background"
            and before.execution_phase == "running"
            and before.execution_revision == expected_execution_revision
        ):
            raise SupervisorContractError(
                "R3",
                "suspended",
                "background",
                "stale retry claim",
            )

        rc: int | None = None
        try:
            marker_restore = await self._restore_authoritative_background_marker(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                generation=expected_execution_revision,
            )
            if marker_restore is not None:
                _, marker_restore_rc = marker_restore
                # rc=1 created the counted reservation; rc=2 only refreshed
                # an existing one. Preserve rollback semantics for each case.
                rc = 0 if marker_restore_rc == 1 else 3
            else:
                rc = await self._run_lua_admit(
                    session_id=session_id,
                    user_id=user_id,
                    expires_at=expires_at,
                    generation=expected_execution_revision,
                )
                if rc in (6, 7):
                    rc = await self._run_lua_admit(
                        session_id=session_id,
                        user_id=user_id,
                        expires_at=expires_at,
                        generation=expected_execution_revision,
                        authoritative_repair=True,
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
            if rc == 4:
                raise SupervisorContractError(
                    "R3",
                    "suspended",
                    "background",
                    "stale retry claim",
                )
            if rc not in (0, 3, 5):
                raise RuntimeError(
                    f"background retry admission invariant lost: "
                    f"session={session_id} rc={rc}"
                )
            if rc == 3:
                sync_rc = await self._sync_background_expiry_to_redis(
                    session_id=session_id,
                    user_id=user_id,
                    expires_at=expires_at,
                    execution_revision=expected_execution_revision,
                )
                if sync_rc != 1:
                    await self._lua_revoke(
                        session_id=session_id,
                        user_id=user_id,
                        reason="resume_retry_sync_stale",
                        expected_generation=expected_execution_revision,
                        allow_legacy=True,
                    )
                    raise SupervisorContractError(
                        "R3",
                        "suspended",
                        "background",
                        "stale retry claim",
                    )

            after = await self._read_session_snapshot(session_id)
        except SupervisorContractError:
            raise
        except BaseException:
            try:
                await self.rollback_background_resume_admission(
                    session_id=session_id,
                    user_id=user_id,
                    # An interrupted Lua request has ambiguous durability. The
                    # existing-slot path restores the exact previous expiry if
                    # generation r landed, and is a no-op against r+1.
                    admission_rc=rc if rc in (0, 3, 5) else 3,
                    previous_expires_at=previous_expires_at,
                    expected_execution_revision=expected_execution_revision,
                )
            except Exception:
                logger.exception(
                    "background retry admission exception cleanup failed: %s",
                    session_id,
                )
            raise
        if not (
            after is not None
            and after.status == SessionStatus.RUNNING
            and after.execution_mode == "background"
            and after.execution_phase == "running"
            and after.execution_revision == expected_execution_revision
        ):
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="resume_retry_stale",
                expected_generation=expected_execution_revision,
                allow_legacy=True,
            )
            raise SupervisorContractError(
                "R3",
                "suspended",
                "background",
                "stale retry claim",
            )
        # Retry claim already persisted phase/expires/retry budget. Keep this
        # path Redis-only so a stale retry cannot reopen a terminal PG row.
        if rc == 3:
            await self._reset_inflight_counts(session_id=session_id)
        return rc

    async def rollback_background_resume_admission(
        self,
        *,
        session_id: str,
        user_id: str,
        admission_rc: int,
        previous_expires_at: datetime | None,
        expected_execution_revision: int,
    ) -> None:
        if admission_rc == 0:
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="resume_retry_rollback",
                expected_generation=expected_execution_revision,
                allow_legacy=True,
            )
            return
        if admission_rc not in (3, 5):
            return
        if previous_expires_at is None:
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="resume_retry_rollback_missing_expiry",
                expected_generation=expected_execution_revision,
                allow_legacy=True,
            )
            return
        await self._sync_background_expiry_to_redis(
            session_id=session_id,
            user_id=user_id,
            expires_at=previous_expires_at,
            execution_revision=expected_execution_revision,
        )

    async def revoke_background_resume_admission(
        self,
        *,
        session_id: str,
        user_id: str,
        admission_rc: int,
        expected_execution_revision: int,
    ) -> None:
        if admission_rc not in (0, 3, 5):
            return
        await self._lua_revoke(
            session_id=session_id,
            user_id=user_id,
            reason="resume_retry_terminal",
            expected_generation=expected_execution_revision,
            allow_legacy=True,
        )

    async def cleanup_background_slot(
        self,
        *,
        session_id: str,
        user_id: str,
        reason: str,
    ) -> None:
        await self._revoke_authoritative_slot(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
        )

    async def terminate_expired_background(
        self,
        *,
        session_id: str,
        user_id: str,
        sweep_now: datetime,
        notification_emitter=None,
    ) -> bool:
        """Terminalize only when PostgreSQL still proves the swept expiry."""
        before = await self._read_session_snapshot(session_id)
        if before is None:
            await self._revoke_authoritative_slot(
                session_id=session_id,
                user_id=user_id,
                reason="watchdog_missing_pg_row",
            )
            return False
        expected_revision = before.execution_revision
        if self._uow_factory is not None:
            async with self._uow_factory() as uow:
                transitioned = (
                    await self._require_state_machine().terminate_expired_background(
                        session_id,
                        SessionStatus.TIMED_OUT,
                        "watchdog_timeout",
                        expires_at_lte=sweep_now,
                        session_repo=uow.session,
                        expected_execution_revision=expected_revision,
                        pending_event=None,
                    )
                )
                if transitioned:
                    await _commit_uow_if_real(uow)
        elif self._repo is not None:
            transitioned = (
                await self._require_state_machine().terminate_expired_background(
                    session_id,
                    SessionStatus.TIMED_OUT,
                    "watchdog_timeout",
                    expires_at_lte=sweep_now,
                    session_repo=self._repo,
                    expected_execution_revision=expected_revision,
                    pending_event=None,
                )
            )
        else:
            raise RuntimeError("supervisor has no repository source")

        if not transitioned:
            await self._repair_swept_background_projection(
                session_id=session_id,
                fallback_user_id=user_id,
                sweep_now=sweep_now,
            )
            return False
        if notification_emitter is not None:
            try:
                await notification_emitter.emit(
                    user_id=user_id,
                    event_type="bg_failed_watchdog",
                    payload={"session_id": session_id},
                )
            except Exception:
                logger.warning(
                    "bg_failed_watchdog emit failed: session=%s",
                    session_id,
                    exc_info=True,
                )
        await self._lua_revoke(
            session_id=session_id,
            user_id=user_id,
            reason="watchdog_timeout",
            expected_generation=expected_revision,
            allow_legacy=True,
        )
        await self._maybe_stop_supervisor_for_session(session_id)
        return True

    async def _repair_swept_background_projection(
        self,
        *,
        session_id: str,
        fallback_user_id: str,
        sweep_now: datetime,
    ) -> None:
        """Repair or remove the generation proven by the latest PG row."""
        current = await self._read_session_snapshot(session_id)
        expiry = self._future_background_expiry(current, after=sweep_now)
        if expiry is None:
            await self._revoke_authoritative_slot(
                session_id=session_id,
                user_id=(
                    str(current.user_id or fallback_user_id)
                    if current is not None
                    else fallback_user_id
                ),
                reason="watchdog_sweep_authoritative_cleanup",
                restore_if_background=current is not None,
            )
            return
        projection_user_id = str(current.user_id or fallback_user_id)
        await self._ensure_authoritative_background_projection(
            session_id=session_id,
            user_id=projection_user_id,
            expires_at=expiry,
            generation=current.execution_revision,
        )

        # A reconnect can commit foreground while the repair Lua is running.
        # Re-read PG and revoke the projection we just wrote if it is stale.
        refreshed = await self._read_session_snapshot(session_id)
        refreshed_expiry = self._future_background_expiry(
            refreshed,
            after=sweep_now,
        )
        if refreshed_expiry is None:
            await self._revoke_authoritative_slot(
                session_id=session_id,
                user_id=projection_user_id,
                reason="watchdog_sweep_repair_stale",
                restore_if_background=True,
            )
            return
        if refreshed_expiry != expiry:
            await self._ensure_authoritative_background_projection(
                session_id=session_id,
                user_id=projection_user_id,
                expires_at=refreshed_expiry,
                generation=refreshed.execution_revision,
            )

    def _future_background_expiry(
        self,
        session: Session | None,
        *,
        after: datetime,
    ) -> datetime | None:
        if (
            session is None
            or session.status != SessionStatus.RUNNING
            or session.execution_mode != "background"
            or session.execution_phase not in ("running", "suspended")
        ):
            return None
        expiry = self._as_utc(session.expires_at)
        if expiry is None or expiry <= self._as_utc(after):
            return None
        return expiry

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
        notification_emitter=None,
    ) -> None:
        emit_bg_failed_watchdog = False
        session = None
        # codex r11 [HIGH CONTRACT] — explicit commit so swallowed
        # CancelledError on UoW commit cannot leave the registry stop
        # firing on a non-durable terminal write. Mirrors the runner's
        # explicit ``await _commit_uow_if_real(uow)`` pattern at
        # agent_task_runner.py:3092. Falls back to the previous
        # auto-commit path when the supervisor was constructed with a
        # direct ``session_repository`` (test wiring) — that path lacks
        # an explicit commit hook by design.
        if self._uow_factory is not None:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id)
                if (
                    session is not None
                    and session.status
                    not in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)
                    and session.execution_mode == "background"
                    and session.execution_phase in ("running", "suspended")
                ):
                    transitioned = await self._require_state_machine().terminate(
                        session_id, status, terminal_reason, session_repo=uow.session
                    )
                    # Raise on commit failure so post-commit side-effects
                    # below (lua_revoke / stop) only run on durable terminal.
                    await _commit_uow_if_real(uow)
                    emit_bg_failed_watchdog = (
                        transitioned is not False
                        and terminal_reason == "watchdog_timeout"
                        and bool(getattr(session, "was_background", False))
                        and getattr(session, "user_id", None) is not None
                    )
        elif self._repo is not None:
            session = await self._repo.get_by_id(session_id)
            if (
                session is not None
                and session.status
                not in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)
                and session.execution_mode == "background"
                and session.execution_phase in ("running", "suspended")
            ):
                transitioned = await self._require_state_machine().terminate(
                    session_id, status, terminal_reason, session_repo=self._repo
                )
                emit_bg_failed_watchdog = (
                    transitioned is not False
                    and terminal_reason == "watchdog_timeout"
                    and bool(getattr(session, "was_background", False))
                    and getattr(session, "user_id", None) is not None
                )
        if notification_emitter is not None and emit_bg_failed_watchdog:
            try:
                await notification_emitter.emit(
                    user_id=str(session.user_id),
                    event_type="bg_failed_watchdog",
                    payload={"session_id": session_id},
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "bg_failed_watchdog emit failed: session=%s err=%s",
                    session_id,
                    exc,
                )
        # C3 PR-3c (codex r6 [HIGH CONTRACT] + r10 [HIGH CONTRACT]) —
        # non-runner terminal write (idle_watchdog timeout, admin cancel,
        # retry-budget exhaustion). Order matters:
        #   1. DB terminal write committed above.
        #   2. ``_lua_revoke`` clears the Redis supervisor slot. MUST run
        #      before the new stop seam below — codex r10 caught that
        #      stop-then-revoke was a regression: an outer CancelledError
        #      at the stop await would re-raise (helper deliberately
        #      propagates cancel) and skip _lua_revoke, leaving a stale
        #      Redis slot after the DB row was already terminal.
        #   3. ``_maybe_stop_supervisor_for_session`` stops the per-pod
        #      MailboxSupervisor task. Shielded internally; safe to put
        #      last because if cancel hits here, the only thing skipped
        #      is the supervisor task — the registry's `stop_all()` at
        #      pod shutdown sweeps any leak, and the supervisor doesn't
        #      do anything visible after the session is terminal anyway.
        await self._revoke_authoritative_slot(
            session_id=session_id,
            user_id=user_id,
            reason=terminal_reason,
        )
        await self._maybe_stop_supervisor_for_session(session_id)

    async def reconcile_running_background_at_boot(
        self,
        *,
        notification_emitter=None,
    ) -> dict[str, int]:
        finishing = 0
        suspended = 0
        # codex r9 [HIGH CONTRACT] + r10 [HIGH CONTRACT] — per-row explicit
        # commit so "added to finished_session_ids" durably implies
        # "DB terminal write succeeded". The shared ``_repo_context`` /
        # DBUnitOfWork swallows ``CancelledError`` during commit
        # (db_uow.py:71) for SSE-disconnect ergonomics, which means
        # "left the with-block" does NOT imply "commit durably succeeded".
        # The runner faces the same constraint and solves it with explicit
        # ``await _commit_uow_if_real(uow)`` inside its shielded terminal
        # task (agent_task_runner.py:3092). We mirror that here: open a
        # fresh UoW per FINISHING row, call ``commit()`` directly so
        # commit failure surfaces as a raised exception, then add to the
        # stop-list ONLY on observed success. Suspended rows still use the
        # auto-commit ``_repo_context`` because they don't trigger a
        # downstream registry side-effect — a swallowed commit just means
        # the next pod restart retries the suspend.
        rows: list = []
        finished_session_ids: list[str] = []
        async with self._repo_context() as repo:
            rows = await repo.find_running_background()

        for row in rows:
            try:
                if row.status == SessionStatus.FINISHING:
                    transitioned: bool | None = None
                    if self._uow_factory is not None:
                        # Production path: fresh UoW + explicit commit so
                        # commit failure raises and skips the registry
                        # stop side-effect for this row.
                        async with self._uow_factory() as uow:
                            transitioned = await self._require_state_machine().terminate(
                                row.session_id,
                                SessionStatus.TIMED_OUT,
                                "server_restart",
                                session_repo=uow.session,
                            )
                            await _commit_uow_if_real(uow)
                    elif self._repo is not None:
                        # Test path: direct repo without UoW. No explicit
                        # commit available — fall back to the prior
                        # behavior (acceptable because tests don't exercise
                        # commit-cancel ergonomics).
                        transitioned = await self._require_state_machine().terminate(
                            row.session_id,
                            SessionStatus.TIMED_OUT,
                            "server_restart",
                            session_repo=self._repo,
                        )
                    # codex r11 [HIGH CONTRACT] — commit succeeded above
                    # (explicit raise on the prod path). Append to the
                    # stop-list IMMEDIATELY so a subsequent best-effort
                    # side-effect failure (lua_revoke / notification)
                    # CANNOT cancel the supervisor stop and leak the slot.
                    # The terminal DB write is durable; the registry MUST
                    # see the stop.
                    if transitioned is not False:
                        finished_session_ids.append(row.session_id)
                    # Best-effort Redis revoke — failure is isolated so
                    # the supervisor stop still runs at the post-commit
                    # phase below.
                    try:
                        await self._revoke_authoritative_slot(
                            session_id=row.session_id,
                            user_id=row.user_id,
                            reason="server_restart",
                        )
                    except Exception:
                        logger.exception(
                            "supervisor boot reconcile: _lua_revoke failed for %s "
                            "— DB terminal already committed; supervisor stop "
                            "will still fire from finished_session_ids",
                            row.session_id,
                        )
                    if (
                        transitioned is not False
                        and notification_emitter is not None
                    ):
                        try:
                            await notification_emitter.emit(
                                user_id=row.user_id,
                                event_type="bg_terminal_server_restart",
                                payload={"session_id": row.session_id},
                            )
                        except Exception:
                            logger.exception(
                                "supervisor boot reconcile: notification emit "
                                "failed for %s — supervisor stop will still fire",
                                row.session_id,
                            )
                    finishing += 1
                else:
                    async with self._repo_context() as repo:
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
        try:
            await self.reconcile_global_background_memberships()
        except Exception:
            logger.exception("supervisor boot global membership reconcile failed")
        # C3 PR-3c (codex r6 + r9 + r10 [HIGH CONTRACT]) — post-commit
        # phase. Only sessions whose explicit ``commit()`` above returned
        # normally reach this loop; commit failure (including swallowed
        # CancelledError on the legacy auto-commit path, now eliminated
        # for FINISHING rows) skips ``finished_session_ids.append``, so
        # stopping a supervisor here implies the session truly is
        # terminal in DB.
        for sid in finished_session_ids:
            await self._maybe_stop_supervisor_for_session(sid)
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

    async def _reset_inflight_counts(self, *, session_id: str) -> None:
        try:
            await self._redis.hset(
                self._hot_key(session_id),
                mapping={
                    "inflight_llm_count": 0,
                    "inflight_tool_count": 0,
                },
            )
        except Exception:
            logger.warning(
                "reset inflight counts failed for %s",
                session_id,
                exc_info=True,
            )

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

            await self._revoke_authoritative_slot(
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

    async def _lua_revoke(
        self,
        *,
        session_id: str,
        user_id: str,
        reason: str,
        expected_generation: int,
        allow_legacy: bool = False,
    ) -> int:
        raw_marker = self._decode_redis_value(
            await self._redis.hget(
                self._background_reconcile_marker_key(), session_id
            )
        )
        marker = self._decode_background_reconcile_marker(raw_marker)
        expected_marker = ""
        released_marker = ""
        released_due = ""
        if marker is not None and marker.phase == "held":
            released = self._released_background_reconcile_marker(
                marker,
                now=self._now_utc(),
            )
            expected_marker = raw_marker or ""
            released_marker = self._encode_background_reconcile_marker(released)
            released_due = f"{released.next_check_at.timestamp():.6f}"
        rc = await run_lua_with_fallback(
            self._redis,
            source=LUA_REVOKE,
            sha=self._sha_revoke or LUA_REVOKE_SHA,
            keys=[
                self._system_key(),
                self._user_key(user_id),
                self._bg_key(user_id),
                self._generation_key(user_id),
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                expected_generation,
                1 if allow_legacy else 0,
                expected_marker,
                released_marker,
                released_due,
            ],
            meter=self._meter,
        )
        self._meter_inc("revoke", reason=reason)
        return int(rc)

    async def _revoke_authoritative_slot(
        self,
        *,
        session_id: str,
        user_id: str,
        reason: str,
        restore_if_background: bool = False,
    ) -> int:
        """Cleanup an actual slot and optionally repair a concurrent PG winner."""
        current = await self._read_session_snapshot(session_id)
        raw_generation = await self._redis.hget(
            self._generation_key(user_id), session_id
        )
        decoded_generation = self._decode_redis_value(raw_generation)
        if decoded_generation is None:
            raw_membership = await self._redis.hget(
                self._system_members_key(), session_id
            )
            decoded_membership = self._decode_system_membership(raw_membership)
            decoded_generation = (
                str(decoded_membership[0])
                if decoded_membership is not None
                else None
            )
        if decoded_generation is not None:
            expected_generation = int(decoded_generation)
            allow_legacy = True
        elif current is None:
            expected_generation = 0
            allow_legacy = True
        elif current.execution_mode == "background" and current.status == SessionStatus.RUNNING:
            expected_generation = current.execution_revision
            allow_legacy = True
        else:
            expected_generation = max(current.execution_revision - 1, 0)
            allow_legacy = True
        # PostgreSQL is authoritative here. A missing or lower membership
        # generation can therefore be cleaned exactly once; expected_generation
        # still fences a newer owner. Callers that are repairing a crash window
        # may opt into the post-revoke PG reread below. Final/delete callers do
        # not, otherwise a still-visible pre-delete row would recreate the slot.
        rc = await self._lua_revoke(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
            expected_generation=expected_generation,
            allow_legacy=allow_legacy,
        )
        if rc < 0:
            raise RuntimeError(
                f"background membership invariant lost for {session_id}"
            )

        # Crash-repair can race a new generation-r+1 admission: after deleting
        # its Redis reservation, that admission may still win the PG CAS. PG is
        # authoritative, so reread and restore exactly the winning generation.
        refreshed = await self._read_session_snapshot(session_id)
        if (
            restore_if_background
            and refreshed is not None
            and refreshed.status == SessionStatus.RUNNING
            and refreshed.execution_mode == "background"
            and refreshed.expires_at is not None
        ):
            await self._ensure_authoritative_background_projection(
                session_id=session_id,
                user_id=str(refreshed.user_id or user_id),
                expires_at=self._as_utc(refreshed.expires_at),
                generation=refreshed.execution_revision,
            )
        return rc

    async def _cleanup_failed_background_admission(
        self,
        *,
        session_id: str,
        user_id: str,
        reason: str,
        generation: int,
    ) -> None:
        """Do not let one same-generation CAS loser delete the winner's slot."""
        try:
            current = await self._read_session_snapshot(session_id)
        except Exception:
            logger.warning(
                "background admission cleanup deferred; PG reread failed: %s",
                session_id,
                exc_info=True,
            )
            return
        if (
            current is not None
            and current.status == SessionStatus.RUNNING
            and current.execution_mode == "background"
            and current.execution_revision == generation
        ):
            return
        rc = await self._lua_revoke(
            session_id=session_id,
            user_id=user_id,
            reason=reason,
            expected_generation=generation,
            allow_legacy=True,
        )
        if rc < 0:
            raise RuntimeError(
                f"background membership invariant lost for {session_id}"
            )
        refreshed = await self._read_session_snapshot(session_id)
        if (
            refreshed is not None
            and refreshed.status == SessionStatus.RUNNING
            and refreshed.execution_mode == "background"
            and refreshed.execution_revision == generation
            and refreshed.expires_at is not None
        ):
            await self._ensure_authoritative_background_projection(
                session_id=session_id,
                user_id=str(refreshed.user_id or user_id),
                expires_at=self._as_utc(refreshed.expires_at),
                generation=generation,
            )

    async def _admit_background_slot(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        from_phase: str,
        generation: int,
    ) -> int:
        rc = await self._run_lua_admit(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            generation=generation,
        )
        if rc == 6:
            # A generation-less slot predates this fencing contract. PG has
            # already proved the session is foreground, so it is safe to
            # remove that legacy projection before retrying admission.
            await self._lua_revoke(
                session_id=session_id,
                user_id=user_id,
                reason="legacy_generation_cleanup",
                expected_generation=max(generation - 1, 0),
                allow_legacy=True,
            )
            rc = await self._run_lua_admit(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                generation=generation,
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
            self._meter_inc("admit", result="same_generation")
        if rc == 4:
            self._meter_inc("admit", result="stale_generation")
            raise SupervisorContractError(
                "R3", from_phase, "background", "stale background generation"
            )
        return rc

    async def _run_lua_admit(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        generation: int,
        authoritative_repair: bool = False,
    ) -> int:
        expires_at_unix = expires_at.astimezone(timezone.utc).timestamp()
        activity_at_unix = datetime.now(timezone.utc).timestamp()
        slot_ttl = self._background_slot_ttl_seconds(expires_at)
        rc = await run_lua_with_fallback(
            self._redis,
            source=LUA_ADMIT,
            sha=self._sha_admit or LUA_ADMIT_SHA,
            keys=[
                self._system_key(),
                self._user_key(user_id),
                self._hot_key(session_id),
                self._bg_key(user_id),
                self._generation_key(user_id),
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                f"{expires_at_unix:.6f}",
                str(self._max_system_bg),
                str(self._max_user_bg),
                f"{activity_at_unix:.6f}",
                generation,
                1 if authoritative_repair else 0,
                self._encode_system_membership(user_id, generation),
                slot_ttl,
            ],
            meter=self._meter,
        )
        return int(rc)

    async def _redis_expiry_matches(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        execution_revision: int,
    ) -> bool:
        expected = expires_at.timestamp()
        required_ttl = self._background_slot_ttl_seconds(expires_at)
        raw_hash = await self._redis.hget(self._user_key(user_id), session_id)
        raw_score = await self._redis.zscore(self._bg_key(user_id), session_id)
        raw_generation = await self._redis.hget(
            self._generation_key(user_id), session_id
        )
        raw_membership = await self._redis.hget(self._system_members_key(), session_id)
        user_ttl = await self._redis.ttl(self._user_key(user_id))
        zset_ttl = await self._redis.ttl(self._bg_key(user_id))
        generation_ttl = await self._redis.ttl(self._generation_key(user_id))
        try:
            hash_value = float(self._decode_redis_value(raw_hash) or "nan")
            score_value = float(raw_score)
            generation_value = int(self._decode_redis_value(raw_generation) or "-1")
            decoded_membership = self._decode_system_membership(raw_membership)
            membership_value = (
                decoded_membership[0] if decoded_membership is not None else -1
            )
            membership_user_id = (
                decoded_membership[1] if decoded_membership is not None else None
            )
        except (TypeError, ValueError):
            return False
        return (
            abs(hash_value - expected) <= 0.001
            and abs(score_value - expected) <= 0.001
            and generation_value == execution_revision
            and membership_value == execution_revision
            and membership_user_id == user_id
            and int(user_ttl) >= required_ttl - 1
            and int(zset_ttl) >= required_ttl - 1
            and int(generation_ttl) >= required_ttl - 1
        )

    async def _sync_background_expiry_to_redis(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        execution_revision: int,
    ) -> int:
        expires_at_unix = expires_at.timestamp()
        slot_ttl = self._background_slot_ttl_seconds(expires_at)
        result = await run_lua_with_fallback(
            self._redis,
            source=LUA_SYNC_BACKGROUND_EXPIRY,
            sha=(
                self._sha_sync_background_expiry
                or LUA_SYNC_BACKGROUND_EXPIRY_SHA
            ),
            keys=[
                self._user_key(user_id),
                self._bg_key(user_id),
                self._generation_key(user_id),
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                f"{expires_at_unix:.6f}",
                slot_ttl,
                execution_revision,
                self._encode_system_membership(user_id, execution_revision),
            ],
            meter=self._meter,
        )
        return int(result)

    async def _restore_authoritative_background_marker(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        generation: int,
    ) -> tuple[_BackgroundReconcileMarker, int] | None:
        marker_raw = self._decode_redis_value(
            await self._redis.hget(
                self._background_reconcile_marker_key(), session_id
            )
        )
        marker = self._decode_background_reconcile_marker(marker_raw)
        if marker is None:
            return None
        if generation < marker.generation:
            raise ModeTransitionFenceLostError(
                f"newer background reconcile marker owns {session_id}"
            )
        if marker.user_id is not None and marker.user_id != user_id:
            raise ModeTransitionFenceLostError(
                f"background reconcile marker owner changed for {session_id}"
            )
        marker_rc = await self._restore_background_from_marker(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            generation=generation,
            expected_marker=marker_raw or "",
        )
        if marker_rc > 0:
            return marker, marker_rc
        if marker_rc < 0:
            raise ModeTransitionFenceLostError(
                f"newer background generation already owns {session_id}"
            )
        return None

    async def _ensure_authoritative_background_projection(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        generation: int,
    ) -> None:
        marker_restore = await self._restore_authoritative_background_marker(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            generation=generation,
        )
        if marker_restore is not None:
            return
        sync_rc = await self._sync_background_expiry_to_redis(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            execution_revision=generation,
        )
        if sync_rc == 1:
            return
        if sync_rc == 0:
            raise ModeTransitionFenceLostError(
                f"newer background generation already owns {session_id}"
            )
        # First use normal admission. Complete projection loss is therefore
        # counted exactly once and still respects the system/user caps. When
        # the user slot survived but generation/membership did not, rc=6/7
        # proves the old counter already included that slot. The authoritative
        # retry atomically upgrades that counted projection without a
        # revoke/readmit gap or another counter increment.
        repair_rc = await self._run_lua_admit(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            generation=generation,
        )
        if repair_rc in (6, 7):
            repair_rc = await self._run_lua_admit(
                session_id=session_id,
                user_id=user_id,
                expires_at=expires_at,
                generation=generation,
                authoritative_repair=True,
            )
        if repair_rc in (1, 2):
            logger.error(
                "background projection repair blocked by quota: session=%s rc=%s",
                session_id,
                repair_rc,
            )
        if repair_rc not in (0, 3, 5):
            raise RuntimeError(
                f"cannot safely repair background quota projection: "
                f"session={session_id} rc={repair_rc}"
            )
        sync_rc = await self._sync_background_expiry_to_redis(
            session_id=session_id,
            user_id=user_id,
            expires_at=expires_at,
            execution_revision=generation,
        )
        if sync_rc != 1:
            raise RuntimeError(
                f"background projection repair did not converge: "
                f"session={session_id} rc={sync_rc}"
            )

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

    @asynccontextmanager
    async def _transition_repo_context(
        self,
    ) -> AsyncIterator[tuple[SessionRepository, IUnitOfWork | None]]:
        """Expose the owning UoW where transition durability must be explicit."""
        if self._repo is not None:
            yield self._repo, None
            return

        if self._uow_factory is None:
            raise RuntimeError("supervisor has no repository source")
        async with self._uow_factory() as uow:
            yield uow.session, uow

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

    def _now_utc(self) -> datetime:
        now = self._utcnow()
        if not isinstance(now, datetime):
            raise TypeError("utcnow must return datetime")
        if now.tzinfo is None:
            return now.replace(tzinfo=timezone.utc)
        return now.astimezone(timezone.utc)

    @staticmethod
    def _pending_execution_event(
        *,
        execution_revision: int,
        execution_mode: Literal["foreground", "background"],
        execution_phase: Literal[
            "running", "recovering", "idle", "suspended", "terminating", "terminated"
        ],
        transition_reason: str,
        retry_budget_remaining: int,
        background_reason: Literal["explicit", "auto_degrade"] | None = None,
        expires_at: datetime | None = None,
        suspended_reason: Literal["bg_idle_timeout", "server_restart"] | None = None,
        terminal_reason: Literal[
            "natural",
            "user_cancel",
            "server_restart",
            "resume_state_lost",
            "watchdog_timeout",
        ] | None = None,
    ) -> PendingExecutionEvent:
        return PendingExecutionEvent(
            payload=ExecutionStatePayload(
                execution_revision=execution_revision,
                execution_mode=execution_mode,
                execution_phase=execution_phase,
                transition_reason=transition_reason,
                background_reason=background_reason,
                expires_at=expires_at,
                retry_budget_remaining=retry_budget_remaining,
                suspended_reason=suspended_reason,
                terminal_reason=terminal_reason,
            )
        )

    @staticmethod
    def _as_utc(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _is_running_auto_degrade(session: Session | None) -> bool:
        return bool(
            session is not None
            and session.status == SessionStatus.RUNNING
            and session.execution_mode == "background"
            and session.execution_phase == "running"
            and session.background_reason == "auto_degrade"
        )

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
    def _generation_key(user_id: str) -> str:
        return f"supervisor:bg-generation:{user_id}"

    @staticmethod
    def _system_members_key() -> str:
        return "supervisor:system:bg-members"

    @staticmethod
    def _background_reconcile_marker_key() -> str:
        return "supervisor:system:bg-reconcile-pending"

    @staticmethod
    def _background_reconcile_due_key() -> str:
        return "supervisor:system:bg-reconcile-due"

    async def _find_legacy_projection_user(
        self,
        *,
        session_id: str,
        expected_generation: int,
    ) -> str | None:
        prefix = "supervisor:bg-generation:"
        candidates: list[str] = []
        async for raw_key in self._redis.scan_iter(match=f"{prefix}*"):
            key = self._decode_redis_value(raw_key)
            if key is None or not key.startswith(prefix):
                continue
            raw_generation = self._decode_redis_value(
                await self._redis.hget(key, session_id)
            )
            try:
                generation = int(raw_generation) if raw_generation is not None else None
            except ValueError:
                continue
            if generation == expected_generation:
                candidates.append(key[len(prefix):])
        unique_candidates = sorted(set(candidates))
        if len(unique_candidates) == 1:
            return unique_candidates[0]
        if len(unique_candidates) > 1:
            logger.warning(
                "ambiguous legacy background generation owners: "
                "session=%s generation=%s users=%s",
                session_id,
                expected_generation,
                unique_candidates,
            )
        return None

    async def _mark_background_reconcile_held(
        self,
        *,
        session_id: str,
        generation: int,
        user_id: str | None,
        require_generation: bool,
        now: datetime,
    ) -> int:
        marker = _BackgroundReconcileMarker(
            phase="held",
            generation=generation,
            next_check_at=now + timedelta(seconds=_BG_RECONCILE_SETTLE_SECONDS),
            gc_at=now + timedelta(seconds=_BG_RECONCILE_RETENTION_SECONDS),
            token=uuid.uuid4().hex,
            require_generation=require_generation,
            user_id=user_id,
        )
        result = await run_lua_with_fallback(
            self._redis,
            source=LUA_MARK_BACKGROUND_RECONCILE_HELD,
            sha=(
                self._sha_mark_background_reconcile_held
                or LUA_MARK_BACKGROUND_RECONCILE_HELD_SHA
            ),
            keys=[
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                generation,
                self._encode_background_reconcile_marker(marker),
                f"{marker.next_check_at.timestamp():.6f}",
            ],
            meter=self._meter,
        )
        return int(result)

    async def _scan_background_reconcile_markers(
        self,
    ) -> list[tuple[str, str, _BackgroundReconcileMarker]]:
        cursor: int | str | bytes = 0
        markers: list[tuple[str, str, _BackgroundReconcileMarker]] = []
        while True:
            cursor, page = await self._redis.hscan(
                self._background_reconcile_marker_key(),
                cursor=cursor,
                count=100,
            )
            for raw_session_id, raw_value in page.items():
                session_id = self._decode_redis_value(raw_session_id)
                marker_raw = self._decode_redis_value(raw_value)
                marker = self._decode_background_reconcile_marker(marker_raw)
                if session_id is None or marker_raw is None or marker is None:
                    logger.warning(
                        "invalid background reconcile marker: session=%r value=%r",
                        raw_session_id,
                        raw_value,
                    )
                    continue
                markers.append((session_id, marker_raw, marker))
            if int(cursor) == 0:
                break
        return markers

    async def _reconcile_background_markers(
        self,
        *,
        now: datetime,
    ) -> dict[str, int]:
        cleaned = 0
        repaired = 0
        skipped = 0
        for session_id, scanned_raw, scanned_marker in (
            await self._scan_background_reconcile_markers()
        ):
            try:
                async with self.mode_transition_fence(session_id=session_id):
                    marker_raw = self._decode_redis_value(
                        await self._redis.hget(
                            self._background_reconcile_marker_key(), session_id
                        )
                    )
                    marker = self._decode_background_reconcile_marker(marker_raw)
                    if marker_raw != scanned_raw or marker != scanned_marker:
                        continue
                    membership = self._decode_system_membership(
                        await self._redis.hget(
                            self._system_members_key(), session_id
                        )
                    )
                    if membership is not None and membership[0] > marker.generation:
                        await self._gc_background_reconcile_marker(
                            session_id=session_id,
                            expected_marker=marker_raw,
                        )
                        continue
                    current = await self._read_session_snapshot(session_id)
                    if (
                        current is not None
                        and current.execution_revision < marker.generation
                    ):
                        continue
                    is_live_background = bool(
                        current is not None
                        and current.status == SessionStatus.RUNNING
                        and current.execution_mode == "background"
                        and current.execution_phase in ("running", "suspended")
                        and current.expires_at is not None
                        and current.user_id is not None
                    )
                    if is_live_background:
                        assert current is not None
                        restore_rc = await self._restore_background_from_marker(
                            session_id=session_id,
                            user_id=str(current.user_id),
                            expires_at=self._as_utc(current.expires_at),
                            generation=current.execution_revision,
                            expected_marker=marker_raw,
                        )
                        if restore_rc > 0:
                            repaired += 1
                        elif restore_rc < 0:
                            await self._gc_background_reconcile_marker(
                                session_id=session_id,
                                expected_marker=marker_raw,
                            )
                            skipped += 1
                        continue
                    if marker.phase == "held":
                        if now < marker.next_check_at:
                            continue
                        release_rc = await self._release_held_background_reconcile(
                            session_id=session_id,
                            marker=marker,
                            expected_marker=marker_raw,
                            now=now,
                            allow_missing_membership=membership is None,
                        )
                        if release_rc == 1:
                            cleaned += 1
                        elif release_rc < 0:
                            skipped += 1
                            if now >= marker.gc_at:
                                await self._gc_background_reconcile_marker(
                                    session_id=session_id,
                                    expected_marker=marker_raw,
                                )
                        continue
                    if now >= marker.gc_at:
                        await self._gc_background_reconcile_marker(
                            session_id=session_id,
                            expected_marker=marker_raw,
                        )
            except Exception:
                skipped += 1
                logger.exception(
                    "background reconcile marker failed: session=%s",
                    session_id,
                )
        return {"cleaned": cleaned, "repaired": repaired, "skipped": skipped}

    async def _release_held_background_reconcile(
        self,
        *,
        session_id: str,
        marker: _BackgroundReconcileMarker,
        expected_marker: str,
        now: datetime,
        allow_missing_membership: bool,
    ) -> int:
        released = self._released_background_reconcile_marker(marker, now=now)
        user_id = marker.user_id or "__global_only__"
        result = await run_lua_with_fallback(
            self._redis,
            source=LUA_RELEASE_HELD_BACKGROUND_RECONCILE,
            sha=(
                self._sha_release_held_background_reconcile
                or LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SHA
            ),
            keys=[
                self._system_key(),
                self._user_key(user_id),
                self._bg_key(user_id),
                self._generation_key(user_id),
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                marker.generation,
                expected_marker,
                self._encode_background_reconcile_marker(released),
                f"{released.next_check_at.timestamp():.6f}",
                1 if marker.user_id is not None else 0,
                1 if marker.require_generation else 0,
                1 if allow_missing_membership else 0,
            ],
            meter=self._meter,
        )
        return int(result)

    async def _restore_background_from_marker(
        self,
        *,
        session_id: str,
        user_id: str,
        expires_at: datetime,
        generation: int,
        expected_marker: str,
    ) -> int:
        expiry = self._as_utc(expires_at)
        slot_ttl = self._background_slot_ttl_seconds(expiry)
        result = await run_lua_with_fallback(
            self._redis,
            source=LUA_RESTORE_BACKGROUND_FROM_MARKER,
            sha=(
                self._sha_restore_background_from_marker
                or LUA_RESTORE_BACKGROUND_FROM_MARKER_SHA
            ),
            keys=[
                self._system_key(),
                self._user_key(user_id),
                self._bg_key(user_id),
                self._generation_key(user_id),
                self._system_members_key(),
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[
                session_id,
                f"{expiry.timestamp():.6f}",
                slot_ttl,
                generation,
                self._encode_system_membership(user_id, generation),
                expected_marker,
            ],
            meter=self._meter,
        )
        return int(result)

    async def _gc_background_reconcile_marker(
        self,
        *,
        session_id: str,
        expected_marker: str,
    ) -> int:
        result = await run_lua_with_fallback(
            self._redis,
            source=LUA_GC_BACKGROUND_RECONCILE_MARKER,
            sha=(
                self._sha_gc_background_reconcile_marker
                or LUA_GC_BACKGROUND_RECONCILE_MARKER_SHA
            ),
            keys=[
                self._background_reconcile_marker_key(),
                self._background_reconcile_due_key(),
            ],
            args=[session_id, expected_marker],
            meter=self._meter,
        )
        return int(result)

    async def _scan_global_background_memberships(
        self,
    ) -> list[tuple[str, str | None, int]]:
        cursor: int | str | bytes = 0
        memberships: list[tuple[str, str | None, int]] = []
        while True:
            cursor, page = await self._redis.hscan(
                self._system_members_key(),
                cursor=cursor,
                count=100,
            )
            for raw_session_id, raw_value in page.items():
                decoded = self._decode_system_membership(raw_value)
                if decoded is None:
                    logger.warning(
                        "invalid global background membership: session=%r value=%r",
                        raw_session_id,
                        raw_value,
                    )
                    continue
                session_id = self._decode_redis_value(raw_session_id)
                if session_id is None:
                    continue
                generation, user_id = decoded
                memberships.append((session_id, user_id, generation))
            if int(cursor) == 0:
                break
        return memberships

    def _background_slot_ttl_seconds(self, expires_at: datetime) -> int:
        expiry = self._as_utc(expires_at)
        if expiry is None:
            return _BG_SLOT_CLEANUP_GRACE_SECONDS
        remaining = max((expiry - self._now_utc()).total_seconds(), 0.0)
        return max(
            math.ceil(remaining) + _BG_SLOT_CLEANUP_GRACE_SECONDS,
            _BG_SLOT_CLEANUP_GRACE_SECONDS,
        )

    @staticmethod
    def _encode_system_membership(user_id: str, generation: int) -> str:
        return f"v1|{generation}|{user_id}"

    @classmethod
    def _decode_system_membership(
        cls,
        value: object,
    ) -> tuple[int, str | None] | None:
        decoded = cls._decode_redis_value(value)
        if decoded is None:
            return None
        if decoded.startswith("v1|"):
            parts = decoded.split("|", 2)
            if len(parts) != 3:
                return None
            try:
                return int(parts[1]), parts[2] or None
            except ValueError:
                return None
        try:
            return int(decoded), None
        except ValueError:
            return None

    @staticmethod
    def _encode_background_reconcile_marker(
        marker: _BackgroundReconcileMarker,
    ) -> str:
        return "|".join(
            (
                "v1",
                marker.phase,
                str(marker.generation),
                f"{marker.next_check_at.timestamp():.6f}",
                f"{marker.gc_at.timestamp():.6f}",
                marker.token,
                "1" if marker.require_generation else "0",
                marker.user_id or "",
            )
        )

    @staticmethod
    def _decode_background_reconcile_marker(
        value: object,
    ) -> _BackgroundReconcileMarker | None:
        decoded = ExecutionSupervisor._decode_redis_value(value)
        if decoded is None:
            return None
        parts = decoded.split("|", 7)
        if len(parts) != 8 or parts[0] != "v1" or parts[1] not in (
            "held",
            "released",
        ):
            return None
        try:
            generation = int(parts[2])
            next_check_at = datetime.fromtimestamp(float(parts[3]), timezone.utc)
            gc_at = datetime.fromtimestamp(float(parts[4]), timezone.utc)
        except (ValueError, OverflowError, OSError):
            return None
        if not parts[5] or parts[6] not in ("0", "1"):
            return None
        return _BackgroundReconcileMarker(
            phase=parts[1],
            generation=generation,
            next_check_at=next_check_at,
            gc_at=gc_at,
            token=parts[5],
            require_generation=parts[6] == "1",
            user_id=parts[7] or None,
        )

    @staticmethod
    def _released_background_reconcile_marker(
        marker: _BackgroundReconcileMarker,
        *,
        now: datetime,
    ) -> _BackgroundReconcileMarker:
        return _BackgroundReconcileMarker(
            phase="released",
            generation=marker.generation,
            next_check_at=now,
            gc_at=now + timedelta(seconds=_BG_RECONCILE_RETENTION_SECONDS),
            token=marker.token,
            require_generation=marker.require_generation,
            user_id=marker.user_id,
        )

    @staticmethod
    def _decode_redis_value(value: object) -> str | None:
        if value is None:
            return None
        if isinstance(value, (bytes, bytearray)):
            return value.decode()
        return str(value)

    @classmethod
    def _parse_count(cls, value: object) -> int:
        decoded = cls._decode_redis_value(value)
        if decoded is None:
            return 0
        try:
            return max(int(decoded), 0)
        except (TypeError, ValueError):
            return 0
