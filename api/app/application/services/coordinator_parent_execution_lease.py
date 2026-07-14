"""Run-scoped, phase-aware parent execution lease.

This application-layer service keeps the parent execution owner alive while
durably healthy coordinator children are running, and while the backend owns
reduction/apply/rollback work.  It is deliberately not a task deadline.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Awaitable, Callable, Protocol, Sequence

from app.application.services.coordinator_liveness_lease_service import (
    CoordinatorChildLease,
)

logger = logging.getLogger(__name__)


class CoordinatorParentPhase(StrEnum):
    WAITING_CHILDREN = "waiting_children"
    REDUCING = "reducing"
    APPLYING = "applying"
    ROLLBACK = "rollback"


class _ChildLivenessState(StrEnum):
    FRESH = "fresh"
    ALL_STALE = "all_stale"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CoordinatorParentLeaseContext:
    root_session_id: str
    parent_session_id: str
    user_id: str | None
    coordinator_run_id: str
    step_id: str
    child_session_ids: tuple[str, ...]
    phase: CoordinatorParentPhase
    observed_monotonic: float


class CoordinatorLivenessReader(Protocol):
    async def get_lease(
        self, child_session_id: str,
    ) -> CoordinatorChildLease | None: ...

    def is_stale(self, lease: CoordinatorChildLease | None) -> bool: ...


ParentLeaseCallback = Callable[
    [CoordinatorParentLeaseContext], Awaitable[None]
]
OwnerAlive = Callable[[], bool]


def _current_task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class CoordinatorParentExecutionLease:
    """Idempotent lifecycle handle for one ``(step_id, run_id)`` lease."""

    def __init__(
        self,
        *,
        root_session_id: str,
        parent_session_id: str,
        user_id: str | None = None,
        coordinator_run_id: str,
        step_id: str,
        child_session_ids: Sequence[str],
        liveness_service: CoordinatorLivenessReader,
        owner_alive: OwnerAlive,
        phase: CoordinatorParentPhase = CoordinatorParentPhase.WAITING_CHILDREN,
        interval_seconds: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        touch_parent_activity: ParentLeaseCallback | None = None,
        renew_auto_degrade: ParentLeaseCallback | None = None,
        renew_parent_sandbox: ParentLeaseCallback | None = None,
        renew_quota: ParentLeaseCallback | None = None,
        on_all_children_stale: ParentLeaseCallback | None = None,
    ) -> None:
        identifiers = {
            "root_session_id": root_session_id,
            "parent_session_id": parent_session_id,
            "coordinator_run_id": coordinator_run_id,
            "step_id": step_id,
        }
        if not all(
            isinstance(value, str) and value.strip()
            for value in identifiers.values()
        ):
            raise ValueError("parent lease identifiers must be non-empty strings")
        if user_id is not None and (
            not isinstance(user_id, str) or not user_id.strip()
        ):
            raise ValueError("user_id must be a non-empty string when provided")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("interval_seconds must be finite and positive")
        child_ids = tuple(child_session_ids)
        if not child_ids or not all(
            isinstance(child_id, str) and child_id.strip()
            for child_id in child_ids
        ):
            raise ValueError("child_session_ids must contain non-empty strings")
        if len(set(child_ids)) != len(child_ids):
            raise ValueError("child_session_ids must be unique")
        initial_clock = float(clock())
        if not math.isfinite(initial_clock):
            raise ValueError("clock must return a finite value")

        self._root_session_id = root_session_id
        self._parent_session_id = parent_session_id
        self._user_id = user_id
        self._coordinator_run_id = coordinator_run_id
        self._step_id = step_id
        self._child_session_ids = child_ids
        self._liveness = liveness_service
        self._owner_alive = owner_alive
        self._phase = CoordinatorParentPhase(phase)
        self._interval_seconds = float(interval_seconds)
        self._clock = clock
        self._sleep = sleep
        self._renew_callbacks = tuple(
            callback
            for callback in (
                touch_parent_activity,
                renew_auto_degrade,
                renew_parent_sandbox,
                renew_quota,
            )
            if callback is not None
        )
        self._on_all_children_stale = on_all_children_stale
        self._all_stale_notified = False
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def phase(self) -> CoordinatorParentPhase:
        return self._phase

    @property
    def done(self) -> bool:
        return self._task is not None and self._task.done()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(
            self._run(),
            name=(
                "coordinator-parent-lease:"
                f"{self._step_id}:{self._coordinator_run_id}"
            ),
        )

    def set_phase(self, phase: CoordinatorParentPhase) -> None:
        if self._stop_event.is_set():
            return
        self._phase = CoordinatorParentPhase(phase)

    def stop(self) -> None:
        """Cancel shutdown without blocking; ``drain`` owns the await."""
        self._stop_event.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def drain(self) -> None:
        task = self._task
        if task is None:
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                # This is cancellation of the caller doing the drain, not just
                # the background loop's own structured cancellation. Ensure
                # the lease is gone, then preserve the caller's signal.
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            # The background loop itself ended via callback/stop cancellation.
            # It is already fully drained and must not cancel its owner.

    def _context(self) -> CoordinatorParentLeaseContext:
        observed = float(self._clock())
        if not math.isfinite(observed):
            raise ValueError("clock must continue returning finite values")
        return CoordinatorParentLeaseContext(
            root_session_id=self._root_session_id,
            parent_session_id=self._parent_session_id,
            user_id=self._user_id,
            coordinator_run_id=self._coordinator_run_id,
            step_id=self._step_id,
            child_session_ids=self._child_session_ids,
            phase=self._phase,
            observed_monotonic=observed,
        )

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            phase = self._phase
            if phase is CoordinatorParentPhase.WAITING_CHILDREN:
                liveness_state = await self._read_child_liveness_state()
                should_renew = liveness_state is _ChildLivenessState.FRESH
                if liveness_state is _ChildLivenessState.ALL_STALE:
                    await self._notify_all_children_stale_once()
            else:
                should_renew = bool(self._owner_alive())
                if not should_renew:
                    return

            if should_renew:
                await self._renew_once()
            await self._sleep_or_stop()

    async def _read_child_liveness_state(self) -> _ChildLivenessState:
        read_failed = False
        for child_id in self._child_session_ids:
            try:
                lease = await self._liveness.get_lease(child_id)
                is_stale = self._liveness.is_stale(lease)
            except asyncio.CancelledError:
                if _current_task_is_cancelling():
                    raise
                read_failed = True
                logger.warning(
                    "coordinator parent lease liveness read failed "
                    "step=%s run=%s child=%s; tick state is unknown",
                    self._step_id,
                    self._coordinator_run_id,
                    child_id,
                    exc_info=True,
                )
                continue
            except Exception:
                read_failed = True
                logger.warning(
                    "coordinator parent lease liveness read failed "
                    "step=%s run=%s child=%s; tick state is unknown",
                    self._step_id,
                    self._coordinator_run_id,
                    child_id,
                    exc_info=True,
                )
                continue
            if not is_stale:
                return _ChildLivenessState.FRESH
        if read_failed:
            return _ChildLivenessState.UNKNOWN
        return _ChildLivenessState.ALL_STALE

    async def _notify_all_children_stale_once(self) -> None:
        if self._all_stale_notified or self._on_all_children_stale is None:
            return
        succeeded = await self._run_callback(
            self._on_all_children_stale,
            self._context(),
            kind="all_children_stale",
        )
        if succeeded:
            self._all_stale_notified = True

    async def _renew_once(self) -> None:
        context = self._context()
        for callback in self._renew_callbacks:
            await self._run_callback(callback, context, kind="renew")

    async def _run_callback(
        self,
        callback: ParentLeaseCallback,
        context: CoordinatorParentLeaseContext,
        *,
        kind: str,
    ) -> bool:
        try:
            await callback(context)
        except asyncio.CancelledError:
            if _current_task_is_cancelling():
                raise
            logger.warning(
                "coordinator parent lease callback failed kind=%s "
                "step=%s run=%s phase=%s callback=%r",
                kind,
                self._step_id,
                self._coordinator_run_id,
                context.phase.value,
                callback,
                exc_info=True,
            )
            return False
        except Exception:
            logger.warning(
                "coordinator parent lease callback failed kind=%s "
                "step=%s run=%s phase=%s callback=%r",
                kind,
                self._step_id,
                self._coordinator_run_id,
                context.phase.value,
                callback,
                exc_info=True,
            )
            return False
        return True

    async def _sleep_or_stop(self) -> None:
        sleep_task = asyncio.create_task(self._sleep(self._interval_seconds))
        stop_task = asyncio.create_task(self._stop_event.wait())
        tasks = (sleep_task, stop_task)
        try:
            _done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


CoordinatorParentExecutionLeaseFactory = Callable[
    ...,
    CoordinatorParentExecutionLease,
]
