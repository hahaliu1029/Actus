"""Structured parent-wait lifecycle guard for coordinator runs."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from typing import Awaitable, AsyncIterator, Callable, Hashable, Protocol, Sequence

from app.application.services.coordinator_parent_execution_lease import (
    CoordinatorParentPhase,
)

logger = logging.getLogger(__name__)


def _current_task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


class _IdleWatchdog(Protocol):
    def pause_idle(self, key: Hashable) -> None: ...

    def resume_idle(self, key: Hashable) -> None: ...


class _ParentLeaseHandle(Protocol):
    def start(self) -> None: ...

    def set_phase(self, phase: CoordinatorParentPhase) -> None: ...

    def stop(self) -> None: ...

    async def drain(self) -> None: ...


@dataclass
class _GuardedRun:
    pause_key: Hashable
    lease: _ParentLeaseHandle | None
    release_quota: Callable[[], Awaitable[None]] | None


class CoordinatorWaitGuard:
    """Own idle-pause keys by coordinator step and run identifiers.

    The guard deliberately does not inspect graph events. Dispatch owns
    ``enter_run`` and terminal/liveness handling owns ``resume_run``; the
    enclosing step scope is the exception/cancellation cleanup backstop.
    """

    def __init__(
        self,
        *,
        watchdog: _IdleWatchdog | None = None,
        parent_lease_factory: Callable[..., object] | None = None,
    ) -> None:
        self.watchdog = watchdog
        self._parent_lease_factory = parent_lease_factory
        self._runs_by_step: dict[str, dict[str, _GuardedRun]] = {}
        self._step_owners: dict[str, asyncio.Task[object]] = {}
        self._step_scope_depth: dict[str, int] = {}

    @staticmethod
    def _pause_key(step_id: str, run_id: str) -> tuple[str, str, str]:
        return ("coordinator-wait", step_id, run_id)

    @asynccontextmanager
    async def step_scope(self, step_id: str) -> AsyncIterator[None]:
        owner = asyncio.current_task()
        if owner is None:
            raise RuntimeError("coordinator step scope requires an asyncio task")
        depth = self._step_scope_depth.get(step_id, 0)
        if depth == 0:
            self._step_owners[step_id] = owner
        elif self._step_owners.get(step_id) is not owner:
            raise RuntimeError(
                f"coordinator step scope {step_id!r} is already owned by "
                "another asyncio task"
            )
        self._step_scope_depth[step_id] = depth + 1
        try:
            yield
        finally:
            remaining = self._step_scope_depth.get(step_id, 1) - 1
            if remaining > 0:
                self._step_scope_depth[step_id] = remaining
            else:
                self._step_scope_depth.pop(step_id, None)
                try:
                    await self.resume_all_for_step(step_id)
                finally:
                    self._step_owners.pop(step_id, None)

    async def enter_run(
        self,
        step_id: str,
        run_id: str,
        *,
        root_session_id: str | None = None,
        parent_session_id: str | None = None,
        user_id: str | None = None,
        child_session_ids: Sequence[str] = (),
        phase: CoordinatorParentPhase = CoordinatorParentPhase.WAITING_CHILDREN,
        release_quota: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        step_runs = self._runs_by_step.setdefault(step_id, {})
        if run_id in step_runs:
            return
        key = self._pause_key(step_id, run_id)
        lease: _ParentLeaseHandle | None = None
        pause_attempted = False
        start_attempted = False
        try:
            if self._parent_lease_factory is not None:
                if root_session_id is None or parent_session_id is None:
                    raise ValueError(
                        "parent lease factory requires root_session_id and "
                        "parent_session_id"
                    )
                owner = self._step_owners.get(step_id) or asyncio.current_task()
                if owner is None:
                    raise RuntimeError(
                        "parent lease requires an asyncio owner task"
                    )
                lease = self._parent_lease_factory(
                    root_session_id=root_session_id,
                    parent_session_id=parent_session_id,
                    user_id=user_id,
                    coordinator_run_id=run_id,
                    step_id=step_id,
                    child_session_ids=tuple(child_session_ids),
                    owner_alive=lambda owner=owner: not owner.done(),
                    phase=phase,
                )

            if self.watchdog is not None:
                # Set before the call: a watchdog implementation may mutate
                # its key set and then raise. ``resume_idle`` is idempotent.
                pause_attempted = True
                self.watchdog.pause_idle(key)

            if lease is not None:
                start_attempted = True
                lease.start()

            # Publish only after every preceding ownership step succeeded.
            step_runs[run_id] = _GuardedRun(
                pause_key=key,
                lease=lease,
                release_quota=release_quota,
            )
        except BaseException:
            cancelled: asyncio.CancelledError | None = None
            if lease is not None and start_attempted:
                try:
                    lease.stop()
                except asyncio.CancelledError as exc:
                    if _current_task_is_cancelling():
                        cancelled = exc
                    else:
                        logger.warning(
                            "coordinator parent lease rollback stop failed "
                            "step=%s run=%s",
                            step_id,
                            run_id,
                            exc_info=True,
                        )
                except BaseException:
                    logger.warning(
                        "coordinator parent lease rollback stop failed "
                        "step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
                try:
                    await lease.drain()
                except asyncio.CancelledError as exc:
                    if _current_task_is_cancelling():
                        cancelled = exc
                    else:
                        logger.warning(
                            "coordinator parent lease rollback drain failed "
                            "step=%s run=%s",
                            step_id,
                            run_id,
                            exc_info=True,
                        )
                    # The first drain await was interrupted. Retry after the
                    # cancellation has been observed so the lease can finish
                    # its own structured shutdown before we propagate.
                    try:
                        await lease.drain()
                    except asyncio.CancelledError as retry_exc:
                        if _current_task_is_cancelling():
                            cancelled = cancelled or retry_exc
                        else:
                            logger.warning(
                                "coordinator parent lease rollback drain "
                                "retry failed step=%s run=%s",
                                step_id,
                                run_id,
                                exc_info=True,
                            )
                    except BaseException:
                        logger.warning(
                            "coordinator parent lease rollback drain retry "
                            "failed step=%s run=%s",
                            step_id,
                            run_id,
                            exc_info=True,
                        )
                except BaseException:
                    logger.warning(
                        "coordinator parent lease rollback drain failed "
                        "step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            if pause_attempted and self.watchdog is not None:
                try:
                    self.watchdog.resume_idle(key)
                except asyncio.CancelledError as exc:
                    if _current_task_is_cancelling():
                        cancelled = cancelled or exc
                    else:
                        logger.warning(
                            "coordinator wait rollback resume failed "
                            "step=%s run=%s",
                            step_id,
                            run_id,
                            exc_info=True,
                        )
                except BaseException:
                    logger.warning(
                        "coordinator wait rollback resume failed "
                        "step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            if not step_runs:
                self._runs_by_step.pop(step_id, None)
            if cancelled is not None:
                raise cancelled
            raise

    def set_phase(
        self,
        step_id: str,
        run_id: str,
        phase: CoordinatorParentPhase,
    ) -> None:
        guarded = self._runs_by_step.get(step_id, {}).get(run_id)
        if guarded is None or guarded.lease is None:
            return
        guarded.lease.set_phase(phase)

    def owns_quota_release(self, step_id: str, run_id: str) -> bool:
        """Return whether this guard owns the run's exact quota release."""
        guarded = self._runs_by_step.get(step_id, {}).get(run_id)
        return guarded is not None and guarded.release_quota is not None

    async def resume_run(self, step_id: str, run_id: str) -> bool:
        """Stop and drain one run.

        Returns ``True`` when the popped guard owned a quota-release callback,
        even if that best-effort callback failed. Callers use this ownership
        signal to avoid a second exact release after transfer.
        """
        step_runs = self._runs_by_step.get(step_id)
        if not step_runs or run_id not in step_runs:
            return False
        guarded = step_runs.pop(run_id)
        if not step_runs:
            self._runs_by_step.pop(step_id, None)
        cancelled = await self._stop_and_drain(
            guarded, step_id=step_id, run_id=run_id,
        )
        if cancelled is not None:
            raise cancelled
        return guarded.release_quota is not None

    async def resume_all_for_step(self, step_id: str) -> None:
        step_runs = self._runs_by_step.pop(step_id, None)
        if not step_runs:
            return
        cancelled: asyncio.CancelledError | None = None
        for run_id, guarded in step_runs.items():
            run_cancelled = await self._stop_and_drain(
                guarded, step_id=step_id, run_id=run_id,
            )
            if run_cancelled is not None:
                # Every run still gets stopped/drained and its watchdog key
                # resumed before the caller's cancellation is re-propagated.
                cancelled = run_cancelled
        if cancelled is not None:
            raise cancelled

    async def _stop_and_drain(
        self,
        guarded: _GuardedRun,
        *,
        step_id: str,
        run_id: str,
    ) -> asyncio.CancelledError | None:
        cancelled: asyncio.CancelledError | None = None
        if guarded.lease is not None:
            try:
                guarded.lease.stop()
            except asyncio.CancelledError as exc:
                if _current_task_is_cancelling():
                    cancelled = exc
                else:
                    logger.warning(
                        "coordinator parent lease stop failed step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            except Exception:
                logger.warning(
                    "coordinator parent lease stop failed step=%s run=%s",
                    step_id,
                    run_id,
                    exc_info=True,
                )
            try:
                await guarded.lease.drain()
            except asyncio.CancelledError as exc:
                if _current_task_is_cancelling():
                    cancelled = exc
                else:
                    logger.warning(
                        "coordinator parent lease drain failed step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            except Exception:  # cleanup must preserve backend outcome
                logger.warning(
                    "coordinator parent lease drain failed step=%s run=%s",
                    step_id,
                    run_id,
                    exc_info=True,
                )
        if guarded.release_quota is not None:
            try:
                await guarded.release_quota()
            except asyncio.CancelledError as exc:
                if _current_task_is_cancelling():
                    cancelled = cancelled or exc
                else:
                    logger.warning(
                        "coordinator quota release failed step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            except Exception:
                logger.warning(
                    "coordinator quota release failed step=%s run=%s",
                    step_id,
                    run_id,
                    exc_info=True,
                )
        if self.watchdog is not None:
            try:
                self.watchdog.resume_idle(guarded.pause_key)
            except asyncio.CancelledError as exc:
                if _current_task_is_cancelling():
                    cancelled = exc
                else:
                    logger.warning(
                        "coordinator wait resume failed step=%s run=%s",
                        step_id,
                        run_id,
                        exc_info=True,
                    )
            except Exception:
                logger.warning(
                    "coordinator wait resume failed step=%s run=%s",
                    step_id,
                    run_id,
                    exc_info=True,
                )
        return cancelled
