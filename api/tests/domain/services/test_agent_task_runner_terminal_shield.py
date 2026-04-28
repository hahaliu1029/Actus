"""B4 Issue 1D: _set_terminal_status whole-body shield contract.

Spec: docs/superpowers/specs/2026-04-27-b4-1d-finishing-drain-design.md §3.3, §5

Verifies:
1. Happy path — drain succeeds, status writes, completion callback fires,
   NO marker written.
2. Outer cancel during happy drain — caller sees CancelledError, shielded
   body completes (status row written, callback fired, no marker).
3. Drain timeout WITH healthy marker persister — status written + marker
   present + outer returns normally.
4. Outer cancel + drain timeout — outer raises CancelledError; marker
   present (drain drove it, not the cancel).
5. Status write uses fresh UoW (different instance from runner._uow).
6. Status commit failure observable via done callback (ERROR + traceback).
7. Terminal task observation: unexpected raise → done callback logs ERROR.
8. Registry leak guard: happy path → finished task removed from registry.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.agent_task_runner import (
    _PENDING_TERMINAL_TASKS,
    AgentTaskRunner,
)
from app.domain.services.cost_callback_handler import (
    CostCallbackHandler,
    FlushResult,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# Many tests construct AgentTaskRunner without going through the full
# __init__ to avoid pulling in the dozens of collaborators it normally
# requires. We bypass init via object.__new__ and assign only the
# attributes the terminal-status path reads.

def _make_runner_for_terminal(
    cost_handler: Any | None,
    uow_factory: Any,
    on_session_complete: Any | None = None,
    session_id: str = "sess-T",
) -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = session_id
    runner._cost_callback_handler = cost_handler
    runner._uow_factory = uow_factory
    runner._uow = MagicMock(name="runner._uow")  # marker for "stale UoW"
    runner._on_session_complete = on_session_complete
    return runner


def _make_uow_factory(commit_succeeds: bool = True) -> Any:
    """Yields a MagicMock-backed UoW that supports ``async with`` + commit."""
    yielded_uows: List[Any] = []

    @asynccontextmanager
    async def _ctx():
        uow = MagicMock(name=f"fresh-uow-{len(yielded_uows)}")
        uow.session = MagicMock()
        uow.session.update_status = AsyncMock(return_value=None)
        uow.db_session = MagicMock()
        if commit_succeeds:
            uow.db_session.commit = AsyncMock(return_value=None)
        else:
            uow.db_session.commit = AsyncMock(
                side_effect=RuntimeError("simulated commit failure")
            )
        yielded_uows.append(uow)
        yield uow

    factory = MagicMock(side_effect=_ctx)
    factory.yielded_uows = yielded_uows  # type: ignore[attr-defined]
    return factory


class TestHappyPath:
    async def test_drain_success_writes_status_no_marker(self) -> None:
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock()
        callback_seen: List[str] = []

        async def on_complete(session_id: str) -> None:
            callback_seen.append(session_id)

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(
            cost_handler, factory, on_session_complete=on_complete
        )

        await runner._set_terminal_status(SessionStatus.COMPLETED)

        cost_handler.flush_pending.assert_awaited_once_with(timeout=3.0)
        cost_handler.write_session_degraded_marker.assert_not_awaited()
        # Status write hit a fresh UoW with explicit commit.
        assert len(factory.yielded_uows) == 1
        factory.yielded_uows[0].session.update_status.assert_awaited_once()
        factory.yielded_uows[0].db_session.commit.assert_awaited_once()
        # Completion callback fired with the session id.
        assert callback_seen == ["sess-T"]


class TestOuterCancel:
    async def test_outer_cancel_during_happy_drain_body_completes(self) -> None:
        """Spec §3.5 row "Outer CancelledError mid-drain":
        outer raises CancelledError, shielded body still completes,
        no marker written (drain succeeded).
        """
        from app.domain.models.session import SessionStatus

        # Block flush_pending so we can cancel the caller mid-drain.
        flush_started = asyncio.Event()
        flush_release = asyncio.Event()

        async def slow_flush(timeout: float) -> FlushResult:
            del timeout  # mock receives the kwarg but doesn't use it
            flush_started.set()
            await flush_release.wait()
            return FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(side_effect=slow_flush)
        cost_handler.write_session_degraded_marker = AsyncMock()

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        # Schedule the call as a task so we can cancel it mid-drain.
        caller_task = asyncio.create_task(
            runner._set_terminal_status(SessionStatus.COMPLETED)
        )
        await flush_started.wait()
        caller_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await caller_task

        # Release the inner flush — terminal body should complete in bg.
        flush_release.set()
        # Wait for terminal task to settle.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if not _PENDING_TERMINAL_TASKS:
                break

        # Body completed: status write hit the factory once, marker NOT.
        assert len(factory.yielded_uows) == 1
        factory.yielded_uows[0].db_session.commit.assert_awaited_once()
        cost_handler.write_session_degraded_marker.assert_not_awaited()

    async def test_outer_cancel_with_drain_timeout_marker_present(self) -> None:
        """Spec §3.5: outer cancel during a drain that ALSO times out →
        outer sees CancelledError, marker present (drain drove it, not cancel).
        """
        from app.domain.models.session import SessionStatus

        flush_started = asyncio.Event()
        flush_release = asyncio.Event()

        async def slow_timeout_flush(timeout: float) -> FlushResult:
            del timeout  # mock receives the kwarg but doesn't use it
            flush_started.set()
            await flush_release.wait()
            return FlushResult(
                drained=False, pending_count=1, persist_failures=0
            )

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(side_effect=slow_timeout_flush)
        cost_handler.write_session_degraded_marker = AsyncMock(return_value=True)

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        caller_task = asyncio.create_task(
            runner._set_terminal_status(SessionStatus.COMPLETED)
        )
        await flush_started.wait()
        caller_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller_task
        # Release flush; shielded body finishes off in background.
        flush_release.set()
        for _ in range(50):
            await asyncio.sleep(0.01)
            if not _PENDING_TERMINAL_TASKS:
                break

        # Marker WAS written (drain timeout drove it, not the outer cancel).
        cost_handler.write_session_degraded_marker.assert_awaited_once_with(
            reason="drain_timeout"
        )


class TestDrainTimeout:
    async def test_drain_timeout_dispatches_marker_with_healthy_persister(
        self,
    ) -> None:
        """Spec §5: drain timeout WITH healthy marker persister →
        status written + marker dispatched + outer returns normally.

        Marker presence is gated on a successful persister (write returns
        True); see spec §2 best-effort goal + §8 ADR.
        """
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=False, pending_count=2, persist_failures=0
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock(return_value=True)

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        await runner._set_terminal_status(SessionStatus.COMPLETED)

        cost_handler.write_session_degraded_marker.assert_awaited_once_with(
            reason="drain_timeout"
        )
        # Status write still happened.
        assert len(factory.yielded_uows) == 1
        factory.yielded_uows[0].db_session.commit.assert_awaited_once()

    async def test_persist_failures_drives_marker_dispatch(self) -> None:
        """drained=True but persist_failures>0 → reason="drain_persist_failures"."""
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=True, pending_count=0, persist_failures=3
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock(return_value=True)

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        await runner._set_terminal_status(SessionStatus.COMPLETED)

        cost_handler.write_session_degraded_marker.assert_awaited_once_with(
            reason="drain_persist_failures"
        )

    async def test_cancelled_marker_persister_does_not_skip_status_write(
        self,
    ) -> None:
        """Cancellation guard regression (marker writer cancellation guard):

        If ``write_session_degraded_marker`` were to raise
        ``CancelledError`` on a cancelled marker task (the bug before the
        ``if task.cancelled():`` guard), the cancel would propagate out
        of ``_terminal_op`` BEFORE the fresh-UoW status write — leaving
        the session stuck in its prior status.

        With the guard, the marker writer returns ``False`` cleanly, and
        ``_terminal_op`` proceeds to the status write. This test pins
        that contract by simulating the writer returning ``False`` (the
        post-guard behavior) and asserting the status write still fires
        + the fresh UoW commit still runs.
        """
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=False, pending_count=1, persist_failures=0
            )
        )
        # The guard converts a cancelled marker task into a clean
        # ``return False``. This mock pins that downstream contract.
        cost_handler.write_session_degraded_marker = AsyncMock(return_value=False)

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        # Must NOT raise CancelledError — that was the bug.
        await runner._set_terminal_status(SessionStatus.COMPLETED)

        # Marker writer was called for the drain timeout.
        cost_handler.write_session_degraded_marker.assert_awaited_once_with(
            reason="drain_timeout"
        )
        # Status write still fired — fresh UoW + explicit commit.
        assert len(factory.yielded_uows) == 1, (
            "Status write must still happen when marker writer returns "
            "False — otherwise the session would be stuck in prior status."
        )
        factory.yielded_uows[0].session.update_status.assert_awaited_once()
        factory.yielded_uows[0].db_session.commit.assert_awaited_once()


class TestFreshUoW:
    async def test_status_write_uses_fresh_uow_not_runner_uow(self) -> None:
        """Spec §3.3: status write must use a UoW from self._uow_factory(),
        NOT self._uow. Happy path: drain succeeds, no marker write, so the
        only factory call attributable to terminal op is the status write.
        """
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock()

        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        # Sanity: runner._uow is the stale instance from setup.
        stale_uow = runner._uow

        await runner._set_terminal_status(SessionStatus.COMPLETED)

        # Exactly one fresh UoW yielded (happy path: only status-write).
        assert len(factory.yielded_uows) == 1
        fresh = factory.yielded_uows[0]
        assert fresh is not stale_uow, (
            "Status write must use a UoW from self._uow_factory(), not "
            "the stale self._uow."
        )
        fresh.session.update_status.assert_awaited_once_with(
            "sess-T", SessionStatus.COMPLETED
        )


class TestCommitFailure:
    async def test_status_commit_failure_logged_with_traceback(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Spec §3.5 + §5: explicit ``await uow.db_session.commit()`` raises
        on failure; ``_terminal_op`` propagates; done callback logs at ERROR
        with full traceback. Test captures via caplog and asserts
        record.exc_info is populated.
        """
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock()

        factory = _make_uow_factory(commit_succeeds=False)
        runner = _make_runner_for_terminal(cost_handler, factory)

        # ``asyncio.shield`` only protects against CANCELLATION; ordinary
        # exceptions raised inside the shielded task PROPAGATE to the
        # awaiter normally. So when the inner ``_terminal_op`` raises the
        # commit RuntimeError, ``await asyncio.shield(terminal_task)``
        # re-raises it to the caller. We must catch that explicitly here,
        # then check that the done callback ALSO observed and logged the
        # exception with full traceback.
        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError, match="simulated commit failure"):
                await runner._set_terminal_status(SessionStatus.COMPLETED)
            # Done callback fires as the task transitions to done. Wait
            # for the registry to clear so we know the callback ran.
            for _ in range(50):
                await asyncio.sleep(0.01)
                if not _PENDING_TERMINAL_TASKS:
                    break

        terminal_errors = [
            r for r in caplog.records
            if "terminal task" in r.getMessage() and "raised" in r.getMessage()
        ]
        assert len(terminal_errors) == 1
        rec = terminal_errors[0]
        assert rec.exc_info is not None
        assert rec.exc_info[0] is RuntimeError
        # The traceback object MUST be attached so operators can debug.
        assert rec.exc_info[2] is not None, (
            "Done callback must pass __traceback__ via the explicit "
            "(type, value, traceback) tuple form so log formatters render "
            "the full stack."
        )


class TestRegistryHygiene:
    async def test_happy_path_removes_task_from_registry(self) -> None:
        """Spec §5 'Registry leak guard': after happy-path completion, the
        finished task is removed from _PENDING_TERMINAL_TASKS by the done
        callback.
        """
        from app.domain.models.session import SessionStatus

        cost_handler = MagicMock(spec=CostCallbackHandler)
        cost_handler.flush_pending = AsyncMock(
            return_value=FlushResult(
                drained=True, pending_count=0, persist_failures=0
            )
        )
        cost_handler.write_session_degraded_marker = AsyncMock()
        factory = _make_uow_factory()
        runner = _make_runner_for_terminal(cost_handler, factory)

        registry_before = len(_PENDING_TERMINAL_TASKS)
        await runner._set_terminal_status(SessionStatus.COMPLETED)
        # Wait for done callback to clean up.
        for _ in range(50):
            await asyncio.sleep(0.01)
            if len(_PENDING_TERMINAL_TASKS) == registry_before:
                break

        assert len(_PENDING_TERMINAL_TASKS) == registry_before, (
            "Done callback must discard finished task; otherwise the "
            "registry grows unbounded over the process lifetime."
        )
