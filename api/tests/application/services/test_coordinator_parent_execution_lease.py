"""Run-scoped parent execution lease tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from app.application.services.coordinator_liveness_lease_service import (
    CoordinatorChildLease,
)
from app.application.services.coordinator_parent_execution_lease import (
    CoordinatorParentExecutionLease,
    CoordinatorParentPhase,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _lease(child_id: str, *, phase: str = "starting") -> CoordinatorChildLease:
    return CoordinatorChildLease(
        root_session_id="root",
        parent_session_id="parent",
        child_session_id=child_id,
        coordinator_run_id="run",
        work_unit_id=f"wu-{child_id}",
        last_seen_epoch=1.0,
        phase=phase,
        authority_age_seconds=0.0,
    )


class _Liveness:
    def __init__(self, leases: dict[str, CoordinatorChildLease | None]) -> None:
        self.leases = leases

    async def get_lease(self, child_id: str) -> CoordinatorChildLease | None:
        return self.leases.get(child_id)

    def is_stale(self, lease: CoordinatorChildLease | None) -> bool:
        return lease is None or lease.authority_age_seconds == 999.0


async def _wait_until(predicate, *, attempts: int = 100) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not reached")


async def test_waiting_renews_only_when_at_least_one_child_is_fresh() -> None:
    liveness = _Liveness({"fresh": _lease("fresh"), "missing": None})
    touched = AsyncMock()
    degraded = AsyncMock()
    sandbox = AsyncMock()
    quota = AsyncMock()
    all_stale = AsyncMock()
    release_tick = asyncio.Event()

    async def sleep(_: float) -> None:
        await release_tick.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh", "missing"),
        liveness_service=liveness,
        owner_alive=lambda: True,
        interval_seconds=15,
        sleep=sleep,
        touch_parent_activity=touched,
        renew_auto_degrade=degraded,
        renew_parent_sandbox=sandbox,
        renew_quota=quota,
        on_all_children_stale=all_stale,
    )

    handle.start()
    await _wait_until(lambda: touched.await_count == 1)
    assert degraded.await_count == sandbox.await_count == quota.await_count == 1
    all_stale.assert_not_awaited()

    liveness.leases["fresh"] = None
    release_tick.set()
    await _wait_until(lambda: all_stale.await_count == 1)
    assert touched.await_count == 1

    # The orphan callback is once-per-run even if later ticks remain stale.
    await asyncio.sleep(0)
    assert all_stale.await_count == 1
    handle.stop()
    await handle.drain()


@pytest.mark.parametrize(
    "phase",
    [
        CoordinatorParentPhase.REDUCING,
        CoordinatorParentPhase.APPLYING,
        CoordinatorParentPhase.ROLLBACK,
    ],
)
async def test_backend_phases_renew_past_three_hours_until_owner_done(
    phase: CoordinatorParentPhase,
) -> None:
    ticks = 0
    touched = AsyncMock()
    owner_alive = True

    class FakeClock:
        value = 0.0

        def __call__(self) -> float:
            return self.value

    clock = FakeClock()

    async def sleep(seconds: float) -> None:
        nonlocal ticks, owner_alive
        assert seconds == 900
        ticks += 1
        clock.value += seconds
        if clock.value > 3 * 60 * 60:
            owner_alive = False
        await asyncio.sleep(0)

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("terminal",),
        liveness_service=_Liveness({"terminal": None}),
        owner_alive=lambda: owner_alive,
        phase=phase,
        interval_seconds=900,
        clock=clock,
        sleep=sleep,
        touch_parent_activity=touched,
    )

    handle.start()
    await handle.drain()
    assert clock.value > 3 * 60 * 60
    assert ticks == 13
    assert touched.await_count == 13
    assert all(
        call.args[0].phase is phase
        for call in touched.await_args_list
    )
    assert handle.done


async def test_phase_change_and_callback_failure_are_isolated() -> None:
    phases: list[CoordinatorParentPhase] = []
    release_tick = asyncio.Event()

    async def failing(context) -> None:
        phases.append(context.phase)
        raise RuntimeError("peripheral unavailable")

    succeeding = AsyncMock()

    async def sleep(_: float) -> None:
        await release_tick.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh",),
        liveness_service=_Liveness({"fresh": _lease("fresh")}),
        owner_alive=lambda: True,
        interval_seconds=1,
        sleep=sleep,
        touch_parent_activity=failing,
        renew_quota=succeeding,
    )
    handle.start()
    await _wait_until(lambda: succeeding.await_count == 1)
    handle.set_phase(CoordinatorParentPhase.APPLYING)
    release_tick.set()
    await _wait_until(lambda: succeeding.await_count >= 2)
    assert phases[:2] == [
        CoordinatorParentPhase.WAITING_CHILDREN,
        CoordinatorParentPhase.APPLYING,
    ]
    handle.stop()
    await handle.drain()


async def test_manual_cancelled_callback_is_logged_and_later_callbacks_continue(
    caplog,
) -> None:
    cancelled_calls = 0
    succeeding = AsyncMock()

    async def cancelled(_context) -> None:
        nonlocal cancelled_calls
        cancelled_calls += 1
        raise asyncio.CancelledError

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh",),
        liveness_service=_Liveness({"fresh": _lease("fresh")}),
        owner_alive=lambda: True,
        interval_seconds=1,
        sleep=lambda _seconds: asyncio.sleep(0),
        touch_parent_activity=cancelled,
        renew_quota=succeeding,
    )
    handle.start()
    await _wait_until(lambda: succeeding.await_count >= 2)
    assert cancelled_calls >= 2
    assert "callback failed" in caplog.text
    handle.stop()
    await handle.drain()
    assert handle.done


@pytest.mark.parametrize("cancel_via", ["task", "stop"])
async def test_real_parent_lease_task_cancellation_stops_callback_loop(
    cancel_via: str,
) -> None:
    entered = asyncio.Event()
    blocked = asyncio.Event()
    succeeding = AsyncMock()

    async def callback(_context) -> None:
        entered.set()
        await blocked.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh",),
        liveness_service=_Liveness({"fresh": _lease("fresh")}),
        owner_alive=lambda: True,
        interval_seconds=1,
        touch_parent_activity=callback,
        renew_quota=succeeding,
    )
    handle.start()
    await entered.wait()
    if cancel_via == "stop":
        handle.stop()
    else:
        assert handle._task is not None
        handle._task.cancel()

    await handle.drain()
    succeeding.assert_not_awaited()
    assert handle.done


async def test_start_stop_and_drain_are_explicitly_idempotent() -> None:
    touched = AsyncMock()
    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh",),
        liveness_service=_Liveness({"fresh": _lease("fresh")}),
        owner_alive=lambda: True,
        interval_seconds=60,
        touch_parent_activity=touched,
    )

    handle.start()
    handle.start()
    await _wait_until(lambda: touched.await_count == 1)
    handle.stop()
    handle.stop()
    await handle.drain()
    await handle.drain()
    assert handle.done


async def test_liveness_read_failure_is_unknown_then_recovers_next_tick(
    caplog,
) -> None:
    fresh = _lease("child")
    calls = 0
    release_tick = asyncio.Event()
    touched = AsyncMock()
    orphaned = AsyncMock()

    class FlakyLiveness(_Liveness):
        async def get_lease(self, child_id: str):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("redis unavailable")
            return fresh

    async def sleep(_: float) -> None:
        await release_tick.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child",),
        liveness_service=FlakyLiveness({}),
        owner_alive=lambda: True,
        interval_seconds=1,
        sleep=sleep,
        touch_parent_activity=touched,
        on_all_children_stale=orphaned,
    )
    handle.start()
    await _wait_until(lambda: calls == 1)
    touched.assert_not_awaited()
    orphaned.assert_not_awaited()
    assert "liveness read failed" in caplog.text

    release_tick.set()
    await _wait_until(lambda: touched.await_count == 1)
    orphaned.assert_not_awaited()
    handle.stop()
    await handle.drain()


@pytest.mark.parametrize(
    "stale_error",
    [RuntimeError("clock unavailable"), asyncio.CancelledError()],
)
async def test_liveness_stale_check_failure_is_unknown_then_recovers_next_tick(
    stale_error: BaseException,
    caplog,
) -> None:
    checks = 0
    release_tick = asyncio.Event()
    touched = AsyncMock()
    orphaned = AsyncMock()

    class FlakyLiveness(_Liveness):
        def is_stale(self, lease: CoordinatorChildLease | None) -> bool:
            nonlocal checks
            checks += 1
            if checks == 1:
                raise stale_error
            return False

    async def sleep(_: float) -> None:
        await release_tick.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child",),
        liveness_service=FlakyLiveness({"child": _lease("child")}),
        owner_alive=lambda: True,
        interval_seconds=1,
        sleep=sleep,
        touch_parent_activity=touched,
        on_all_children_stale=orphaned,
    )
    handle.start()
    await _wait_until(lambda: checks == 1)
    touched.assert_not_awaited()
    orphaned.assert_not_awaited()
    assert "liveness read failed" in caplog.text

    release_tick.set()
    await _wait_until(lambda: touched.await_count == 1)
    orphaned.assert_not_awaited()
    handle.stop()
    await handle.drain()


async def test_cancelling_drain_cleans_loop_then_propagates_to_caller() -> None:
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def callback(_context) -> None:
        entered.set()
        await blocked.wait()

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("fresh",),
        liveness_service=_Liveness({"fresh": _lease("fresh")}),
        owner_alive=lambda: True,
        interval_seconds=1,
        touch_parent_activity=callback,
    )
    handle.start()
    await entered.wait()
    drainer = asyncio.create_task(handle.drain())
    await asyncio.sleep(0)
    drainer.cancel()

    with pytest.raises(asyncio.CancelledError):
        await drainer
    assert handle.done


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_interval_must_be_finite_and_positive(value: float) -> None:
    with pytest.raises(ValueError, match="interval_seconds"):
        CoordinatorParentExecutionLease(
            root_session_id="root",
            parent_session_id="parent",
            coordinator_run_id="run",
            step_id="step",
            child_session_ids=("child",),
            liveness_service=_Liveness({}),
            owner_alive=lambda: True,
            interval_seconds=value,
        )


async def test_quota_renew_context_keeps_user_and_run_identity_past_six_hours(
) -> None:
    observed: list[tuple[str, str]] = []
    clock_value = 0.0
    owner_alive = True

    async def renew_quota(context) -> None:
        observed.append((context.user_id, context.coordinator_run_id))

    async def sleep(seconds: float) -> None:
        nonlocal clock_value, owner_alive
        clock_value += seconds
        if clock_value > 7 * 60 * 60:
            owner_alive = False
        await asyncio.sleep(0)

    handle = CoordinatorParentExecutionLease(
        root_session_id="root",
        parent_session_id="parent",
        user_id="user",
        coordinator_run_id="run",
        step_id="step",
        child_session_ids=("child",),
        liveness_service=_Liveness({"child": _lease("child")}),
        owner_alive=lambda: owner_alive,
        phase=CoordinatorParentPhase.REDUCING,
        interval_seconds=60 * 60,
        clock=lambda: clock_value,
        sleep=sleep,
        renew_quota=renew_quota,
    )

    handle.start()
    await handle.drain()

    assert clock_value > 7 * 60 * 60
    assert observed == [("user", "run")] * 8
