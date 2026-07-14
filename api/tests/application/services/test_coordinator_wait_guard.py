"""Structured coordinator wait lifecycle guard tests."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from app.application.services.coordinator_wait_guard import CoordinatorWaitGuard
from app.application.services.coordinator_parent_execution_lease import (
    CoordinatorParentPhase,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RecordingWatchdog:
    def __init__(self) -> None:
        self.paused: list[str] = []
        self.resumed: list[str] = []

    def pause_idle(self, key: str) -> None:
        self.paused.append(key)

    def resume_idle(self, key: str) -> None:
        self.resumed.append(key)


async def test_enter_and_resume_run_are_idempotent_and_step_scoped():
    watchdog = _RecordingWatchdog()
    guard = CoordinatorWaitGuard(watchdog=watchdog)

    await guard.enter_run("step-a", "run-1")
    await guard.enter_run("step-a", "run-1")
    await guard.enter_run("step-b", "run-1")

    assert len(watchdog.paused) == 2
    assert await guard.resume_run("step-a", "run-1") is False
    assert await guard.resume_run("step-a", "run-1") is False
    assert watchdog.resumed == [watchdog.paused[0]]

    await guard.resume_all_for_step("step-b")
    assert watchdog.resumed == watchdog.paused


async def test_step_scope_resumes_all_runs_on_success():
    watchdog = _RecordingWatchdog()
    guard = CoordinatorWaitGuard(watchdog=watchdog)

    async with guard.step_scope("step-a"):
        await guard.enter_run("step-a", "run-1")
        await guard.enter_run("step-a", "run-2")

    assert watchdog.resumed == watchdog.paused


async def test_step_scope_resumes_all_runs_on_exception_and_cancel():
    watchdog = _RecordingWatchdog()
    guard = CoordinatorWaitGuard(watchdog=watchdog)

    with pytest.raises(RuntimeError, match="boom"):
        async with guard.step_scope("step-error"):
            await guard.enter_run("step-error", "run-1")
            raise RuntimeError("boom")

    with pytest.raises(asyncio.CancelledError):
        async with guard.step_scope("step-cancel"):
            await guard.enter_run("step-cancel", "run-2")
            raise asyncio.CancelledError

    assert watchdog.resumed == watchdog.paused


async def test_none_watchdog_and_parent_lease_factory_are_supported():
    guard = CoordinatorWaitGuard(watchdog=None, parent_lease_factory=None)

    async with guard.step_scope("step-a"):
        await guard.enter_run("step-a", "run-1")
        await guard.resume_run("step-a", "run-1")

    await guard.resume_all_for_step("unknown")


async def test_resume_run_reports_transferred_quota_ownership() -> None:
    released = 0

    async def release_quota() -> None:
        nonlocal released
        released += 1

    guard = CoordinatorWaitGuard()
    await guard.enter_run("step-a", "run-1", release_quota=release_quota)

    assert await guard.resume_run("step-a", "run-1") is True
    assert await guard.resume_run("step-a", "run-1") is False
    assert released == 1


class _Handle:
    def __init__(self) -> None:
        self.started = 0
        self.stopped = 0
        self.drained = 0
        self.phases: list[CoordinatorParentPhase] = []

    def start(self) -> None:
        self.started += 1

    def set_phase(self, phase: CoordinatorParentPhase) -> None:
        self.phases.append(phase)

    def stop(self) -> None:
        self.stopped += 1

    async def drain(self) -> None:
        self.drained += 1


async def test_one_parent_lease_per_step_run_and_phase_is_step_scoped():
    watchdog = _RecordingWatchdog()
    handles: list[_Handle] = []

    def factory(**kwargs):
        handle = _Handle()
        handles.append(handle)
        assert kwargs["child_session_ids"] == ("child-1", "child-2")
        assert kwargs["owner_alive"]() is True
        return handle

    guard = CoordinatorWaitGuard(
        watchdog=watchdog, parent_lease_factory=factory,
    )
    async with guard.step_scope("step-a"):
        kwargs = {
            "root_session_id": "root",
            "parent_session_id": "parent",
            "child_session_ids": ("child-1", "child-2"),
        }
        await guard.enter_run("step-a", "run", **kwargs)
        await guard.enter_run("step-a", "run", **kwargs)
        guard.set_phase("step-a", "run", CoordinatorParentPhase.REDUCING)
        guard.set_phase("other-step", "run", CoordinatorParentPhase.APPLYING)

    assert len(handles) == 1
    assert handles[0].started == 1
    assert handles[0].phases == [CoordinatorParentPhase.REDUCING]
    assert handles[0].stopped == handles[0].drained == 1
    assert watchdog.resumed == watchdog.paused


async def test_nested_scope_only_drains_at_outer_exit():
    handle = _Handle()
    guard = CoordinatorWaitGuard(
        parent_lease_factory=MagicMock(return_value=handle),
    )
    async with guard.step_scope("step"):
        async with guard.step_scope("step"):
            await guard.enter_run(
                "step",
                "run",
                root_session_id="root",
                parent_session_id="parent",
                child_session_ids=("child",),
            )
        assert handle.stopped == 0
    assert handle.stopped == handle.drained == 1


@pytest.mark.parametrize("exit_kind", ["exception", "cancel"])
async def test_parent_lease_drains_before_watchdog_resume_on_abnormal_exit(
    exit_kind: str,
):
    events: list[str] = []

    class Handle(_Handle):
        def stop(self) -> None:
            events.append("stop")

        async def drain(self) -> None:
            events.append("drain")

    class Watchdog(_RecordingWatchdog):
        def resume_idle(self, key) -> None:
            events.append("resume")
            super().resume_idle(key)

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(), parent_lease_factory=lambda **_: Handle(),
    )
    error = RuntimeError("boom") if exit_kind == "exception" else asyncio.CancelledError()
    with pytest.raises(type(error)):
        async with guard.step_scope("step"):
            await guard.enter_run(
                "step", "run", root_session_id="root",
                parent_session_id="parent", child_session_ids=("child",),
            )
            raise error
    assert events == ["stop", "drain", "resume"]


async def test_enter_run_pause_failure_rolls_back_without_publishing_handle():
    events: list[str] = []
    handle = _Handle()

    class Watchdog(_RecordingWatchdog):
        def pause_idle(self, key) -> None:
            events.append("pause")
            raise RuntimeError("pause failed")

        def resume_idle(self, key) -> None:
            events.append("resume")

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(), parent_lease_factory=lambda **_: handle,
    )

    with pytest.raises(RuntimeError, match="pause failed"):
        await guard.enter_run(
            "step", "run", root_session_id="root",
            parent_session_id="parent", child_session_ids=("child",),
        )

    assert events == ["pause", "resume"]
    assert handle.started == handle.stopped == handle.drained == 0
    assert guard._runs_by_step == {}


@pytest.mark.parametrize(
    "start_error",
    [RuntimeError("start failed"), asyncio.CancelledError()],
)
async def test_enter_run_start_failure_stops_and_drains_without_leak(
    start_error: BaseException,
):
    events: list[str] = []

    class Handle(_Handle):
        def start(self) -> None:
            events.append("start")
            raise start_error

        def stop(self) -> None:
            events.append("stop")

        async def drain(self) -> None:
            events.append("drain")

    class Watchdog(_RecordingWatchdog):
        def pause_idle(self, key) -> None:
            events.append("pause")

        def resume_idle(self, key) -> None:
            events.append("resume")

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(), parent_lease_factory=lambda **_: Handle(),
    )

    with pytest.raises(type(start_error)) as caught:
        await guard.enter_run(
            "step", "run", root_session_id="root",
            parent_session_id="parent", child_session_ids=("child",),
        )

    assert caught.value is start_error
    assert events == ["pause", "start", "stop", "drain", "resume"]
    assert guard._runs_by_step == {}


async def test_enter_run_cleanup_failure_does_not_mask_start_error():
    original = RuntimeError("start failed")

    class Handle(_Handle):
        def start(self) -> None:
            raise original

        async def drain(self) -> None:
            raise RuntimeError("cleanup failed")

    guard = CoordinatorWaitGuard(
        watchdog=_RecordingWatchdog(),
        parent_lease_factory=lambda **_: Handle(),
    )

    with pytest.raises(RuntimeError) as caught:
        await guard.enter_run(
            "step", "run", root_session_id="root",
            parent_session_id="parent", child_session_ids=("child",),
        )

    assert caught.value is original
    assert guard._runs_by_step == {}


async def test_enter_run_real_cancel_during_start_rollback_wins_after_cleanup(
) -> None:
    events: list[str] = []
    drain_entered = asyncio.Event()
    drain_calls = 0

    class Handle(_Handle):
        def start(self) -> None:
            events.append("start")
            raise RuntimeError("start failed")

        def stop(self) -> None:
            events.append("stop")

        async def drain(self) -> None:
            nonlocal drain_calls
            drain_calls += 1
            events.append("drain")
            if drain_calls == 1:
                drain_entered.set()
                await asyncio.Event().wait()

    class Watchdog(_RecordingWatchdog):
        def pause_idle(self, key) -> None:
            events.append("pause")

        def resume_idle(self, key) -> None:
            events.append("resume")

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(), parent_lease_factory=lambda **_: Handle(),
    )
    entering = asyncio.create_task(guard.enter_run(
        "step", "run", root_session_id="root",
        parent_session_id="parent", child_session_ids=("child",),
    ))
    await drain_entered.wait()
    entering.cancel()

    with pytest.raises(asyncio.CancelledError):
        await entering

    assert "stop" in events
    assert "drain" in events
    assert "resume" in events
    assert events.index("stop") < events.index("drain") < events.index("resume")
    assert guard._runs_by_step == {}


async def test_normal_cleanup_attempts_every_stage_for_every_run() -> None:
    events: list[str] = []

    class Handle(_Handle):
        def __init__(
            self,
            run_id: str,
            *,
            stop_error: BaseException | None = None,
            drain_error: BaseException | None = None,
        ) -> None:
            super().__init__()
            self.run_id = run_id
            self.stop_error = stop_error
            self.drain_error = drain_error

        def stop(self) -> None:
            events.append(f"stop:{self.run_id}")
            if self.stop_error is not None:
                raise self.stop_error

        async def drain(self) -> None:
            events.append(f"drain:{self.run_id}")
            if self.drain_error is not None:
                raise self.drain_error

    handles = {
        "run-1": Handle("run-1", stop_error=RuntimeError("stop failed")),
        "run-2": Handle(
            "run-2",
            stop_error=asyncio.CancelledError(),
            drain_error=RuntimeError("drain failed"),
        ),
        "run-3": Handle("run-3"),
    }

    class Watchdog(_RecordingWatchdog):
        def resume_idle(self, key) -> None:
            run_id = key[2]
            events.append(f"resume:{run_id}")
            if run_id == "run-1":
                raise RuntimeError("resume failed")
            if run_id == "run-2":
                raise asyncio.CancelledError

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(),
        parent_lease_factory=lambda **kwargs: handles[
            kwargs["coordinator_run_id"]
        ],
    )
    for run_id in handles:
        await guard.enter_run(
            "step", run_id, root_session_id="root",
            parent_session_id="parent", child_session_ids=("child",),
        )

    await guard.resume_all_for_step("step")

    assert events == [
        "stop:run-1", "drain:run-1", "resume:run-1",
        "stop:run-2", "drain:run-2", "resume:run-2",
        "stop:run-3", "drain:run-3", "resume:run-3",
    ]
    assert guard._runs_by_step == {}


@pytest.mark.parametrize("backend_fails", [False, True])
async def test_cleanup_failures_do_not_replace_backend_outcome(
    backend_fails: bool,
) -> None:
    original = RuntimeError("backend failed")

    class Handle(_Handle):
        def stop(self) -> None:
            raise RuntimeError("stop failed")

        async def drain(self) -> None:
            raise RuntimeError("drain failed")

    class Watchdog(_RecordingWatchdog):
        def resume_idle(self, key) -> None:
            raise RuntimeError("resume failed")

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(), parent_lease_factory=lambda **_: Handle(),
    )

    async def run_step() -> None:
        async with guard.step_scope("step"):
            await guard.enter_run(
                "step", "run", root_session_id="root",
                parent_session_id="parent", child_session_ids=("child",),
            )
            if backend_fails:
                raise original

    if backend_fails:
        with pytest.raises(RuntimeError) as caught:
            await run_step()
        assert caught.value is original
    else:
        await run_step()


async def test_real_caller_cancellation_cleans_all_runs_then_propagates() -> None:
    events: list[str] = []
    first_drain_entered = asyncio.Event()

    class Handle(_Handle):
        def __init__(self, run_id: str) -> None:
            super().__init__()
            self.run_id = run_id

        def stop(self) -> None:
            events.append(f"stop:{self.run_id}")

        async def drain(self) -> None:
            events.append(f"drain:{self.run_id}")
            if self.run_id == "run-1":
                first_drain_entered.set()
                await asyncio.Event().wait()

    handles = {run_id: Handle(run_id) for run_id in ("run-1", "run-2")}

    class Watchdog(_RecordingWatchdog):
        def resume_idle(self, key) -> None:
            events.append(f"resume:{key[2]}")

    guard = CoordinatorWaitGuard(
        watchdog=Watchdog(),
        parent_lease_factory=lambda **kwargs: handles[
            kwargs["coordinator_run_id"]
        ],
    )
    for run_id in handles:
        await guard.enter_run(
            "step", run_id, root_session_id="root",
            parent_session_id="parent", child_session_ids=("child",),
        )

    cleanup = asyncio.create_task(guard.resume_all_for_step("step"))
    await first_drain_entered.wait()
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup

    assert events == [
        "stop:run-1", "drain:run-1", "resume:run-1",
        "stop:run-2", "drain:run-2", "resume:run-2",
    ]
    assert guard._runs_by_step == {}


@pytest.mark.parametrize("exit_kind", ["success", "exception", "cancel"])
async def test_step_scope_drains_parent_lease_then_releases_quota_once(
    exit_kind: str,
) -> None:
    events: list[str] = []

    class Handle(_Handle):
        def stop(self) -> None:
            events.append("stop")

        async def drain(self) -> None:
            events.append("drain")

    async def release_quota() -> None:
        events.append("release")

    guard = CoordinatorWaitGuard(
        parent_lease_factory=lambda **kwargs: (
            events.append(f"factory-user:{kwargs['user_id']}") or Handle()
        ),
    )

    async def execute() -> None:
        async with guard.step_scope("step"):
            await guard.enter_run(
                "step",
                "run",
                root_session_id="root",
                parent_session_id="parent",
                user_id="user",
                child_session_ids=("child",),
                release_quota=release_quota,
            )
            if exit_kind == "exception":
                raise RuntimeError("backend failed")
            if exit_kind == "cancel":
                raise asyncio.CancelledError

    if exit_kind == "exception":
        with pytest.raises(RuntimeError, match="backend failed"):
            await execute()
    elif exit_kind == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await execute()
    else:
        await execute()

    assert events == ["factory-user:user", "stop", "drain", "release"]


async def test_quota_release_failure_is_best_effort_and_not_retried() -> None:
    releases = 0

    async def failing_release() -> None:
        nonlocal releases
        releases += 1
        raise RuntimeError("redis unavailable")

    guard = CoordinatorWaitGuard()
    async with guard.step_scope("step"):
        await guard.enter_run(
            "step", "run", release_quota=failing_release,
        )

    assert releases == 1
