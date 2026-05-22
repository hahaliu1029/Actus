"""C3 PR-3b — SupervisorRegistry (spec §3.2 M8 + §6.2).

Per-pod registry of MailboxSupervisor asyncio.Tasks. Detects task-level
crashes and restarts on the same root with a new instance_id. Pod-level
recovery is a separate concern (reconcile_orphans, spec §3.2 M8).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from app.application.services.supervisor_registry import (
    SupervisorRegistry,
)


class _FakeSupervisor:
    """Minimal MailboxSupervisor shape — just enough for the registry to
    treat us as a task target. ``run()`` either raises (to simulate a
    crash) or sleeps until cancelled (the alive path).
    """

    def __init__(self, root: str, *, will_crash: bool = False) -> None:
        self.root = root
        self.will_crash = will_crash
        self.run_calls = 0
        # Registry injects _ready_event before calling run() — mirror the
        # production MailboxSupervisor attribute.
        self._ready_event: asyncio.Event | None = None

    async def run(self) -> None:
        self.run_calls += 1
        # Signal readiness so registry.spawn() unblocks.
        if self._ready_event is not None:
            self._ready_event.set()
        if self.will_crash:
            raise RuntimeError("synthetic supervisor crash")
        # Idle alive — wait for cancellation.
        await asyncio.Event().wait()


class _Factory:
    """Callable factory the registry uses to spawn supervisors. Tracks
    every supervisor it created so tests can assert on identity / count /
    restart counts.
    """

    def __init__(self, crash_first: bool = False) -> None:
        self.created: list[_FakeSupervisor] = []
        self.crash_first = crash_first

    def __call__(self, root_session_id: str) -> _FakeSupervisor:
        will_crash = self.crash_first and len(self.created) == 0
        sup = _FakeSupervisor(root_session_id, will_crash=will_crash)
        self.created.append(sup)
        return sup


@pytest.mark.anyio
async def test_spawn_creates_supervisor_task():
    factory = _Factory()
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=10.0)
    await reg.spawn("root-1")
    health = await reg.health_check()
    assert health == {"root-1": "alive"}
    assert len(factory.created) == 1
    await reg.stop_all()


@pytest.mark.anyio
async def test_spawn_same_root_twice_is_idempotent():
    factory = _Factory()
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=10.0)
    await reg.spawn("root-1")
    await reg.spawn("root-1")
    assert len(factory.created) == 1, "second spawn must be a no-op"
    await reg.stop_all()


@pytest.mark.anyio
async def test_health_check_detects_crashed_task(caplog):
    """When a supervisor crashes mid-run, health_check reports 'crashed'
    until the restart loop kicks in. Use a long restart_interval_s so the
    crashed task surfaces in the health snapshot.
    """
    factory = _Factory(crash_first=True)
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=10.0)
    await reg.spawn("root-1")
    # Crashed supervisor's run() raised immediately — its asyncio.Task.done()
    # is now True with an exception set. Give the event loop a tick.
    await asyncio.sleep(0.05)
    with caplog.at_level(logging.WARNING):
        health = await reg.health_check()
    assert health == {"root-1": "crashed"}
    await reg.stop_all()


@pytest.mark.anyio
async def test_crashed_task_is_restarted_with_new_instance_id(caplog):
    """Spec §3.2 M8 — crashed supervisor task is restarted on the same
    root with a fresh instance_id. Short restart_interval_s drives the
    restart loop into the test window.
    """
    factory = _Factory(crash_first=True)
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=0.05)
    await reg.spawn("root-1")
    # First supervisor crashed; wait for the restart loop tick.
    with caplog.at_level(logging.WARNING):
        await asyncio.sleep(0.4)
    # A second supervisor was constructed and is alive now.
    assert len(factory.created) >= 2, (
        f"expected >=2 supervisors (crash + restart), got {len(factory.created)}"
    )
    # The replacement supervisor must NOT be the original (different identity).
    assert factory.created[0] is not factory.created[1]
    # Restart logged a warning surface.
    restart_logs = [
        rec for rec in caplog.records if "supervisor restarted" in rec.getMessage()
    ]
    assert restart_logs, "expected restart warning in logs"
    await reg.stop_all()


@pytest.mark.anyio
async def test_stop_all_cancels_all_tasks():
    factory = _Factory()
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=10.0)
    await reg.spawn("root-1")
    await reg.spawn("root-2")
    await reg.spawn("root-3")
    health = await reg.health_check()
    assert set(health.keys()) == {"root-1", "root-2", "root-3"}
    await reg.stop_all()
    # After stop_all, slots cleared.
    health_after = await reg.health_check()
    assert health_after == {}


@pytest.mark.anyio
async def test_concurrent_spawn_same_root_is_safe():
    """Codex r1 [P0] regression — two concurrent ``spawn(root)`` calls MUST
    NOT both create a task and orphan one.

    Pre-fix sequence:
      Coro A: ``not in _slots`` → create_task A → suspend on
              ``await asyncio.wait_for(ready_event.wait(), ...)``
      Coro B: ``not in _slots`` (still empty — A hasn't written the slot)
              → create_task B → suspend on same await
      Both resume → both write ``_slots[root]`` → last write wins.
      First task is now invisible to ``stop_all`` (orphan).

    Post-fix: the slot is registered synchronously right after
    ``create_task`` returns, so when B re-enters the spawn body the
    idempotent fast path fires and only ONE supervisor is ever created
    per root.
    """
    factory = _Factory()
    reg = SupervisorRegistry(supervisor_factory=factory, restart_interval_s=10.0)
    # Fire two spawns for the same root concurrently. ``asyncio.gather``
    # interleaves them; both observe the empty registry before either
    # writes its slot under the pre-fix code.
    await asyncio.gather(reg.spawn("root-1"), reg.spawn("root-1"))
    # Exactly one supervisor was constructed and one slot exists.
    assert len(factory.created) == 1, (
        f"expected exactly 1 supervisor for concurrent spawn(root-1), "
        f"got {len(factory.created)}"
    )
    assert "root-1" in reg._slots  # noqa: SLF001
    await reg.stop_all()
