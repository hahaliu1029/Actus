"""PR-9b-A INV-A11 + INV-A12 — DefaultCoordinatorChildRunnerStarter contract.

Locks: (1) the 11-step start sequence (manifest decode → ChildPermissionContext
with 7 fields → background task spawn with set_name + active-tasks registration);
(2) the _on_task_done cancelled-guard (Task.exception() raises CancelledError
on cancelled tasks, so the callback MUST check task.cancelled() first).
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest


pytestmark = pytest.mark.anyio


@dataclass(frozen=True)
class _FakeWorkUnit:
    work_unit_id: str = "wu-1"


class _FakeArtifactStorage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.calls: list[str] = []

    async def get_bytes(self, ref: str) -> bytes:
        self.calls.append(ref)
        return self._payload


class _FakeSessionRepository:
    def __init__(self, revision: int = 7) -> None:
        self._revision = revision
        self.calls: list[str] = []

    async def read_mode_revision(self, session_id: str) -> int:
        self.calls.append(session_id)
        return self._revision


class _FakeRunnerFactory:
    def __init__(self) -> None:
        self.built = MagicMock()
        self.built.runner = MagicMock()
        self.call_args: dict[str, Any] = {}

    async def build(self, **kwargs: Any) -> Any:
        self.call_args = kwargs
        return self.built


class _FakeCoordinatorLimits:
    max_tool_calls_per_child: int = 100
    max_token_cost_usd_per_child: float = 2.0
    max_wallclock_seconds_per_child: int = 600


def _manifest_bytes() -> bytes:
    return json.dumps({
        "allowed_tools": ["file_read", "file_write"],
        "write_lease": [{"path": "/a/b", "op": "modify"}],
        "runtime_caps": [],
    }).encode("utf-8")


async def test_start_invokes_run_work_unit_with_decoded_manifest(monkeypatch):
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    captured: dict[str, Any] = {}

    class _FakeChildRunner:
        def __init__(self, **kwargs: Any) -> None:
            captured["ctor_kwargs"] = kwargs

        async def run_work_unit(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _FakeChildRunner,
    )

    artifact = _FakeArtifactStorage(_manifest_bytes())
    session_repo = _FakeSessionRepository(revision=11)
    runner_factory = _FakeRunnerFactory()
    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory,
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=session_repo,
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=artifact,
        coordinator_limits=_FakeCoordinatorLimits(),
    )

    await starter.start(
        coordinator_run_id="run-1",
        work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1",
        spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(),
        root_session_id="root-1",
        parent_session_id="parent-1",
        parent_sandbox=MagicMock(),
    )

    for _ in range(50):
        if "run_kwargs" in captured:
            break
        await asyncio.sleep(0.01)

    assert artifact.calls == ["ref-1"]
    assert session_repo.calls == ["child-1"]
    cpc = runner_factory.call_args["child_permission_context"]
    assert cpc.parent_session_id == "parent-1"
    assert cpc.child_session_id == "child-1"
    assert cpc.coordinator_run_id == "run-1"
    assert cpc.work_unit_id == "wu-1"
    assert cpc.session_mode_revision == 11
    assert cpc.lease_expiry is None

    rk = captured["run_kwargs"]
    assert rk["coordinator_run_id"] == "run-1"
    assert rk["root_session_id"] == "root-1"
    assert hasattr(rk["spawn_manifest"], "allowed_tools"), \
        "must pass decoded SpawnManifest, not raw ref"
    assert rk["spawn_manifest"].allowed_tools == frozenset({"file_read", "file_write"})


async def test_start_sets_task_name_and_registers_in_active(monkeypatch):
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    started = asyncio.Event()

    class _BlockingChildRunner:
        def __init__(self, **kwargs: Any) -> None: ...
        async def run_work_unit(self, **kwargs: Any) -> None:
            started.set()
            await asyncio.sleep(10)

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _BlockingChildRunner,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=_FakeRunnerFactory(),
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
    )
    await starter.start(
        coordinator_run_id="run-2",
        work_unit=_FakeWorkUnit("wu-2"),
        child_session_id="child-2",
        spawn_manifest_ref="ref-2",
        cancel_event=asyncio.Event(),
        root_session_id="root-2",
        parent_session_id="parent-2",
        parent_sandbox=MagicMock(),
    )
    await started.wait()
    assert "child-2" in starter._active_tasks
    task = starter._active_tasks["child-2"]
    assert task.get_name() == "coord-child-child-2"
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


async def test_done_callback_pops_on_success(monkeypatch):
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    class _ImmediateChildRunner:
        def __init__(self, **kwargs: Any) -> None: ...
        async def run_work_unit(self, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _ImmediateChildRunner,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=_FakeRunnerFactory(),
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
    )
    await starter.start(
        coordinator_run_id="run-3",
        work_unit=_FakeWorkUnit("wu-3"),
        child_session_id="child-3",
        spawn_manifest_ref="ref-3",
        cancel_event=asyncio.Event(),
        root_session_id="root-3",
        parent_session_id="parent-3",
        parent_sandbox=MagicMock(),
    )
    for _ in range(50):
        if "child-3" not in starter._active_tasks:
            break
        await asyncio.sleep(0.01)
    assert "child-3" not in starter._active_tasks


async def test_done_callback_pops_and_logs_on_exception(monkeypatch, caplog):
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    class _RaisingChildRunner:
        def __init__(self, **kwargs: Any) -> None: ...
        async def run_work_unit(self, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _RaisingChildRunner,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=_FakeRunnerFactory(),
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
    )
    with caplog.at_level(logging.ERROR):
        await starter.start(
            coordinator_run_id="run-4",
            work_unit=_FakeWorkUnit("wu-4"),
            child_session_id="child-4",
            spawn_manifest_ref="ref-4",
            cancel_event=asyncio.Event(),
            root_session_id="root-4",
            parent_session_id="parent-4",
            parent_sandbox=MagicMock(),
        )
        for _ in range(50):
            if "child-4" not in starter._active_tasks:
                break
            await asyncio.sleep(0.01)
    assert "child-4" not in starter._active_tasks
    assert any("crashed unhandled" in r.message for r in caplog.records)


async def test_done_callback_handles_cancelled_without_raising(monkeypatch, caplog):
    """INV-A12 critical: cancelled task -> callback MUST return early before
    Task.exception() (which would raise CancelledError out of the callback)."""
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    started = asyncio.Event()

    class _BlockingChildRunner:
        def __init__(self, **kwargs: Any) -> None: ...
        async def run_work_unit(self, **kwargs: Any) -> None:
            started.set()
            await asyncio.sleep(60)

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _BlockingChildRunner,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=_FakeRunnerFactory(),
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
    )
    await starter.start(
        coordinator_run_id="run-5",
        work_unit=_FakeWorkUnit("wu-5"),
        child_session_id="child-5",
        spawn_manifest_ref="ref-5",
        cancel_event=asyncio.Event(),
        root_session_id="root-5",
        parent_session_id="parent-5",
        parent_sandbox=MagicMock(),
    )
    await started.wait()
    task = starter._active_tasks["child-5"]

    with caplog.at_level(logging.ERROR):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        for _ in range(50):
            if "child-5" not in starter._active_tasks:
                break
            await asyncio.sleep(0.01)

    assert "child-5" not in starter._active_tasks
    assert not any("crashed unhandled" in r.message for r in caplog.records)
