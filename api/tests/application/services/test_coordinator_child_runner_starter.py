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


def _fake_lifecycle():
    from unittest.mock import AsyncMock, MagicMock

    handle = MagicMock()
    handle.get_browser = AsyncMock(return_value=MagicMock())
    svc = MagicMock()
    svc.bind_new = AsyncMock(return_value=handle)
    return svc


def _fake_resolve():
    import types
    from unittest.mock import MagicMock

    return types.SimpleNamespace(
        execution_supervisor=MagicMock(),
        uow_factory=lambda: MagicMock(),
    )


def _make_starter(
    *,
    runner_factory=None,
    lifecycle=None,
    publisher=None,
    envelope_factory=None,
    resolve=None,
):
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    return DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory or _FakeRunnerFactory(),
        mailbox_publisher=publisher or MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=envelope_factory or MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
        sandbox_lifecycle_service=lifecycle or _fake_lifecycle(),
        resolve_child_runner_deps=resolve or _fake_resolve,
    )


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
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
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
        user_id="user-1",
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
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
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
        user_id="user-2",
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
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
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
        user_id="user-3",
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
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
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
            user_id="user-4",
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
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
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
        user_id="user-5",
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


async def test_start_provisions_child_sandbox_and_cost_handler(monkeypatch):
    """bind_new is called with the child session + user_id; the per-child cost
    handler + per-child sandbox/browser flow into factory.build; the
    CoordinatorChildRunner receives a child_sandbox Port (A1)."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    lifecycle = _fake_lifecycle()
    built = MagicMock()
    runner_factory = MagicMock()
    runner_factory.build = AsyncMock(return_value=built)
    captured = {}
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "build_supervisor_aware_callback_handler",
        lambda supervisor, session_id, user_id, uow_factory: captured.setdefault(
            "cost", (session_id, user_id)
        ) or MagicMock(),
    )
    # CoordinatorChildRunner is real here; its run_work_unit will run on the built
    # mock runner. To keep the test fast + isolated, monkeypatch it to a no-op fake:
    class _NoopChildRunner:
        def __init__(self, **kwargs): captured["ctor"] = kwargs
        async def run_work_unit(self, **kwargs): return None
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _NoopChildRunner,
    )
    starter = _make_starter(runner_factory=runner_factory, lifecycle=lifecycle)
    await starter.start(
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(),
        user_id="user-1",
    )
    lifecycle.bind_new.assert_awaited_once_with("child-1", user_id="user-1")
    assert captured["cost"] == ("child-1", "user-1")
    bk = runner_factory.build.call_args.kwargs
    assert bk["user_id"] == "user-1"
    assert "sandbox" in bk and "browser" in bk and "cost_callback_handler" in bk
    assert captured["ctor"].get("child_sandbox") is not None  # A1: child_sandbox Port threaded
    # A1: the child gets its OWN sandbox port, never the parent's handle.
    assert captured["ctor"]["child_sandbox"] is not captured["ctor"]["parent_sandbox"]


async def test_start_failure_after_bind_new_publishes_failed_and_does_not_destroy(monkeypatch):
    """A factory.build failure after bind_new publishes a FAILED RESULT_READY
    for the work_unit AND issues NO sandbox destroy (M1); start does not raise."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock
    lifecycle = _fake_lifecycle()  # has bind_new; NO destroy attribute used
    handle = lifecycle.bind_new.return_value  # the handle _fake_lifecycle hands out
    handle.destroy = AsyncMock()
    runner_factory = MagicMock()
    runner_factory.build = AsyncMock(side_effect=RuntimeError("build boom"))
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    envelope_factory = MagicMock()
    envelope_factory.make_result_ready = MagicMock(return_value="ENVELOPE")
    starter = _make_starter(
        runner_factory=runner_factory, lifecycle=lifecycle,
        publisher=publisher, envelope_factory=envelope_factory,
    )
    await starter.start(  # must NOT raise
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(),
        user_id="user-1",
    )
    publisher.publish.assert_awaited()  # FAILED envelope published
    lifecycle.bind_new.assert_awaited_once()
    # M1: the starter never destroys — real regression gates, not just
    # "fake has no destroy attr" (a MagicMock would auto-create the call).
    handle.destroy.assert_not_called()
    lifecycle.destroy.assert_not_called()
