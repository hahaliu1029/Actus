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
    # [S2 PR-4 §3.6] snapshot caps
    max_snapshot_paths: int = 20000
    max_snapshot_files: int = 8000
    max_snapshot_total_bytes: int = 100 * 1024 * 1024
    max_snapshot_seconds: float = 30.0


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


import contextlib


_STARTER_LOGGER_NAME = (
    "app.application.services.coordinator_child_runner_starter"
)


@contextlib.contextmanager
def _capture_warnings():
    """Capture WARNING+ messages from the starter module's logger.

    NOTE: this project installs ``_RedactingPropagateOnlyLogger`` as the
    default logger class (app/infrastructure/logging/redaction.py), whose
    ``addHandler`` is a documented no-op — handlers attached to a named
    logger are silently dropped and records only reach the root logger via
    ``propagate=True``. So we attach to the ROOT logger and filter by
    ``record.name`` instead of attaching to the named starter logger.
    """
    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.name == _STARTER_LOGGER_NAME:
                records.append(record.getMessage())

    root = logging.getLogger()
    handler = _H(level=logging.WARNING)
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)


class _FakeChildLlm:
    """Minimal real-shaped child llm: the starter's pricing pre-bind reads
    model_name + _identifying_params (same source as runtime
    invocation_params — spec §3-6). ('openai_official', 'gpt-4o') IS in the
    static pricing table, so the happy path resolves a real price."""

    model_name = "gpt-4o"
    _identifying_params = {"model": "gpt-4o", "provider_id": "openai_official"}


def _fake_resolve():
    import types
    from unittest.mock import MagicMock

    return types.SimpleNamespace(
        execution_supervisor=MagicMock(),
        uow_factory=lambda: MagicMock(),
        llm=_FakeChildLlm(),
    )


def _fake_resolve_without_llm():
    """Legacy shape (pre-C2b): no llm attribute — exercises the fail-soft
    'skip token callback + WARNING' path (spec R4#3)."""
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

        def attach_budget_callback(self, cb: Any) -> None:  # C2b late-inject
            captured["attached_cb"] = cb

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
        def attach_budget_callback(self, cb: Any) -> None: ...
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
        def attach_budget_callback(self, cb: Any) -> None: ...
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
        def attach_budget_callback(self, cb: Any) -> None: ...
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
        def attach_budget_callback(self, cb: Any) -> None: ...
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
        def attach_budget_callback(self, cb: Any) -> None: ...
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
    assert "sandbox_accessor" in bk and "browser_accessor" in bk and "cost_callback_handler" in bk
    # INV-SPM-8: the child is ALWAYS eager — the bare bind_new handle is wrapped in
    # an EagerSandboxAccessor (zero-provision get()), NEVER an on_demand accessor.
    # Type-pin + identity-pin so a future on_demand slip on the child path turns red.
    from app.application.services.sandbox_accessors import EagerSandboxAccessor

    child_handle = lifecycle.bind_new.return_value
    assert isinstance(bk["sandbox_accessor"], EagerSandboxAccessor)
    assert bk["sandbox_accessor"].peek() is child_handle
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


# ── C2b budget D10(b): starter ctor coordinator_metrics shape ────────────────


async def test_starter_ctor_accepts_optional_coordinator_metrics():
    """None-tolerant new kwarg: existing constructions (no metrics) stay
    valid; an explicit metrics object is stored for runner threading."""
    starter = _make_starter()
    assert starter._coordinator_metrics is None

    metrics = MagicMock(name="coordinator_metrics")
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    starter2 = DefaultCoordinatorChildRunnerStarter(
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
        coordinator_metrics=metrics,
    )
    assert starter2._coordinator_metrics is metrics


# ── C2b budget §3-6: starter late-inject (spec §5-10) ────────────────────────


def _capturing_child_runner(monkeypatch, captured: dict):
    """Monkeypatch CoordinatorChildRunner with a ctor-capturing noop fake."""

    class _NoopChildRunner:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs
            captured["instance"] = self
            self.attached = None

        def attach_budget_callback(self, cb):
            captured["attached"] = cb

        async def run_work_unit(self, **kwargs):
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "CoordinatorChildRunner",
        _NoopChildRunner,
    )
    return _NoopChildRunner


async def _start_once(starter):
    await starter.start(
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(),
        user_id="user-1",
    )


async def test_late_inject_constructs_and_wires_budget_callback(monkeypatch):
    """[spec §5-10 happy] After step 7: a BudgetEnforcementCallback is built
    with runner=child_runner + the per-child cap; attach_budget_callback AND
    built.runner.set_budget_callback both receive THE SAME instance; runner
    ctor got budget= + coordinator_metrics=."""
    from app.application.services.budget_enforcement_callback import (
        BudgetEnforcementCallback,
    )

    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)
    runner_factory = _FakeRunnerFactory()
    metrics = MagicMock(name="metrics")
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory,
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
        coordinator_metrics=metrics,
    )
    await _start_once(starter)

    # Runner ctor threading (D2 + D10c).
    budget = captured["ctor"]["budget"]
    assert budget.max_token_cost_usd == 2.0  # _FakeCoordinatorLimits value
    assert budget.max_wallclock_seconds == 600
    assert captured["ctor"]["coordinator_metrics"] is metrics

    # Late-inject: same callback instance to BOTH attach + setter chain.
    cb = captured["attached"]
    assert isinstance(cb, BudgetEnforcementCallback)
    assert cb._cap == 2.0
    assert cb._runner is captured["instance"]
    runner_factory.built.runner.set_budget_callback.assert_called_once_with(cb)


async def test_zero_wallclock_limit_reaches_child_as_unlimited_budget(monkeypatch):
    """Production starter wiring preserves zero so the child creates no watchdog."""
    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)

    class _ZeroWallclockLimits(_FakeCoordinatorLimits):
        max_wallclock_seconds_per_child: int = 0

    runner_factory = _FakeRunnerFactory()
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory,
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_ZeroWallclockLimits(),
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
    )

    await _start_once(starter)

    runner_budget = captured["ctor"]["budget"]
    context_budget = runner_factory.call_args["child_permission_context"].budget
    assert runner_budget is context_budget
    assert runner_budget.max_wallclock_seconds == 0


async def test_pricing_source_is_built_llm_identifying_params(monkeypatch):
    """[spec §5-10 R6#1 source-of-truth] get_price receives (model,
    provider_id) from the BUILT child llm's _identifying_params — NOT from
    any LLMConfig. get_price returning None → NO callback + WARNING."""
    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)
    price_calls: list = []

    def fake_get_price(model, provider_id=None):
        price_calls.append((model, provider_id))
        return None  # unpriced → fail-soft

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.get_price",
        fake_get_price,
    )
    runner_factory = _FakeRunnerFactory()
    starter = _make_starter(runner_factory=runner_factory)

    with _capture_warnings() as records:
        await _start_once(starter)

    assert price_calls == [("gpt-4o", "openai_official")]
    assert captured.get("attached") is None  # no callback constructed
    runner_factory.built.runner.set_budget_callback.assert_not_called()
    assert any("token budget" in m.lower() and "disabled" in m.lower()
               for m in records), records


async def test_pricing_prebound_price_flows_into_compute_cost(monkeypatch):
    """[spec §5-10 / §3-6 PricingFn 契约，codex plan-R2#2] The price resolved
    ONCE at construction is the SAME object handed to every per-call
    compute_cost(usage_metadata, price) — kills a _PreBoundPricing mutant
    that ignores the bound price or re-resolves per call."""
    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)

    sentinel_price = {"input": 1}  # opaque — only identity matters here
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.get_price",
        lambda model, provider_id=None: sentinel_price,
    )
    cc_calls: list = []
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.compute_cost",
        lambda usage_metadata, price: cc_calls.append((usage_metadata, price)) or 0.0,
    )
    runner_factory = _FakeRunnerFactory()
    starter = _make_starter(runner_factory=runner_factory)
    await _start_once(starter)

    cb = captured["attached"]
    assert cb is not None, "happy path must construct the callback"

    # [codex plan-R3#1] POISON get_price after construction: a mutant that
    # re-resolves the price per call (instead of using the construction-time
    # pre-bound dict) would call the now-raising get_price and fail here.
    def _poisoned_get_price(model, provider_id=None):
        raise AssertionError(
            "get_price called AFTER construction — price must be pre-bound"
        )

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.get_price",
        _poisoned_get_price,
    )

    usage = {"input_tokens": 7}
    result = cb._pricing.compute_cost(usage)
    assert result == 0.0
    assert cc_calls == [(usage, sentinel_price)], (
        "compute_cost must receive the construction-time pre-bound price, "
        f"got {cc_calls}"
    )


async def test_missing_llm_fail_soft_skips_callback(monkeypatch):
    """[spec R4#3] resolve deps WITHOUT llm → skip token callback + WARNING,
    start() does not raise; wallclock budget still threads through."""
    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)
    runner_factory = _FakeRunnerFactory()
    starter = _make_starter(
        runner_factory=runner_factory, resolve=_fake_resolve_without_llm,
    )
    with _capture_warnings() as records:
        await _start_once(starter)

    assert captured.get("attached") is None
    runner_factory.built.runner.set_budget_callback.assert_not_called()
    assert captured["ctor"]["budget"] is not None  # watchdog path unaffected
    assert any("token budget" in m.lower() and "disabled" in m.lower()
               for m in records), records


async def test_non_positive_token_cap_skips_callback(monkeypatch):
    """[spec L8 starter 半] max_token_cost_usd <= 0 → skip construction +
    WARNING (a cap=0 callback would trip on the first call — dishonest)."""
    captured: dict = {}
    _capturing_child_runner(monkeypatch, captured)

    class _ZeroTokenLimits(_FakeCoordinatorLimits):
        max_token_cost_usd_per_child: float = 0.0

    runner_factory = _FakeRunnerFactory()
    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    starter = DefaultCoordinatorChildRunnerStarter(
        runner_factory=runner_factory,
        mailbox_publisher=MagicMock(),
        mailbox_subscriber=MagicMock(),
        envelope_factory=MagicMock(),
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_ZeroTokenLimits(),
        sandbox_lifecycle_service=_fake_lifecycle(),
        resolve_child_runner_deps=_fake_resolve,
    )
    with _capture_warnings() as records:
        await _start_once(starter)

    assert captured.get("attached") is None
    runner_factory.built.runner.set_budget_callback.assert_not_called()
    assert any("non-positive" in m.lower() for m in records), records


# ── C2b budget §3-9 (R3#1+R4#2): request_stop_started rollback stop ──────────


class _StopRecordingChildRunner:
    """Real-ish stop semantics: first-wins reason + event set — what
    request_stop_started must drive on each STARTED child."""

    instances: list["_StopRecordingChildRunner"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.cancel_event = kwargs["cancel_event"]
        self.stop_reason = None
        type(self).instances.append(self)

    def attach_budget_callback(self, cb: Any) -> None: ...

    def request_stop(self, reason: Any) -> None:
        if self.stop_reason is None:
            self.stop_reason = reason
        self.cancel_event.set()

    async def run_work_unit(self, **kwargs: Any) -> None:
        await asyncio.sleep(30)  # stays RUNNING until stopped/cancelled


async def test_request_stop_started_scoped_to_given_children(monkeypatch):
    """[spec §5-14 starter 半 / R4#2 隔离] Stop EXACTLY the listed children —
    a concurrent run's child on the SAME lifespan-singleton starter is
    untouched (event unset). Unknown ids are a silent no-op."""
    from app.application.services.coordinator_child_runner import StopReason

    _StopRecordingChildRunner.instances = []
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "CoordinatorChildRunner",
        _StopRecordingChildRunner,
    )
    starter = _make_starter()

    async def _start(child_sid: str, run_id: str):
        await starter.start(
            coordinator_run_id=run_id, work_unit=_FakeWorkUnit(f"wu-{child_sid}"),
            child_session_id=child_sid, spawn_manifest_ref="ref",
            cancel_event=asyncio.Event(), root_session_id="root",
            parent_session_id="parent", parent_sandbox=MagicMock(),
            user_id="user-1",
        )

    await _start("run1-c1", "run-1")
    await _start("run1-c2", "run-1")
    await _start("run2-c1", "run-2")  # concurrent run, same starter
    r1c1, r1c2, r2c1 = _StopRecordingChildRunner.instances

    starter.request_stop_started(["run1-c1", "run1-c2", "ghost-id"])

    assert r1c1.stop_reason == StopReason.PARENT_CANCEL
    assert r1c2.stop_reason == StopReason.PARENT_CANCEL
    assert r1c1.cancel_event.is_set() and r1c2.cancel_event.is_set()
    # R4#2: the other run's child is NOT stopped.
    assert r2c1.stop_reason is None
    assert not r2c1.cancel_event.is_set()

    # Cleanup: cancel the three live tasks.
    for sid in ("run1-c1", "run1-c2", "run2-c1"):
        task = starter._active_tasks.get(sid)
        if task is not None:
            task.cancel()
    await asyncio.sleep(0)


async def test_active_runners_reaped_on_task_done(monkeypatch):
    """_active_runners must not leak: the done-callback pops it together with
    _active_tasks (lifespan-singleton starter would otherwise grow forever)."""

    class _ImmediateChildRunner:
        def __init__(self, **kwargs: Any) -> None: ...
        def attach_budget_callback(self, cb: Any) -> None: ...
        async def run_work_unit(self, **kwargs: Any) -> None:
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "CoordinatorChildRunner",
        _ImmediateChildRunner,
    )
    starter = _make_starter()
    await starter.start(
        coordinator_run_id="run-r", work_unit=_FakeWorkUnit("wu-r"),
        child_session_id="child-r", spawn_manifest_ref="ref",
        cancel_event=asyncio.Event(), root_session_id="root",
        parent_session_id="parent", parent_sandbox=MagicMock(),
        user_id="user-1",
    )
    for _ in range(50):
        if "child-r" not in starter._active_runners:
            break
        await asyncio.sleep(0.01)
    assert "child-r" not in starter._active_runners
    assert "child-r" not in starter._active_tasks


def _manifest_bytes_with_shell() -> bytes:
    """S2 manifest carrying the new write_tree_lease + shell_mode keys."""
    return json.dumps({
        "allowed_tools": ["file_read", "file_write"],
        "write_lease": [{"path": "a/b.py", "op": "modify", "base_digest": "b" * 64}],
        "write_tree_lease": [{"prefix": "workspace", "ops": ["add"]}],
        "shell_mode": True,
        "runtime_caps": [],
    }).encode("utf-8")


async def test_legacy_manifest_missing_shell_mode_decodes_typed_only(monkeypatch):
    """A 3-field legacy manifest ⇒ shell_mode False, tree_leases empty.

    The starter decodes the manifest into a SpawnManifest, builds a
    ChildPermissionContext from it, and hands that cpc to
    ``runner_factory.build(child_permission_context=..., ...)`` — so we read it
    back off the fake factory's recorded ``call_args`` (NOT a runner-ctor kwarg:
    CoordinatorChildRunner.__init__ takes no child_permission_context).
    """
    captured: dict = {}

    class _NoopChildRunner:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs

        def attach_budget_callback(self, cb):
            pass

        async def run_work_unit(self, **kwargs):
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "CoordinatorChildRunner",
        _NoopChildRunner,
    )

    runner_factory = _FakeRunnerFactory()
    # legacy 3-field payload from the existing _manifest_bytes() builder.
    starter = _make_starter(runner_factory=runner_factory)

    await starter.start(
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(), user_id="user-1",
    )

    cpc = runner_factory.call_args["child_permission_context"]
    assert cpc.spawn_manifest.shell_mode is False
    assert cpc.spawn_manifest.tree_leases == ()
    assert cpc.shell_mode is False


async def test_s2_manifest_round_trips_tree_lease_and_shell_mode(monkeypatch):
    from app.domain.models.work_unit import TreeLease

    captured: dict = {}

    class _NoopChildRunner:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs

        def attach_budget_callback(self, cb):
            pass

        async def run_work_unit(self, **kwargs):
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter."
        "CoordinatorChildRunner",
        _NoopChildRunner,
    )

    runner_factory = _FakeRunnerFactory()
    starter = _make_starter(runner_factory=runner_factory)
    # Swap the artifact payload to the S2 manifest carrying the new keys
    # (_make_starter wires a default _FakeArtifactStorage(_manifest_bytes());
    # the field is the public-by-convention `_artifact_storage` decode source).
    starter._artifact_storage = _FakeArtifactStorage(_manifest_bytes_with_shell())

    await starter.start(
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(), user_id="user-1",
    )

    cpc = runner_factory.call_args["child_permission_context"]
    assert cpc.spawn_manifest.shell_mode is True
    assert cpc.shell_mode is True
    assert cpc.spawn_manifest.tree_leases == (
        TreeLease(prefix="workspace", ops=frozenset({"add"})),
    )


async def test_malformed_string_shell_mode_rejected(monkeypatch):
    """[codex PR-3 R2 P1] A non-bool ``shell_mode`` (e.g. the string "false")
    must be REJECTED, not ``bool()``-coerced. ``bool("false")`` is True, so a
    malformed/tampered manifest would fail-OPEN to shell-mode. The decode
    requires a strict bool; a present non-bool propagates loudly (consistent
    with the malformed-manifest contract). Legacy-missing still defaults to the
    bool False, so it is unaffected (covered by the legacy test above)."""
    import json as _json

    runner_factory = _FakeRunnerFactory()
    starter = _make_starter(runner_factory=runner_factory)
    bad = _json.dumps({
        "allowed_tools": ["file_read"],
        "write_lease": [
            {"path": "a/b.py", "op": "modify", "base_digest": "b" * 64},
        ],
        "shell_mode": "false",  # string, not bool — bool("false") is True
        "runtime_caps": [],
    }).encode("utf-8")
    starter._artifact_storage = _FakeArtifactStorage(bad)

    with pytest.raises(ValueError, match="shell_mode"):
        await starter.start(
            coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
            child_session_id="child-1", spawn_manifest_ref="ref-1",
            cancel_event=asyncio.Event(), root_session_id="root-1",
            parent_session_id="parent-1", parent_sandbox=MagicMock(),
            user_id="user-1",
        )


def test_serialize_spawn_manifest_emits_s2_keys():
    from app.domain.models.work_unit import TreeLease, WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import (
        _serialize_spawn_manifest,
    )

    wu = WorkUnit(
        work_unit_id="wu-1", objective="o", phase="write",
        allowed_tools=["file_write"], write_lease=[],
        write_tree_lease=[TreeLease(prefix="workspace", ops=frozenset({"add"}))],
        shell_mode=True,
    )
    data = json.loads(_serialize_spawn_manifest(wu))
    assert data["shell_mode"] is True
    assert data["write_tree_lease"] == [{"prefix": "workspace", "ops": ["add"]}]


async def test_starter_passes_snapshot_limits_to_runner(monkeypatch):
    import app.application.services.coordinator_child_runner_starter as starter_mod

    captured = {}

    class _FakeChildRunner:
        def __init__(self, **kwargs):
            captured["ctor"] = kwargs

        def attach_budget_callback(self, cb):  # noqa: D401
            pass

        async def run_work_unit(self, **kwargs):
            return None

    monkeypatch.setattr(starter_mod, "CoordinatorChildRunner", _FakeChildRunner)
    starter = _make_starter()
    await starter.start(
        coordinator_run_id="run-1", work_unit=_FakeWorkUnit("wu-1"),
        child_session_id="child-1", spawn_manifest_ref="ref-1",
        cancel_event=asyncio.Event(), root_session_id="root-1",
        parent_session_id="parent-1", parent_sandbox=MagicMock(), user_id="user-1",
    )
    sl = captured["ctor"]["snapshot_limits"]
    assert sl.max_snapshot_paths == 20000
    assert sl.max_snapshot_seconds == 30.0


def test_resolve_child_llm_price_finds_glm_5_2() -> None:
    """[child-budget fix] Live regression: the deployed glm-5.2 child llm must
    resolve a static price so BudgetEnforcementCallback attaches instead of
    hitting the 'no static price … token budget enforcement DISABLED' rung."""
    from types import SimpleNamespace

    from app.application.services.coordinator_child_runner_starter import (
        _resolve_child_llm_price,
    )

    llm = SimpleNamespace(
        _identifying_params={"provider_id": "glm", "model": "glm-5.2"},
        model_name="glm-5.2",
    )
    price = _resolve_child_llm_price(llm)
    assert price is not None
    assert price["input"] > 0 and price["output"] > 0


# ── SPM PR-1c Task 17: child_spawn trigger three-classification (§5.2d) ──


class _RecMetrics:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, str]] = []

    def record_provision(
        self, *, mode, trigger, outcome, latency_seconds=None, latency=None
    ) -> None:
        self.records.append((mode, trigger, outcome))

    def last(self, *, trigger: str):
        for mode, trg, outcome in reversed(self.records):
            if trg == trigger:
                return outcome
        return None


def _make_starter_with_metrics(lifecycle, metrics):
    from unittest.mock import AsyncMock

    from app.application.services.coordinator_child_runner_starter import (
        DefaultCoordinatorChildRunnerStarter,
    )

    envelope_factory = MagicMock()
    envelope_factory.make_result_ready = MagicMock(return_value="ENVELOPE")
    publisher = MagicMock()
    publisher.publish = AsyncMock()
    return DefaultCoordinatorChildRunnerStarter(
        runner_factory=_FakeRunnerFactory(),
        mailbox_publisher=publisher,
        mailbox_subscriber=MagicMock(),
        envelope_factory=envelope_factory,
        session_repository=_FakeSessionRepository(),
        coordinator_envelope_store=MagicMock(),
        cost_rollup_service=MagicMock(),
        artifact_storage=_FakeArtifactStorage(_manifest_bytes()),
        coordinator_limits=_FakeCoordinatorLimits(),
        sandbox_lifecycle_service=lifecycle,
        resolve_child_runner_deps=_fake_resolve,
        sandbox_provision_metrics=metrics,
    )


async def _start_child(starter):
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


async def test_starter_ctor_accepts_optional_provision_metrics():
    starter = _make_starter()
    assert starter._provision_metrics is None


async def test_child_spawn_records_failed_on_bind_error():
    from unittest.mock import AsyncMock

    metrics = _RecMetrics()
    lifecycle = _fake_lifecycle()
    lifecycle.bind_new = AsyncMock(side_effect=RuntimeError("bind boom"))
    starter = _make_starter_with_metrics(lifecycle, metrics)
    # RuntimeError → leak-guard publishes FAILED + swallows → start() does NOT raise
    await _start_child(starter)
    assert metrics.last(trigger="child_spawn") == "failed"


async def test_child_spawn_records_cancelled_on_bind_cancel():
    from unittest.mock import AsyncMock

    metrics = _RecMetrics()
    lifecycle = _fake_lifecycle()
    lifecycle.bind_new = AsyncMock(side_effect=asyncio.CancelledError())
    starter = _make_starter_with_metrics(lifecycle, metrics)
    # CancelledError → leak-guard publishes FAILED then re-raises the cancel
    with pytest.raises(asyncio.CancelledError):
        await _start_child(starter)
    assert metrics.last(trigger="child_spawn") == "cancelled"


async def test_child_spawn_records_ok_on_success(monkeypatch):
    from unittest.mock import AsyncMock

    metrics = _RecMetrics()
    lifecycle = _fake_lifecycle()

    class _NoopChildRunner:
        def __init__(self, **kwargs):
            ...

        def attach_budget_callback(self, cb) -> None:
            ...

        async def run_work_unit(self, **kwargs):
            return None

    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner_starter.CoordinatorChildRunner",
        _NoopChildRunner,
    )
    runner_factory = MagicMock()
    runner_factory.build = AsyncMock(return_value=MagicMock())
    starter = _make_starter_with_metrics(lifecycle, metrics)
    starter._runner_factory = runner_factory
    await _start_child(starter)
    assert metrics.last(trigger="child_spawn") == "ok"
