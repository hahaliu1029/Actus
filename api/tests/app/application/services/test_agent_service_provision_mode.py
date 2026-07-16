"""SPM PR-1c Task 17 — ``AgentService._create_task`` provision-mode fork.

Covers the frozen ``TestCreateTaskProvisionFork`` contract (task brief):

* on_demand → ZERO lifecycle calls (要点1), OnDemand accessors on the runner,
  provisioner owner = SESSION OWNER not requester (INV-SPM-12), deferred
  file_processor_factory threading.
* always → byte-identical eager bind + non-provisioner ``run_start`` outcome
  three-classification (ok / failed / cancelled — spec §5.2d).

Reuses the ``test_agent_service_sandbox_binding_persistence`` scaffolding style
(fake UoW/session repo/task cls) but swaps in a fake lifecycle exposing
acquire/bind counters + a recording metrics sink.
"""
from __future__ import annotations

import asyncio
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.agent_service import AgentService
from app.application.services.sandbox_accessors import (
    OnDemandBrowserAccessor,
    OnDemandSandboxAccessor,
)
from app.domain.errors.sandbox_lifecycle import SessionUnboundError
from app.domain.models.session import Session, SessionStatus
from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ── recording metrics fake (implements the SandboxProvisionMetrics surface) ──


class _MetricRecord:
    def __init__(self, mode: str, trigger: str, outcome: str, latency) -> None:
        self.mode = mode
        self.trigger = trigger
        self.outcome = outcome
        self.latency = latency


class _RecordingMetrics:
    def __init__(self) -> None:
        self.records: list[_MetricRecord] = []

    def record_provision(
        self, *, mode, trigger, outcome, latency_seconds=None, latency=None
    ) -> None:
        lat = latency_seconds if latency_seconds is not None else latency
        self.records.append(_MetricRecord(mode, trigger, outcome, lat))

    def record_attachment_skipped(self, *, reason: str) -> None:  # pragma: no cover
        pass

    def last(self, *, trigger: str) -> Optional[_MetricRecord]:
        for r in reversed(self.records):
            if r.trigger == trigger:
                return r
        return None


# ── fake sandbox handle + lifecycle (acquire raises UNBOUND → bind_new) ──


class _FakeHandle:
    def __init__(self, hid: str = "h-on-demand") -> None:
        self.id = hid

    async def get_browser(self):
        return MagicMock()

    def release(self) -> None:
        return None


class _FakeLifecycle:
    """Fresh-session lifecycle: acquire → UNBOUND, bind_new → handle (or raises)."""

    def __init__(self) -> None:
        self.acquire_calls = 0
        self.bind_calls = 0
        self.bind_raises: Optional[BaseException] = None
        # live-event-sink registration target used post-task construction.
        self.registry = MagicMock()

    async def acquire(self, session_id: str) -> _FakeHandle:
        self.acquire_calls += 1
        raise SessionUnboundError(session_id)

    async def bind_new(self, session_id: str, *, user_id: str | None = None) -> _FakeHandle:
        self.bind_calls += 1
        if self.bind_raises is not None:
            raise self.bind_raises
        return _FakeHandle(hid="h-always")


# ── fake sandbox_cls (tracks .create so we can assert zero containers) ──


class _CountingSandboxClass:
    created: list[str] = []

    @classmethod
    async def create(cls, user_id: str | None = None, **_kw):
        cls.created.append(user_id or "?")
        return _FakeHandle(hid=f"sbx-{len(cls.created)}")

    @classmethod
    async def get(cls, sandbox_id: str):
        return None


# ── fake UoW / session repo / task cls (mirrors binding-persistence test) ──


class _SessionRepo:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        s = self._sessions.get(session_id)
        return s.model_copy(deep=True) if s is not None else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def add_event(self, session_id, event) -> None:
        return None


class _UoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.session = _SessionRepo(sessions)
        self.sandbox_lifecycle_log = MagicMock(create=AsyncMock())

    async def __aenter__(self) -> "_UoW":
        return self

    async def __aexit__(self, *a) -> None:
        return None


class _DummyTask:
    def __init__(self) -> None:
        self.id = "task-1"
        self.output_stream = MagicMock()
        self.output_stream.put = AsyncMock(return_value="evt-1")


class _DummyTaskClass:
    @classmethod
    def create(cls, task_runner):
        return _DummyTask()


# ── the env factory the brief's tests consume ──


class _Env:
    def __init__(self, session, lifecycle, sandbox_cls, metrics, runner_kwargs) -> None:
        self.session = session
        self.lifecycle = lifecycle
        self.sandbox_cls = sandbox_cls
        self.metrics = metrics
        self.runner_kwargs = runner_kwargs
        self._service = None

    async def create_task(self):
        return await self._service._create_task(self.session)

    @property
    def captured_provisioner(self):
        # provisioner is threaded into the runner kwargs (on_demand only)
        return self.runner_kwargs.get("sandbox_provisioner")


@pytest.fixture
def agent_service_env(monkeypatch):
    def _make(*, mode: str = "always", requester_id: str | None = None):
        # monkeypatch settings mode + ALLOWED (bypass the field validator gate;
        # on_demand is not yet in the ALLOWED set until PR-2 — Task 17 unit-tests
        # the runtime code path regardless).
        from core.config import Settings, get_settings

        settings = get_settings()
        monkeypatch.setattr(settings, "sandbox_provision_mode", mode, raising=False)
        monkeypatch.setattr(
            Settings,
            "SANDBOX_PROVISION_MODE_ALLOWED",
            {"always", "on_demand", "off"},
            raising=False,
        )

        _CountingSandboxClass.created = []
        session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.PENDING)
        sessions = {"sess-1": session}
        uow = _UoW(sessions)
        lifecycle = _FakeLifecycle()
        metrics = _RecordingMetrics()

        service = AgentService(
            uow_factory=lambda: uow,
            config_snapshot=_default_snapshot(),
            sandbox_cls=_CountingSandboxClass,
            task_cls=_DummyTaskClass,
            search_engine=MagicMock(),
            file_storage=MagicMock(),
            sandbox_lifecycle_service=lifecycle,
            sandbox_provision_metrics=metrics,
        )
        service._supervisor = MagicMock()
        # master-off so _create_task skips the PE build (not under test here).
        service._config_snapshot.agent_config.tool_confirmation.enabled = False

        runner_kwargs: dict = {}

        def _capture_runner(*args, **kwargs):
            runner_kwargs.clear()
            runner_kwargs.update(kwargs)
            # the mock runner must expose start_deferred_skill_sync for hook ②
            # registration (registration only references it — never calls at create).
            m = MagicMock()
            m.start_deferred_skill_sync = AsyncMock()
            return m

        monkeypatch.setattr(
            "app.application.services.agent_service.AgentTaskRunner", _capture_runner
        )

        env = _Env(session, lifecycle, _CountingSandboxClass, metrics, runner_kwargs)
        env._service = service
        return env

    return _make


# ── TestCreateTaskProvisionFork ──


class TestCreateTaskProvisionFork:
    async def test_on_demand_creates_zero_sandbox(self, agent_service_env) -> None:
        env = agent_service_env(mode="on_demand")
        await env.create_task()
        assert env.lifecycle.acquire_calls == 0
        assert env.lifecycle.bind_calls == 0
        assert env.sandbox_cls.created == []  # 纯聊天零容器（G1）

    async def test_on_demand_runner_receives_ondemand_accessors(
        self, agent_service_env
    ) -> None:
        env = agent_service_env(mode="on_demand")
        await env.create_task()
        assert isinstance(env.runner_kwargs["sandbox_accessor"], OnDemandSandboxAccessor)
        assert isinstance(env.runner_kwargs["browser_accessor"], OnDemandBrowserAccessor)
        assert env.runner_kwargs["sandbox_provision_mode"] == "on_demand"
        # attachment flusher + provisioner threaded for incremental-flush backstop.
        assert env.runner_kwargs["attachment_flusher"] is not None
        assert env.runner_kwargs["sandbox_provisioner"] is not None

    async def test_provisioner_owner_is_session_user_not_requester(
        self, agent_service_env
    ) -> None:
        """INV-SPM-12 owner 合同：provisioner user_id == session owner."""
        env = agent_service_env(mode="on_demand", requester_id="admin-999")
        await env.create_task()
        assert env.captured_provisioner._user_id == env.session.user_id
        assert env.captured_provisioner._user_id == "owner-1"

    async def test_on_demand_threads_file_processor_factory(
        self, agent_service_env
    ) -> None:
        import dataclasses

        env = agent_service_env(mode="on_demand")
        # give the snapshot a file_understanding_config so the factory branch runs
        # (_ConfigSnapshot is a frozen dataclass → rebuild via replace).
        env._service._config_snapshot = dataclasses.replace(
            env._service._config_snapshot, file_understanding_config=MagicMock()
        )
        await env.create_task()
        assert callable(env.runner_kwargs["file_processor_factory"])
        assert env.runner_kwargs["file_processor_lookup"] is None

    async def test_always_path_byte_identical(self, agent_service_env) -> None:
        env = agent_service_env(mode="always")
        await env.create_task()
        assert env.lifecycle.bind_calls == 1  # 现状急切
        # always: eager accessors, no provisioner/factory
        assert env.runner_kwargs["sandbox_provisioner"] is None
        assert env.runner_kwargs["file_processor_factory"] is None
        assert env.runner_kwargs["sandbox_provision_mode"] == "always"

    async def test_always_run_start_records_ok(self, agent_service_env) -> None:
        env = agent_service_env(mode="always")
        await env.create_task()
        assert env.metrics.last(trigger="run_start").outcome == "ok"

    async def test_always_run_start_records_failed_on_error(
        self, agent_service_env
    ) -> None:
        env = agent_service_env(mode="always")
        env.lifecycle.bind_raises = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            await env.create_task()
        assert env.metrics.last(trigger="run_start").outcome == "failed"

    async def test_always_run_start_records_cancelled_on_cancel(
        self, agent_service_env
    ) -> None:
        env = agent_service_env(mode="always")
        env.lifecycle.bind_raises = asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await env.create_task()
        assert env.metrics.last(trigger="run_start").outcome == "cancelled"

    async def test_on_demand_records_nothing_at_create(self, agent_service_env) -> None:
        """on_demand defers provision → no run_start record at create time
        (the provisioner emits its own tool_call/skill_sync metrics later)."""
        env = agent_service_env(mode="on_demand")
        await env.create_task()
        assert env.metrics.records == []
