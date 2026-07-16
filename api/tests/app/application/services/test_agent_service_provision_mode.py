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

from app.application.errors.exceptions import SandboxDisabledError
from app.application.services.agent_service import AgentService
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
    OnDemandBrowserAccessor,
    OnDemandSandboxAccessor,
)
from app.domain.errors.sandbox_lifecycle import SessionUnboundError
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import MessageEvent
from app.domain.models.session import Session, SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)
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
    """Fresh-session lifecycle: acquire → UNBOUND, bind_new → handle (or raises).

    SPM PR-3 Task 25 extends the fake with suspend/resume/destroy counters +
    a ``total_calls`` aggregate so the off zero-touch group can assert that the
    ``off`` composition never mutates any sandbox lifecycle state (INV-SPM-3).
    """

    def __init__(self) -> None:
        self.acquire_calls = 0
        self.bind_calls = 0
        self.suspend_calls = 0
        self.resume_calls = 0
        self.destroy_calls = 0
        self.bind_raises: Optional[BaseException] = None
        # live-event-sink registration target used post-task construction.
        self.registry = MagicMock()

    @property
    def total_calls(self) -> int:
        return (
            self.acquire_calls
            + self.bind_calls
            + self.suspend_calls
            + self.resume_calls
            + self.destroy_calls
        )

    async def acquire(self, session_id: str) -> _FakeHandle:
        self.acquire_calls += 1
        raise SessionUnboundError(session_id)

    async def bind_new(self, session_id: str, *, user_id: str | None = None) -> _FakeHandle:
        self.bind_calls += 1
        if self.bind_raises is not None:
            raise self.bind_raises
        return _FakeHandle(hid="h-always")

    async def suspend(self, session_id: str) -> None:
        self.suspend_calls += 1

    async def resume(self, session_id: str) -> None:
        self.resume_calls += 1

    async def destroy(self, session_id: str, reason=None) -> None:
        self.destroy_calls += 1


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
        # monkeypatch settings mode + ALLOWED (bypass the field validator gate).
        # on_demand is unlocked in the production ALLOWED set since PR-2; `off`
        # stays REJECTED at the config layer (ALLOWED={always,on_demand}) until
        # PR-4, so the monkeypatched ALLOWED below is still required to unit-test
        # the off runtime code path.
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


# ══════════════════════════════════════════════════════════════════════════════
# SPM PR-3 Task 25 — off composition graph (INV-SPM-3 zero-touch + CLASS-3 gates)
# ══════════════════════════════════════════════════════════════════════════════


def _set_mode(monkeypatch, mode: str) -> None:
    """Monkeypatch the process settings singleton to ``mode`` and unlock the
    ALLOWED set so the config field-validator (which still REJECTS ``off`` until
    PR-4) doesn't block the runtime code path under test."""
    from core.config import Settings, get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_provision_mode", mode, raising=False)
    monkeypatch.setattr(
        Settings,
        "SANDBOX_PROVISION_MODE_ALLOWED",
        {"always", "on_demand", "off"},
        raising=False,
    )


# ── real off-runner builder (genuine ctor + run-startup None-deref exercise) ──


class _RunnerNoopSessionRepo:
    def __init__(self) -> None:
        self.status_updates: list = []

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def update_to_terminal(self, session_id: str, status, terminal_reason: str) -> None:
        return None

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        return None


class _RunnerNoopDbSession:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _RunnerNoopUoW:
    def __init__(self) -> None:
        self.session = _RunnerNoopSessionRepo()
        self.db_session = _RunnerNoopDbSession()

    async def __aenter__(self) -> "_RunnerNoopUoW":
        return self

    async def __aexit__(self, *a) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def _runner_uow_factory() -> _RunnerNoopUoW:
    return _RunnerNoopUoW()


class _OffWaitFlow:
    """Flow double that yields a single WaitEvent so ``invoke()`` runs the full
    run-startup sequence (peek recheck, skill-bundle-sync startup, flow getters,
    attachment routing) then short-circuits to WAITING — a None-deref in any off
    startup path surfaces as a crash here."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self._skill_context_provider = None

    async def invoke(self, message):
        from app.domain.models.event import WaitEvent

        yield WaitEvent()

    async def close(self) -> None:
        return None


class _OffFakeMCPTool:
    async def initialize(self, mcp_config) -> None:
        return None

    async def cleanup(self) -> None:
        return None

    def get_tools(self):
        return []


class _OffFakeA2ATool:
    def __init__(self) -> None:
        self.manager = None

    async def initialize(self, a2a_config) -> None:
        return None

    async def cleanup(self) -> None:
        return None


class _OffSingleMessageInputStream:
    def __init__(self, message: str) -> None:
        self._items = [
            ("evt-1", MessageEvent(role="user", message=message, attachments=[]).model_dump_json())
        ]

    async def is_empty(self) -> bool:
        return len(self._items) == 0

    async def pop(self):
        return self._items.pop(0)


class _OffOutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return f"event-{len(self.events)}"


class _OffMessageTask:
    def __init__(self, message: str) -> None:
        self.input_stream = _OffSingleMessageInputStream(message)
        self.output_stream = _OffOutputStream()


class _OffFakeSandbox:
    async def ensure_sandbox(self) -> None:
        return None


_SENTINEL = object()


def _mk_runner(*, mode: str, sandbox_accessor=_SENTINEL, browser_accessor=_SENTINEL,
               skill_creator_service=None) -> AgentTaskRunner:
    off = mode == "off"
    if sandbox_accessor is _SENTINEL:
        sandbox_accessor = None if off else EagerSandboxAccessor(_OffFakeSandbox())
    if browser_accessor is _SENTINEL:
        browser_accessor = None if off else EagerBrowserAccessor(object())
    return AgentTaskRunner(
        uow_factory=_runner_uow_factory,
        session_state_machine=DefaultSessionStateMachine(uow_factory=_runner_uow_factory),
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id="sess-off",
        user_id="user-off",
        file_storage=object(),
        browser_accessor=browser_accessor,
        search_engine=object(),
        sandbox_accessor=sandbox_accessor,
        sandbox_provision_mode=mode,
        skill_creator_service=skill_creator_service,
    )


# ── rich AgentService gate-drive harness (CLASS-3 lifecycle-write sites) ──


class _GateSessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session
        self.terminal_calls: list = []
        self.status_calls: list = []
        self.latest_message_calls: list = []

    async def get_by_id(self, session_id: str):
        return self._session if self._session and self._session.id == session_id else None

    async def save(self, session: Session) -> None:
        self._session = session

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_status(self, session_id: str, status) -> None:
        self.status_calls.append((session_id, status))

    async def update_to_terminal(self, session_id: str, status, terminal_reason: str) -> None:
        self.terminal_calls.append((session_id, status, terminal_reason))
        if self._session and self._session.id == session_id:
            self._session.status = status

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        self.latest_message_calls.append((session_id, message))

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None


class _GateUoW:
    def __init__(self, session: Session) -> None:
        self.session = _GateSessionRepo(session)

    async def __aenter__(self) -> "_GateUoW":
        return self

    async def __aexit__(self, *a) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _OffChatInputStream:
    async def put(self, event_json: str) -> str:
        return "evt-user-1"


class _OffChatOutputStream:
    def __init__(self, owner: "_OffChatTask") -> None:
        self._owner = owner

    async def get(self, start_id: str = None, block_ms: int = None):
        self._owner.done_flag = True
        return None, None


class _OffChatTask:
    """Fast-terminating task double for the ``chat()`` re-chat drive: the first
    ``output_stream.get`` marks done so the output loop exits immediately."""

    def __init__(self) -> None:
        self.done_flag = False
        self.input_stream = _OffChatInputStream()
        self.output_stream = _OffChatOutputStream(self)

    @property
    def done(self) -> bool:
        return self.done_flag

    async def invoke(self) -> None:
        return None


def _make_gate_service(session: Session, lifecycle: _FakeLifecycle) -> AgentService:
    uow = _GateUoW(session)
    service = AgentService(
        uow_factory=lambda: uow,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
        sandbox_lifecycle_service=lifecycle,
    )
    service._supervisor = MagicMock()
    return service


class TestOffComposition:
    async def test_off_pure_chat_zero_sandbox_surface(
        self, agent_service_env, monkeypatch
    ) -> None:
        """INV-SPM-3 核心：off 纯聊天全程零沙箱面 + genuine run-startup 序（真跑
        ``invoke()`` + ``_build_lc_tools_full()``，任一 None-deref 会崩）。"""
        # (a) AgentService _create_task(off) builds ZERO sandbox surface.
        env = agent_service_env(mode="off")
        await env.create_task()
        assert env.lifecycle.total_calls == 0  # acquire/bind/resume/suspend/destroy 全零
        assert env.sandbox_cls.created == []  # 零容器
        # off runner gets None accessors + no browser/registry/flusher/provisioner.
        assert env.runner_kwargs["sandbox_accessor"] is None
        assert env.runner_kwargs["browser_accessor"] is None  # 零 browser
        assert env.runner_kwargs["sandbox_provision_mode"] == "off"
        assert env.runner_kwargs["file_processor_lookup"] is None  # 零 registry
        assert env.runner_kwargs["file_processor_factory"] is None
        assert env.runner_kwargs["attachment_flusher"] is None
        assert env.runner_kwargs["sandbox_provisioner"] is None

        # (b) Genuine run-startup: a real off runner must build tools + run invoke()
        # start-up without a None-deref (settings already off from the env fixture).
        monkeypatch.setattr(
            "app.domain.services.agent_task_runner.PlannerReActFlow", _OffWaitFlow
        )
        runner = _mk_runner(mode="off")
        runner._mcp_tool = _OffFakeMCPTool()
        runner._a2a_tool = _OffFakeA2ATool()
        assert runner._skill_bundle_sync is None

        async def _no_prefs(tool_type):
            return {}

        async def _no_skills():
            return []

        monkeypatch.setattr(runner, "_load_user_preferences_map", _no_prefs)
        monkeypatch.setattr(runner, "_load_enabled_skills", _no_skills)

        # tool assembly must not None-deref the skill-guide bundle-sync refs.
        tool_names = [t.name for t in runner._build_lc_tools_full()]
        assert "file_read" not in tool_names  # off → zero sandbox tool face
        assert "shell_execute" not in tool_names

        # full run-startup sequence (peek recheck, skill-bundle-sync startup, flow
        # getters, attachment routing) must complete → WAITING, no crash.
        await runner.invoke(_OffMessageTask("hello"))
        assert runner._uow.session.status_updates[-1] == (
            "sess-off",
            SessionStatus.WAITING,
        )

    def test_runner_ctor_bidirectional_assert(self) -> None:
        # off MUST have both accessors None → a non-None accessor is mis-wiring.
        with pytest.raises(AssertionError):
            _mk_runner(mode="off", sandbox_accessor=EagerSandboxAccessor(_OffFakeSandbox()))
        # non-off MUST have both accessors non-None → None is mis-wiring.
        with pytest.raises(AssertionError):
            _mk_runner(mode="always", sandbox_accessor=None)

    async def test_off_runner_skips_skill_creation_tools(self, monkeypatch) -> None:
        _set_mode(monkeypatch, "off")
        monkeypatch.setattr(
            "app.domain.services.agent_task_runner.PlannerReActFlow", _OffWaitFlow
        )
        # Even WITH a skill_creator_service present, off skips the creation tools.
        runner = _mk_runner(mode="off", skill_creator_service=MagicMock())
        runner._mcp_tool = _OffFakeMCPTool()
        runner._a2a_tool = _OffFakeA2ATool()
        assert runner._skill_bundle_sync is None
        assert runner._create_skill_tool is None
        assert runner._brainstorm_skill_tool is None
        tool_names = [t.name for t in runner._build_lc_tools_full()]
        assert "generate_skill" not in tool_names
        assert "brainstorm_skill" not in tool_names

    async def test_off_completion_zero_lifecycle(self, monkeypatch) -> None:
        """完成回调 ``_on_task_runner_complete`` off skip suspend（历史 binding 不被改写）。"""
        session = Session(
            id="sess-1", user_id="owner-1", status=SessionStatus.RUNNING, worker_type="root"
        )

        async def _get_session(_sid):
            return session

        # off → zero suspend.
        _set_mode(monkeypatch, "off")
        lc_off = _FakeLifecycle()
        svc_off = _make_gate_service(session, lc_off)
        monkeypatch.setattr(svc_off, "get_session", _get_session)
        await svc_off._on_task_runner_complete("sess-1")
        assert lc_off.suspend_calls == 0

        # control (non-inertness): non-off DOES suspend the root binding.
        _set_mode(monkeypatch, "always")
        lc_on = _FakeLifecycle()
        svc_on = _make_gate_service(session, lc_on)
        monkeypatch.setattr(svc_on, "get_session", _get_session)
        await svc_on._on_task_runner_complete("sess-1")
        assert lc_on.suspend_calls == 1

    async def test_off_rechat_completed_session_zero_resume(self, monkeypatch) -> None:
        """COMPLETED 会话二次聊天 off skip resume（历史 SUSPENDED binding 不复活）。"""

        async def _drive(mode: str) -> _FakeLifecycle:
            _set_mode(monkeypatch, mode)
            session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.COMPLETED)
            lc = _FakeLifecycle()
            svc = _make_gate_service(session, lc)

            async def _accessible(*a, **k):
                return session

            async def _check(*a, **k):
                return None

            async def _get_task(_s):
                return None

            created = _OffChatTask()

            async def _create(_s, *, tool_filter=None, force_initial_compaction=False):
                return created

            async def _unread(_sid):
                return None

            monkeypatch.setattr(svc, "_get_accessible_session", _accessible)
            monkeypatch.setattr(svc, "_check_attachments_access", _check)
            monkeypatch.setattr(svc, "_get_task", _get_task)
            monkeypatch.setattr(svc, "_create_task", _create)
            monkeypatch.setattr(svc, "_safe_update_unread_count", _unread)

            gen = svc.chat(session_id="sess-1", user_id="owner-1", message="再问一句")
            # first event is the echoed user message (resume gate already fired).
            first = await asyncio.wait_for(gen.__anext__(), timeout=0.3)
            assert first.type == "message"
            await gen.aclose()
            return lc

        assert (await _drive("off")).resume_calls == 0
        assert (await _drive("always")).resume_calls == 1

    async def test_off_status_reconcile_zero_suspend(self, monkeypatch) -> None:
        """RUNNING-but-task-lost status-reconcile off skip suspend。"""

        async def _drive(mode: str) -> _FakeLifecycle:
            _set_mode(monkeypatch, mode)
            session = Session(
                id="sess-1", user_id="owner-1", status=SessionStatus.RUNNING, worker_type="root"
            )
            lc = _FakeLifecycle()
            svc = _make_gate_service(session, lc)

            async def _accessible(*a, **k):
                return session

            async def _check(*a, **k):
                return None

            async def _get_task(_s):
                return None

            async def _unread(_sid):
                return None

            async def _noop(*a, **k):
                return None

            monkeypatch.setattr(svc, "_get_accessible_session", _accessible)
            monkeypatch.setattr(svc, "_check_attachments_access", _check)
            monkeypatch.setattr(svc, "_get_task", _get_task)
            monkeypatch.setattr(svc, "_safe_update_unread_count", _unread)
            monkeypatch.setattr(
                svc, "_emit_bg_terminal_notification_if_background", _noop
            )
            monkeypatch.setattr(svc, "_maybe_stop_supervisor_for_session", _noop)

            gen = svc.chat(session_id="sess-1", user_id="owner-1", message=None)
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(gen.__anext__(), timeout=0.3)
            return lc

        assert (await _drive("off")).suspend_calls == 0
        assert (await _drive("always")).suspend_calls == 1

    async def test_off_stop_session_zero_suspend(self, monkeypatch) -> None:
        """公开 stop-session 路径 off skip suspend（会话终态转移照常）。"""

        async def _drive(mode: str) -> _FakeLifecycle:
            _set_mode(monkeypatch, mode)
            session = Session(
                id="sess-1", user_id="owner-1", status=SessionStatus.RUNNING, worker_type="root"
            )
            lc = _FakeLifecycle()
            svc = _make_gate_service(session, lc)

            async def _accessible(*a, **k):
                return session

            async def _get_task(_s):
                return None

            async def _noop(*a, **k):
                return None

            monkeypatch.setattr(svc, "_get_accessible_session", _accessible)
            monkeypatch.setattr(svc, "_get_task", _get_task)
            monkeypatch.setattr(
                svc, "_emit_bg_terminal_notification_if_background", _noop
            )
            monkeypatch.setattr(svc, "_maybe_stop_supervisor_for_session", _noop)
            monkeypatch.setattr(svc, "_cleanup_background_slot_if_needed", _noop)
            await svc.stop_session("sess-1", "owner-1")
            return lc

        lc_off = await _drive("off")
        assert lc_off.suspend_calls == 0
        # session still transitioned to terminal (state progression intact).
        lc_on = await _drive("always")
        assert lc_on.suspend_calls == 1

    async def test_off_delete_session_zero_lifecycle_destroy(self, monkeypatch) -> None:
        """session delete off skip lifecycle.destroy（行删除照常）。"""
        from app.application.services.session_service import SessionService

        class _DeleteRepo:
            def __init__(self, session: Session) -> None:
                self._session = session
                self.deleted: list[str] = []

            async def get_by_id(self, session_id: str):
                return self._session if session_id == self._session.id else None

            async def find_descendants(self, session_id, *, user_id, max_depth, limit):
                return []

            async def delete_by_id(self, session_id: str) -> None:
                self.deleted.append(session_id)

        class _DeleteUoW:
            def __init__(self, session: Session) -> None:
                self.session = _DeleteRepo(session)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return None

        async def _drive(mode: str):
            _set_mode(monkeypatch, mode)
            session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.COMPLETED)
            uow = _DeleteUoW(session)
            lc = _FakeLifecycle()
            svc = SessionService(uow_factory=lambda: uow, sandbox_lifecycle_service=lc)
            await svc.delete_session("sess-1", "owner-1")
            return lc, uow

        lc_off, uow_off = await _drive("off")
        assert lc_off.destroy_calls == 0
        assert uow_off.session.deleted == ["sess-1"]  # 行删除照常

        lc_on, uow_on = await _drive("always")
        assert lc_on.destroy_calls == 1
        assert uow_on.session.deleted == ["sess-1"]

    # ── Task 28: off endpoint 409 at AgentService (INV-SPM-7) + zero lifecycle ──

    async def test_off_start_takeover_raises_disabled_zero_lifecycle(
        self, monkeypatch
    ) -> None:
        """off takeover/start → 409 SANDBOX_DISABLED at method top, before any
        lifecycle/lease touch (INV-SPM-7)."""
        _set_mode(monkeypatch, "off")
        session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.RUNNING)
        lc = _FakeLifecycle()
        svc = _make_gate_service(session, lc)
        with pytest.raises(SandboxDisabledError) as exc:
            await svc.start_takeover(session_id="sess-1", user_id="owner-1")
        assert exc.value.msg == "SANDBOX_DISABLED"
        assert exc.value.status_code == 409 and exc.value.code == 409
        assert lc.total_calls == 0

    async def test_off_reopen_takeover_raises_disabled_zero_lifecycle(
        self, monkeypatch
    ) -> None:
        """off takeover/reopen → 409 SANDBOX_DISABLED before any state transition."""
        _set_mode(monkeypatch, "off")
        session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.COMPLETED)
        lc = _FakeLifecycle()
        svc = _make_gate_service(session, lc)
        with pytest.raises(SandboxDisabledError) as exc:
            await svc.reopen_takeover(session_id="sess-1", user_id="owner-1")
        assert exc.value.msg == "SANDBOX_DISABLED"
        assert lc.total_calls == 0

    async def test_off_retry_from_suspend_raises_disabled_zero_lifecycle(
        self, monkeypatch
    ) -> None:
        """r14/codex R13(L): off retry-from-suspend → 409 + ZERO lifecycle touch.
        The lifecycle ``resume`` that would revive the suspended sandbox
        (INV-SPM-9 whitelist member) is unreachable — off-check is at method top,
        ahead of ``_get_accessible_session`` + the ``supervisor.resume`` call."""
        _set_mode(monkeypatch, "off")
        session = Session(id="sess-1", user_id="owner-1", status=SessionStatus.RUNNING)
        lc = _FakeLifecycle()
        svc = _make_gate_service(session, lc)
        with pytest.raises(SandboxDisabledError) as exc:
            await svc.retry_from_suspend(session_id="sess-1", user_id="owner-1")
        assert exc.value.msg == "SANDBOX_DISABLED"
        assert lc.total_calls == 0             # no lifecycle acquire/bind/resume/suspend
        svc._supervisor.resume.assert_not_called()  # supervisor.resume never reached
