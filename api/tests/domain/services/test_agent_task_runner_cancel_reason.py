from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import MessageEvent
from app.domain.models.lifecycle import LifecycleEventKind
from app.domain.models.session import SessionStatus
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, object]] = []
        self.terminal_updates: list[tuple[str, object, str]] = []
        self.add_event_calls: list[tuple[str, object]] = []

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def update_to_terminal(
        self,
        session_id: str,
        status,
        terminal_reason: str,
    ) -> None:
        self.terminal_updates.append((session_id, status, terminal_reason))

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _NoopDbSession:
    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _NoopUoW:
    # B4 Issue 1D: _set_terminal_status now calls self._uow_factory() to get a
    # fresh UoW for the terminal status write (not self._uow). All UoW instances
    # created by _uow_factory share the same _shared_session so that
    # status_updates are observable via runner._uow.session regardless of which
    # factory call produced the write.
    def __init__(self, shared_session: "_NoopSessionRepository | None" = None) -> None:
        self.session = shared_session if shared_session is not None else _NoopSessionRepository()
        self.db_session = _NoopDbSession()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def _make_uow_factory() -> tuple["_NoopUoW", "type[_NoopUoW]"]:
    """Return (root_uow, factory_fn) sharing the same session repository."""
    shared_session = _NoopSessionRepository()
    root_uow = _NoopUoW(shared_session=shared_session)

    def _factory() -> _NoopUoW:
        return _NoopUoW(shared_session=shared_session)

    return root_uow, _factory


class _NoopSandbox:
    async def ensure_sandbox(self) -> None:
        return None


class _NoopTool:
    manager = None

    def connected_server_ids(self) -> tuple[str, ...]:
        return ()

    async def initialize(self, *_args, **_kwargs) -> None:
        return None

    async def cleanup(self) -> None:
        return None


class _InputStream:
    async def is_empty(self) -> bool:
        return True

    async def pop(self):
        return None, None


class _OutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return f"event-{len(self.events)}"


class _DummyTask:
    def __init__(self, cancel_reason: str) -> None:
        self.cancel_reason = cancel_reason
        self.input_stream = _InputStream()
        self.output_stream = _OutputStream()


def _build_runner(session_id: str = "session-cancel") -> AgentTaskRunner:
    root_uow, factory = _make_uow_factory()
    runner = AgentTaskRunner(
        uow_factory=factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id=session_id,
        user_id="user-1",
        file_storage=object(),
        browser_accessor=EagerBrowserAccessor(object()),
        search_engine=object(),
        sandbox_accessor=EagerSandboxAccessor(_NoopSandbox()),
        session_state_machine=DefaultSessionStateMachine(uow_factory=factory),
    )
    # Override runner._uow with the root instance so tests can observe all
    # status_updates regardless of which factory call produced the write.
    runner._uow = root_uow
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopTool()
    runner._skill_tool = _NoopTool()
    return runner


@pytest.mark.parametrize(
    ("status", "terminal_reason", "runner_exception"),
    [
        (SessionStatus.COMPLETED, "natural", False),       # done
        (SessionStatus.COMPLETED, "natural", True),        # runner error
        (SessionStatus.COMPLETED, "user_cancel", False),   # cancel
        (SessionStatus.TIMED_OUT, "watchdog_timeout", False),  # timeout
    ],
)
async def test_external_terminal_owner_suppresses_all_inner_terminal_side_effects(
    status: SessionStatus,
    terminal_reason: str,
    runner_exception: bool,
) -> None:
    runner = _build_runner(f"external-{status.value}-{terminal_reason}")
    runner._external_terminal_owner = True
    runner._runner_exception_terminal = runner_exception
    runner._memory_notification_emitter = AsyncMock()
    runner._was_background = True
    runner._maybe_stop_child_publisher = AsyncMock()
    runner._maybe_stop_mailbox_supervisor = AsyncMock()

    await runner._set_terminal_status_with_notifications(status, terminal_reason)

    assert runner._uow.session.terminal_updates == []
    runner._memory_notification_emitter.emit.assert_not_awaited()
    runner._maybe_stop_child_publisher.assert_not_awaited()
    runner._maybe_stop_mailbox_supervisor.assert_not_awaited()


async def test_external_terminal_owner_keeps_cost_drain_and_tool_cleanup() -> None:
    runner = _build_runner("external-cleanup")
    runner._external_terminal_owner = True
    cost_handler = MagicMock()
    cost_handler.flush_pending = AsyncMock(
        return_value=MagicMock(drained=True, persist_failures=0)
    )
    cost_handler.write_session_degraded_marker = AsyncMock()
    runner._cost_callback_handler = cost_handler

    await runner._set_terminal_status_with_notifications(
        SessionStatus.COMPLETED, "natural"
    )

    cost_handler.flush_pending.assert_awaited_once_with(timeout=3.0)
    assert runner._uow.session.terminal_updates == []

    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)
    runner._cleanup_tools = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)
    runner._cleanup_tools.assert_awaited_once()


async def test_external_terminal_owner_suppresses_terminal_lifecycle_event() -> None:
    runner = _build_runner("external-lifecycle")
    runner._external_terminal_owner = True
    runner._lifecycle_runtime.lifecycle_events_enabled = True
    runner._is_root_session = AsyncMock(return_value=True)
    runner._put_and_add_event = AsyncMock()

    await runner._emit_task_lifecycle(
        _DummyTask(cancel_reason="stop"), LifecycleEventKind.COMPLETED,
    )

    runner._put_and_add_event.assert_not_awaited()


@pytest.mark.parametrize(
    ("external_terminal_owner", "external_heartbeat_owner"),
    [(False, False), (False, True), (True, False), (True, True)],
)
async def test_independent_ownership_quadrants_control_db_and_notification(
    external_terminal_owner: bool,
    external_heartbeat_owner: bool,
) -> None:
    runner = _build_runner(
        f"quadrant-{int(external_terminal_owner)}-"
        f"{int(external_heartbeat_owner)}"
    )
    runner._external_terminal_owner = external_terminal_owner
    runner._external_heartbeat_owner = external_heartbeat_owner
    runner._memory_notification_emitter = AsyncMock()
    runner._was_background = True

    await runner._set_terminal_status_with_notifications(
        SessionStatus.COMPLETED, "natural"
    )

    assert bool(runner._uow.session.terminal_updates) is (
        not external_terminal_owner
    )
    if external_terminal_owner:
        runner._memory_notification_emitter.emit.assert_not_awaited()
    else:
        runner._memory_notification_emitter.emit.assert_awaited_once()


async def _cancel_flow(_message):
    raise asyncio.CancelledError
    if False:
        yield None


async def _failing_flow(_message):
    raise RuntimeError("flow failed")
    if False:
        yield None


def _prime_runner_for_loop_cancellation(runner: AgentTaskRunner, task: _DummyTask) -> None:
    task.input_stream.is_empty = AsyncMock(side_effect=[False, True])
    runner._pop_event = AsyncMock(return_value=MessageEvent(message="hello"))
    runner._run_flow = _cancel_flow
    runner._load_user_preferences_map = AsyncMock(return_value={})
    runner._load_enabled_skills = AsyncMock(return_value=[])
    runner._apply_preselected_skills = AsyncMock()
    runner._skill_bundle_sync.prepare_startup_sync = AsyncMock()
    runner._skill_bundle_sync.await_initial_sync = AsyncMock()
    runner._skill_bundle_sync.start_background_sync = MagicMock()
    runner._select_skills_from_pool = MagicMock(return_value=[])
    runner._select_skills_for_message = AsyncMock(return_value=([], None))


def _prime_runner_for_loop_failure(runner: AgentTaskRunner, task: _DummyTask) -> None:
    _prime_runner_for_loop_cancellation(runner, task)
    runner._run_flow = _failing_flow


async def test_cancel_reason_stop_emits_done_and_marks_completed() -> None:
    runner = _build_runner("session-stop")
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    # B4 Issue 1D: _set_terminal_status spawns an asyncio.create_task for the
    # terminal op. Yield to the event loop so the shielded task completes
    # before we assert on status_updates.
    await asyncio.sleep(0)

    assert runner._uow.session.status_updates == [
        ("session-stop", SessionStatus.RUNNING),
    ]
    assert runner._uow.session.terminal_updates == [
        ("session-stop", SessionStatus.COMPLETED, "user_cancel"),
    ]
    assert len(task.output_stream.events) == 1
    assert '"type":"done"' in task.output_stream.events[0]


async def test_cancel_reason_takeover_start_skips_done_event_and_completed_status() -> None:
    runner = _build_runner("session-takeover-cancel")
    task = _DummyTask(cancel_reason="takeover_start")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    await asyncio.sleep(0)

    assert runner._uow.session.status_updates == [
        ("session-takeover-cancel", SessionStatus.RUNNING),
    ]
    assert task.output_stream.events == []


async def test_cancel_reason_supervisor_suspend_skips_done_event_and_completed_status() -> None:
    runner = _build_runner("session-supervisor-suspend")
    task = _DummyTask(cancel_reason="supervisor_suspend")
    emitter = AsyncMock()
    runner._memory_notification_emitter = emitter
    runner._was_background = True
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    await asyncio.sleep(0)

    assert runner._uow.session.status_updates == [
        ("session-supervisor-suspend", SessionStatus.RUNNING),
    ]
    assert task.output_stream.events == []
    emitter.emit.assert_awaited_once_with(
        user_id="user-1",
        event_type="bg_suspended_timeout",
        payload={"session_id": "session-supervisor-suspend"},
    )


async def test_background_invoke_exception_emits_bg_completed() -> None:
    runner = _build_runner("session-exception-completed")
    task = _DummyTask(cancel_reason="stop")
    emitter = AsyncMock()
    runner._memory_notification_emitter = emitter
    runner._was_background = True
    _prime_runner_for_loop_failure(runner, task)

    await runner.invoke(task)
    await asyncio.sleep(0)

    assert runner._uow.session.terminal_updates == [
        ("session-exception-completed", SessionStatus.COMPLETED, "natural"),
    ]
    emitter.emit.assert_awaited_once_with(
        user_id="user-1",
        event_type="bg_completed",
        payload={"session_id": "session-exception-completed"},
    )


async def test_cancel_reason_session_delete_skips_done_and_completed_status() -> None:
    runner = _build_runner("session-delete")
    task = _DummyTask(cancel_reason="session_delete")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    await asyncio.sleep(0)

    assert runner._uow.session.status_updates == [
        ("session-delete", SessionStatus.RUNNING),
    ]
    assert task.output_stream.events == []


# ---------------------------------------------------------------------------
# C2b follow-up (§6 limitation 2) — terminalize the child session row on a
# typed scope-violation / event-cancel re-raise, WITHOUT emitting a DoneEvent
# (emitting one would let the adapter return success and break the C2b
# typed-propagation chain). Both paths write COMPLETED + "natural".
# ---------------------------------------------------------------------------

async def _scope_violation_flow(_message):
    from app.domain.services.permission.child_scope_gate import ScopeDecision
    from app.domain.services.permission.child_scope_violation import (
        ChildScopeViolation,
    )
    raise ChildScopeViolation(
        ScopeDecision.OUT_OF_PATH_LEASE, tool_name="file_write", target_path="/x"
    )
    if False:
        yield None


async def _event_cancel_flow(_message):
    from app.domain.services.graphs.react_graph import CancelledByEventError
    raise CancelledByEventError("tool_node_entry")
    if False:
        yield None


async def test_child_scope_violation_terminalizes_completed_natural_without_done_event() -> None:
    from app.domain.services.permission.child_scope_violation import (
        ChildScopeViolation,
    )

    runner = _build_runner("session-scope-violation")
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)
    runner._run_flow = _scope_violation_flow
    task.set_child_scope_violation = MagicMock()  # adapter-stash slot

    with pytest.raises(ChildScopeViolation):
        await runner.invoke(task)

    await asyncio.sleep(0)

    # Child row terminalized so a denied child does not leak as zombie RUNNING.
    assert runner._uow.session.terminal_updates == [
        ("session-scope-violation", SessionStatus.COMPLETED, "natural"),
    ]
    # CRITICAL (C2b): NO DoneEvent — else the adapter returns a success
    # ChildRunResult and the coordinator publishes SUCCESS instead of
    # NEEDS_AUTHORIZATION, silently breaking the typed-propagation chain.
    assert task.output_stream.events == []
    # The typed violation was stashed for the adapter to re-raise.
    task.set_child_scope_violation.assert_called_once()


async def test_child_scope_violation_terminal_write_failure_still_reraises() -> None:
    """Best-effort + stash-FIRST: a failing terminal write must NOT mask the
    typed violation. The stash MUST run before the (failing) terminal write —
    else a write that throws would lose the stash the adapter re-raises. We pin
    the ORDER (not just that both ran), because the terminal write is wrapped in
    try/except so a swapped order is otherwise behaviorally indistinguishable."""
    from app.domain.services.permission.child_scope_violation import (
        ChildScopeViolation,
    )

    runner = _build_runner("session-scope-violation-degraded")
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)
    runner._run_flow = _scope_violation_flow

    order: list[str] = []
    task.set_child_scope_violation = MagicMock(
        side_effect=lambda _exc: order.append("stash")
    )

    async def _failing_terminal(*_a, **_k):
        order.append("terminal")
        raise RuntimeError("db down")

    runner._set_terminal_status_with_notifications = AsyncMock(
        side_effect=_failing_terminal
    )

    with pytest.raises(ChildScopeViolation):
        await runner.invoke(task)

    # stash-first invariant: a swapped order would record ["terminal", "stash"].
    assert order == ["stash", "terminal"]


async def test_event_cancel_terminalizes_completed_natural_without_done_event() -> None:
    from app.domain.services.graphs.react_graph import CancelledByEventError

    runner = _build_runner("session-event-cancel")
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)
    runner._run_flow = _event_cancel_flow

    with pytest.raises(CancelledByEventError):
        await runner.invoke(task)

    await asyncio.sleep(0)

    assert runner._uow.session.terminal_updates == [
        ("session-event-cancel", SessionStatus.COMPLETED, "natural"),
    ]
    # The runner's event-cancel arm must stay DoneEvent-free (mirror the
    # ChildScopeViolation arm — typed cancel propagation depends on it).
    assert task.output_stream.events == []


@pytest.mark.parametrize(
    ("flow", "expected_exception", "needs_stash"),
    [
        (_scope_violation_flow, "ChildScopeViolation", True),
        (_event_cancel_flow, "CancelledByEventError", False),
    ],
)
async def test_external_owner_real_typed_invoke_skips_inner_terminal_but_cleans_up(
    flow,
    expected_exception: str,
    needs_stash: bool,
) -> None:
    """Exercise the real invoke exception arms, not only the central helper."""
    from app.domain.services.graphs.react_graph import CancelledByEventError
    from app.domain.services.permission.child_scope_violation import (
        ChildScopeViolation,
    )

    exception_type = {
        "ChildScopeViolation": ChildScopeViolation,
        "CancelledByEventError": CancelledByEventError,
    }[expected_exception]
    runner = _build_runner(f"external-typed-{expected_exception}")
    runner._external_terminal_owner = True
    runner._external_heartbeat_owner = True
    runner._memory_notification_emitter = AsyncMock()
    runner._was_background = True
    runner._cleanup_tools = AsyncMock()
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)
    runner._run_flow = flow
    if needs_stash:
        task.set_child_scope_violation = MagicMock()

    with pytest.raises(exception_type):
        await runner.invoke(task)

    await asyncio.sleep(0)
    assert runner._uow.session.terminal_updates == []
    runner._memory_notification_emitter.emit.assert_not_awaited()
    runner._cleanup_tools.assert_awaited_once()
    assert task.output_stream.events == []
