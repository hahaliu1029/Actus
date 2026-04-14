from __future__ import annotations

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import MessageEvent
from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, object]] = []
        self.add_event_calls: list[tuple[str, object]] = []

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _NoopUoW:
    def __init__(self) -> None:
        self.session = _NoopSessionRepository()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


def _uow_factory() -> _NoopUoW:
    return _NoopUoW()


class _NoopSandbox:
    async def ensure_sandbox(self) -> None:
        return None


class _NoopTool:
    manager = None

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
    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(mcpServers={}),
        a2a_config=A2AConfig(a2a_servers=[]),
        session_id=session_id,
        user_id="user-1",
        file_storage=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_NoopSandbox(),
    )
    runner._mcp_tool = _NoopTool()
    runner._a2a_tool = _NoopTool()
    runner._skill_tool = _NoopTool()
    return runner


async def _cancel_flow(_message):
    raise asyncio.CancelledError
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


async def test_cancel_reason_stop_emits_done_and_marks_completed() -> None:
    runner = _build_runner("session-stop")
    task = _DummyTask(cancel_reason="stop")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    assert runner._uow.session.status_updates == [
        ("session-stop", SessionStatus.RUNNING),
        ("session-stop", SessionStatus.COMPLETED),
    ]
    assert len(task.output_stream.events) == 1
    assert '"type":"done"' in task.output_stream.events[0]


async def test_cancel_reason_takeover_start_skips_done_event_and_completed_status() -> None:
    runner = _build_runner("session-takeover-cancel")
    task = _DummyTask(cancel_reason="takeover_start")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    assert runner._uow.session.status_updates == [
        ("session-takeover-cancel", SessionStatus.RUNNING),
    ]
    assert task.output_stream.events == []


async def test_cancel_reason_session_delete_skips_done_and_completed_status() -> None:
    runner = _build_runner("session-delete")
    task = _DummyTask(cancel_reason="session_delete")
    _prime_runner_for_loop_cancellation(runner, task)

    with pytest.raises(asyncio.CancelledError):
        await runner.invoke(task)

    assert runner._uow.session.status_updates == [
        ("session-delete", SessionStatus.RUNNING),
    ]
    assert task.output_stream.events == []
