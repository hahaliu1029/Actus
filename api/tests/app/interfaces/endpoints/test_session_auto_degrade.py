from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator

import anyio
import pytest
from starlette.requests import Request

from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.event import BaseEvent, ExecutionStateChangedEvent
from app.domain.models.session import Session, SessionStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.endpoints import session_routes
from app.interfaces.schemas.session import ChatRequest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


class _AutoDegradeAgentService:
    def __init__(self, session: Session | None) -> None:
        self._session = session
        self.emitted: list[tuple[str, BaseEvent]] = []

    async def get_session(self, session_id: str) -> Session | None:
        return self._session

    async def _emit_event(self, session_id: str, event: BaseEvent) -> None:
        self.emitted.append((session_id, event))


class _AutoDegradeSupervisor:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.exc = exc
        self.promote_calls: list[dict[str, Any]] = []

    async def promote(self, **kwargs: Any) -> None:
        self.promote_calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc


async def test_do_auto_degrade_promotes_foreground_session_and_emits_state() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor()
    before = datetime.now(timezone.utc)

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    promote_call = supervisor.promote_calls[0]
    expires_at = promote_call["expires_at"]
    assert promote_call["session_id"] == "s1"
    assert promote_call["user_id"] == "test-user"
    assert expires_at.tzinfo == timezone.utc
    assert timedelta(hours=1, minutes=59) <= expires_at - before <= timedelta(
        hours=2, seconds=1
    )

    assert len(agent.emitted) == 1
    emitted_session_id, event = agent.emitted[0]
    assert emitted_session_id == "s1"
    assert isinstance(event, ExecutionStateChangedEvent)
    assert event.payload.execution_mode == "background"
    assert event.payload.execution_phase == "running"
    assert event.payload.transition_reason == "auto_degrade_sse_disconnect"
    assert event.payload.background_reason == "auto_degrade"
    assert event.payload.expires_at == expires_at
    assert event.payload.retry_budget_remaining == 3


async def test_do_auto_degrade_skips_background_session() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="background")
    )
    supervisor = _AutoDegradeSupervisor()

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert supervisor.promote_calls == []
    assert agent.emitted == []


@pytest.mark.parametrize(
    ("session",),
    [
        (
            Session(
                id="s1",
                user_id="test-user",
                execution_mode="foreground",
                status=SessionStatus.COMPLETED,
            ),
        ),
        (
            Session(
                id="s1",
                user_id="test-user",
                execution_mode="foreground",
                status=SessionStatus.TIMED_OUT,
            ),
        ),
        (
            Session(
                id="s1",
                user_id="test-user",
                execution_mode="foreground",
                execution_phase="terminating",
            ),
        ),
        (
            Session(
                id="s1",
                user_id="test-user",
                execution_mode="foreground",
                execution_phase="terminated",
            ),
        ),
    ],
)
async def test_do_auto_degrade_skips_terminal_or_terminating_session(
    session: Session,
) -> None:
    agent = _AutoDegradeAgentService(session)
    supervisor = _AutoDegradeSupervisor()

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert supervisor.promote_calls == []
    assert agent.emitted == []


async def test_do_auto_degrade_promote_rejection_does_not_emit() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(
        SupervisorContractError("bg_quota", "foreground", "background", "quota full")
    )

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    assert agent.emitted == []


class _RecordingLease:
    def __init__(self) -> None:
        self.started = False
        self.released = False

    def start_heartbeat(self) -> None:
        self.started = True

    async def release(self) -> None:
        self.released = True


class _NoConflictScope:
    is_conflict = False
    current_owner = None


class _RecordingSubscriberScope:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.exit_exc_type: type[BaseException] | None = None

    async def __aenter__(self) -> _NoConflictScope:
        self.entered = True
        return _NoConflictScope()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True
        self.exit_exc_type = exc_type


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.scope = _RecordingSubscriberScope()

    def subscriber_scope(self, **kwargs: Any) -> _RecordingSubscriberScope:
        return self.scope


class _AllowSessionService:
    async def get_session(self, **kwargs: Any) -> object:
        return object()


class _CancellingChatAgentService:
    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        raise asyncio.CancelledError()
        yield  # type: ignore[unreachable]


class _EndOfStreamChatAgentService:
    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        raise anyio.EndOfStream()
        yield  # type: ignore[unreachable]


def _fake_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/sessions/s1/chat",
            "headers": [(b"x-connection-id", b"conn-1")],
        }
    )


async def test_chat_cancelled_stream_detaches_auto_degrade_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    agent = _CancellingChatAgentService()
    supervisor = _RecordingSupervisor()
    auto_degrade_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    created_tasks: list[Any] = []
    pending_tasks: set[Any] = set()

    class _FakeTask:
        def __init__(self, coro: Any) -> None:
            self.coro = coro
            self.done_callbacks: list[Any] = []

        def add_done_callback(self, callback: Any) -> None:
            self.done_callbacks.append(callback)

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    def fake_auto_degrade(*args: Any, **kwargs: Any):
        auto_degrade_calls.append((args, dict(kwargs)))

        async def _noop() -> None:
            return None

        return _noop()

    def fake_create_task(coro):
        task = _FakeTask(coro)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        fake_acquire_connection_limit,
    )
    monkeypatch.setattr(
        session_routes,
        "_do_auto_degrade",
        fake_auto_degrade,
        raising=False,
    )
    monkeypatch.setattr(session_routes.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        session_routes,
        "_PENDING_AUTO_DEGRADE_TASKS",
        pending_tasks,
        raising=False,
    )

    response = await session_routes.chat(
        session_id="s1",
        request=ChatRequest(message="hi"),
        fastapi_request=_fake_request(),
        current_user=_fake_user(),
        agent_service=agent,
        session_service=_AllowSessionService(),
        supervisor=supervisor,
        redis_client=object(),
    )

    with pytest.raises(asyncio.CancelledError):
        await response.body_iterator.__anext__()

    assert auto_degrade_calls == [(("s1", "test-user", agent, supervisor), {})]
    assert len(created_tasks) == 1
    task = created_tasks[0]
    assert task in pending_tasks
    assert len(task.done_callbacks) == 1
    task.done_callbacks[0](task)
    assert pending_tasks == set()
    assert supervisor.scope.exited is True
    assert supervisor.scope.exit_exc_type is None
    assert lease.released is True


async def test_chat_end_of_stream_detaches_auto_degrade_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    agent = _EndOfStreamChatAgentService()
    supervisor = _RecordingSupervisor()
    auto_degrade_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    created_tasks: list[Any] = []
    pending_tasks: set[Any] = set()

    class _FakeTask:
        def __init__(self, coro: Any) -> None:
            self.coro = coro
            self.done_callbacks: list[Any] = []

        def add_done_callback(self, callback: Any) -> None:
            self.done_callbacks.append(callback)

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    def fake_auto_degrade(*args: Any, **kwargs: Any):
        auto_degrade_calls.append((args, dict(kwargs)))

        async def _noop() -> None:
            return None

        return _noop()

    def fake_create_task(coro):
        task = _FakeTask(coro)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        fake_acquire_connection_limit,
    )
    monkeypatch.setattr(
        session_routes,
        "_do_auto_degrade",
        fake_auto_degrade,
        raising=False,
    )
    monkeypatch.setattr(session_routes.asyncio, "create_task", fake_create_task)
    monkeypatch.setattr(
        session_routes,
        "_PENDING_AUTO_DEGRADE_TASKS",
        pending_tasks,
        raising=False,
    )

    response = await session_routes.chat(
        session_id="s1",
        request=ChatRequest(message="hi"),
        fastapi_request=_fake_request(),
        current_user=_fake_user(),
        agent_service=agent,
        session_service=_AllowSessionService(),
        supervisor=supervisor,
        redis_client=object(),
    )

    with pytest.raises(anyio.EndOfStream):
        await response.body_iterator.__anext__()

    assert auto_degrade_calls == [(("s1", "test-user", agent, supervisor), {})]
    assert len(created_tasks) == 1
    task = created_tasks[0]
    assert task in pending_tasks
    assert len(task.done_callbacks) == 1
    task.done_callbacks[0](task)
    assert pending_tasks == set()
    assert supervisor.scope.exited is True
    assert supervisor.scope.exit_exc_type is None
    assert lease.released is True
