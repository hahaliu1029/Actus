from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncGenerator

import anyio
import pytest
from starlette.requests import Request

from app.domain.errors.supervisor import SupervisorContractError
from app.domain.models.event import (
    BaseEvent,
    ExecutionStateChangedEvent,
    ExecutionStatePayload,
    PendingExecutionEvent,
)
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

    async def _emit_event(self, session_id: str, event: BaseEvent) -> str:
        self.emitted.append((session_id, event))
        return event.id


class _AutoDegradeSupervisor:
    def __init__(
        self,
        agent: _AutoDegradeAgentService | None = None,
        exc: BaseException | None = None,
        *,
        retry_budget_remaining: int | None = 3,
        can_auto_degrade: list[bool] | None = None,
    ) -> None:
        self.agent = agent
        self.exc = exc
        self.retry_budget_remaining = retry_budget_remaining
        self.promote_calls: list[dict[str, Any]] = []
        self.resume_calls: list[dict[str, Any]] = []
        self.can_auto_degrade = list(can_auto_degrade or [True, True])

    async def can_auto_degrade_after_disconnect(self, **kwargs: Any) -> bool:
        return self.can_auto_degrade.pop(0) if self.can_auto_degrade else True

    def new_auto_degrade_cleanup_expiry(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=2)

    async def promote(self, **kwargs: Any) -> int | None:
        self.promote_calls.append(dict(kwargs))
        if self.exc is not None:
            raise self.exc
        if self.retry_budget_remaining is not None and self.agent is not None:
            current = self.agent._session
            assert current is not None
            revision = current.execution_revision + 1
            payload = ExecutionStatePayload(
                execution_mode="background",
                execution_phase="running",
                transition_reason="auto_degrade_sse_disconnect",
                background_reason="auto_degrade",
                expires_at=kwargs["expires_at"],
                retry_budget_remaining=self.retry_budget_remaining,
                execution_revision=revision,
            )
            self.agent._session = current.model_copy(update={
                "execution_mode": "background",
                "execution_phase": "running",
                "background_reason": "auto_degrade",
                "expires_at": kwargs["expires_at"],
                "execution_revision": revision,
                "pending_execution_event": PendingExecutionEvent(payload=payload),
            })
        return self.retry_budget_remaining

    async def resume(self, **kwargs: Any) -> int | None:
        self.resume_calls.append(dict(kwargs))
        if self.agent is None or self.agent._session is None:
            return None
        current = self.agent._session
        if current.execution_mode != "background" or current.background_reason != "auto_degrade":
            return None
        revision = current.execution_revision + 1
        payload = ExecutionStatePayload(
            execution_mode="foreground",
            execution_phase="running",
            transition_reason="auto_degrade_sse_reconnect",
            retry_budget_remaining=current.retry_budget_remaining,
            execution_revision=revision,
        )
        self.agent._session = current.model_copy(update={
            "execution_mode": "foreground",
            "background_reason": None,
            "expires_at": None,
            "execution_revision": revision,
            "pending_execution_event": PendingExecutionEvent(payload=payload),
        })
        return revision

    async def clear_pending_execution_event(
        self, *, session_id: str, execution_revision: int
    ) -> bool:
        if self.agent is None or self.agent._session is None:
            return False
        current = self.agent._session
        if current.execution_revision != execution_revision:
            return False
        self.agent._session = current.model_copy(update={"pending_execution_event": None})
        return True

    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        yield


async def test_do_auto_degrade_promotes_foreground_session_and_emits_state() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(agent)
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


async def test_do_auto_degrade_uses_session_owner_for_supervisor_quota() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="session-owner", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._do_auto_degrade(
        "s1",
        "admin-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    assert supervisor.promote_calls[0]["user_id"] == "session-owner"


async def test_do_auto_degrade_skips_background_session() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="background")
    )
    supervisor = _AutoDegradeSupervisor(agent)

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
    supervisor = _AutoDegradeSupervisor(agent)

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
        agent,
        exc=SupervisorContractError("bg_quota", "foreground", "background", "quota full")
    )

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    assert agent.emitted == []


async def test_do_auto_degrade_stale_promote_does_not_emit() -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(agent, retry_budget_remaining=None)

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    assert agent.emitted == []


async def test_do_auto_degrade_noops_when_reconnect_is_visible_before_promote(
) -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(agent, can_auto_degrade=[False])

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert supervisor.promote_calls == []
    assert supervisor.resume_calls == []
    assert agent.emitted == []


async def test_do_auto_degrade_compensates_when_reconnect_wins_after_precheck(
) -> None:
    agent = _AutoDegradeAgentService(
        Session(id="s1", user_id="test-user", execution_mode="foreground")
    )
    supervisor = _AutoDegradeSupervisor(agent, can_auto_degrade=[True, False])

    await session_routes._do_auto_degrade(
        "s1",
        "test-user",
        agent,
        supervisor,
    )

    assert len(supervisor.promote_calls) == 1
    assert supervisor.resume_calls == [
        {
            "session_id": "s1",
            "user_id": "test-user",
            "execution_mode": "foreground",
        }
    ]
    assert len(agent.emitted) == 1
    event = agent.emitted[0][1]
    assert isinstance(event, ExecutionStateChangedEvent)
    assert event.payload.execution_mode == "foreground"
    assert event.payload.execution_revision == 2


async def test_reconnect_resumes_running_auto_degrade_to_foreground() -> None:
    session = Session(
        id="s1",
        user_id="session-owner",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        background_reason="auto_degrade",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
    )
    agent = _AutoDegradeAgentService(session)
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1",
        request_user_id="admin-user",
        agent_service=agent,
        supervisor=supervisor,
    )

    assert supervisor.resume_calls == [
        {
            "session_id": "s1",
            "user_id": "session-owner",
            "execution_mode": "foreground",
        }
    ]


@pytest.mark.parametrize(
    "session",
    [
        Session(
            id="s1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            background_reason="explicit",
        ),
        Session(
            id="s1",
            status=SessionStatus.COMPLETED,
            execution_mode="background",
            execution_phase="running",
            background_reason="auto_degrade",
        ),
        Session(
            id="s1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="suspended",
            background_reason="auto_degrade",
        ),
    ],
)
async def test_reconnect_does_not_resume_noneligible_session(session: Session) -> None:
    agent = _AutoDegradeAgentService(session)
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1",
        request_user_id="test-user",
        agent_service=agent,
        supervisor=supervisor,
    )

    assert supervisor.resume_calls == []


async def test_reconnect_running_foreground_retries_projection_cleanup() -> None:
    agent = _AutoDegradeAgentService(Session(
        id="s1",
        user_id="test-user",
        status=SessionStatus.RUNNING,
        execution_mode="foreground",
        execution_phase="running",
    ))
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1",
        request_user_id="test-user",
        agent_service=agent,
        supervisor=supervisor,
    )

    assert len(supervisor.resume_calls) == 1
    assert agent.emitted == []


class _TransitionRaceAgent(_AutoDegradeAgentService):
    def __init__(self) -> None:
        super().__init__(Session(
            id="s1",
            user_id="test-user",
            status=SessionStatus.RUNNING,
            execution_mode="foreground",
            execution_phase="running",
        ))
        self.bg_emit_entered = asyncio.Event()
        self.allow_bg_emit = asyncio.Event()

    async def _emit_event(self, session_id: str, event: BaseEvent) -> None:
        if (
            isinstance(event, ExecutionStateChangedEvent)
            and event.payload.execution_mode == "background"
        ):
            self.bg_emit_entered.set()
            await self.allow_bg_emit.wait()
        await super()._emit_event(session_id, event)


class _TransitionRaceSupervisor(_AutoDegradeSupervisor):
    def __init__(self, agent: _TransitionRaceAgent) -> None:
        super().__init__(agent)
        self.agent = agent
        self.lock = asyncio.Lock()

    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        async with self.lock:
            yield

    async def promote(self, **kwargs: Any) -> int:
        await super().promote(**kwargs)
        assert self.agent._session is not None
        self.agent._session = self.agent._session.model_copy(update={
            "execution_mode": "background",
            "execution_phase": "running",
            "background_reason": "auto_degrade",
            "expires_at": kwargs["expires_at"],
        })
        return 3

    async def resume(self, **kwargs: Any) -> None:
        await super().resume(**kwargs)
        assert self.agent._session is not None
        self.agent._session = self.agent._session.model_copy(update={
            "execution_mode": "foreground",
            "background_reason": None,
            "expires_at": None,
        })


async def test_transition_fence_orders_background_before_concurrent_foreground_event(
) -> None:
    agent = _TransitionRaceAgent()
    supervisor = _TransitionRaceSupervisor(agent)
    degrade = asyncio.create_task(session_routes._do_auto_degrade(
        "s1", "test-user", agent, supervisor,
    ))
    await agent.bg_emit_entered.wait()

    reconnect = asyncio.create_task(
        session_routes._resume_auto_degrade_after_reconnect(
            session_id="s1",
            request_user_id="test-user",
            agent_service=agent,
            supervisor=supervisor,
        )
    )
    await asyncio.sleep(0)
    reconnect_waited_for_fence = not reconnect.done()
    agent.allow_bg_emit.set()
    await asyncio.gather(degrade, reconnect)

    assert reconnect_waited_for_fence is True
    assert [
        event.payload.execution_mode
        for _session_id, event in agent.emitted
        if isinstance(event, ExecutionStateChangedEvent)
    ] == ["background", "foreground"]


class _RetryRevokeSupervisor(_AutoDegradeSupervisor):
    def __init__(self, agent: _AutoDegradeAgentService) -> None:
        super().__init__(agent)
        self.agent = agent

    async def resume(self, **kwargs: Any) -> int | None:
        revision = await super().resume(**kwargs)
        if len(self.resume_calls) == 1:
            raise RuntimeError("redis revoke failed")
        return revision


async def test_second_reconnect_retries_projection_cleanup_after_revoke_failure(
) -> None:
    agent = _AutoDegradeAgentService(Session(
        id="s1",
        user_id="test-user",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        background_reason="auto_degrade",
    ))
    supervisor = _RetryRevokeSupervisor(agent)

    with pytest.raises(RuntimeError, match="redis revoke failed"):
        await session_routes._resume_auto_degrade_after_reconnect(
            session_id="s1", request_user_id="test-user",
            agent_service=agent, supervisor=supervisor,
        )
    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1", request_user_id="test-user",
        agent_service=agent, supervisor=supervisor,
    )

    assert len(supervisor.resume_calls) == 2


class _FailFirstExecutionEmitAgent(_AutoDegradeAgentService):
    def __init__(self, session: Session) -> None:
        super().__init__(session)
        self.emit_attempts = 0

    async def _emit_event(self, session_id: str, event: BaseEvent) -> str | None:
        self.emit_attempts += 1
        if self.emit_attempts == 1:
            return None
        return await super()._emit_event(session_id, event)


async def test_second_reconnect_retries_durable_foreground_event_after_emit_none(
) -> None:
    agent = _FailFirstExecutionEmitAgent(Session(
        id="s1",
        user_id="test-user",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        background_reason="auto_degrade",
        execution_revision=1,
    ))
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1", request_user_id="test-user",
        agent_service=agent, supervisor=supervisor,
    )
    assert agent._session is not None
    assert agent._session.pending_execution_event is not None
    assert agent._session.execution_revision == 2

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1", request_user_id="test-user",
        agent_service=agent, supervisor=supervisor,
    )

    assert agent.emit_attempts == 2
    assert agent._session.pending_execution_event is None
    assert agent._session.execution_revision == 2
    assert len(agent.emitted) == 1
    event = agent.emitted[0][1]
    assert isinstance(event, ExecutionStateChangedEvent)
    assert event.payload.execution_mode == "foreground"
    assert event.payload.execution_revision == 2


async def test_explicit_background_reconnect_retries_pending_event_without_resume(
) -> None:
    payload = ExecutionStatePayload(
        execution_mode="background",
        execution_phase="running",
        transition_reason="explicit_background_admission",
        background_reason="explicit",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=2),
        retry_budget_remaining=3,
        execution_revision=1,
    )
    agent = _FailFirstExecutionEmitAgent(Session(
        id="s1",
        user_id="test-user",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="running",
        background_reason="explicit",
        execution_revision=1,
        pending_execution_event=PendingExecutionEvent(payload=payload),
    ))
    supervisor = _AutoDegradeSupervisor(agent)

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1", request_user_id="test-user",
        agent_service=agent, supervisor=supervisor,
    )
    assert agent._session is not None
    assert agent._session.pending_execution_event is not None

    await session_routes._resume_auto_degrade_after_reconnect(
        session_id="s1", request_user_id="test-user",
        agent_service=agent, supervisor=supervisor,
    )

    assert supervisor.resume_calls == []
    assert agent._session.execution_mode == "background"
    assert agent._session.pending_execution_event is None
    assert agent.emit_attempts == 2
    assert len(agent.emitted) == 1
    assert agent.emitted[0][1].payload.execution_revision == 1


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


class _BlockingCleanupSubscriberScope(_RecordingSubscriberScope):
    def __init__(self) -> None:
        super().__init__()
        self.exit_entered = asyncio.Event()
        self.allow_exit = asyncio.Event()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exit_entered.set()
        await self.allow_exit.wait()
        await super().__aexit__(exc_type, exc, tb)


class _RecordingSupervisor:
    def __init__(self) -> None:
        self.scope = _RecordingSubscriberScope()

    def subscriber_scope(self, **kwargs: Any) -> _RecordingSubscriberScope:
        return self.scope

    @asynccontextmanager
    async def mode_transition_fence(self, *, session_id: str):
        yield

    async def resume(self, **kwargs: Any) -> None:
        return None


class _BlockingCleanupSupervisor(_RecordingSupervisor):
    def __init__(self) -> None:
        super().__init__()
        self.scope = _BlockingCleanupSubscriberScope()


class _ConflictScope:
    is_conflict = True
    current_owner = "test-user:conn-a"


class _RecordingConflictSubscriberScope:
    def __init__(self) -> None:
        self.exited = False

    async def __aenter__(self) -> _ConflictScope:
        return _ConflictScope()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.exited = True


class _ConflictSupervisor:
    def __init__(self) -> None:
        self.scope = _RecordingConflictSubscriberScope()

    def subscriber_scope(self, **kwargs: Any) -> _RecordingConflictSubscriberScope:
        return self.scope


class _AllowSessionService:
    async def get_session(self, **kwargs: Any) -> object:
        return object()


class _CancellingChatAgentService:
    async def get_session(self, session_id: str) -> Session:
        return Session(id=session_id, execution_mode="foreground")

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        raise asyncio.CancelledError()
        yield  # type: ignore[unreachable]


class _EndOfStreamChatAgentService:
    async def get_session(self, session_id: str) -> Session:
        return Session(id=session_id, execution_mode="foreground")

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        raise anyio.EndOfStream()
        yield  # type: ignore[unreachable]


class _IdleChatAgentService:
    async def get_session(self, session_id: str) -> Session:
        return Session(id=session_id, execution_mode="foreground")

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        await asyncio.Event().wait()
        yield  # type: ignore[unreachable]


class _CloseBeforeIterationAgent(_AutoDegradeAgentService):
    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        await asyncio.Event().wait()
        yield  # type: ignore[unreachable]


class _CloseBeforeIterationSupervisor(_RecordingSupervisor):
    def __init__(self, agent: _CloseBeforeIterationAgent) -> None:
        super().__init__()
        self.agent = agent

    async def can_auto_degrade_after_disconnect(self, **kwargs: Any) -> bool:
        return self.scope.exited

    def new_auto_degrade_cleanup_expiry(self) -> datetime:
        return datetime.now(timezone.utc) + timedelta(hours=2)

    async def promote(self, **kwargs: Any) -> int:
        assert self.agent._session is not None
        self.agent._session = self.agent._session.model_copy(update={
            "execution_mode": "background",
            "execution_phase": "running",
            "background_reason": "auto_degrade",
            "expires_at": kwargs["expires_at"],
        })
        return 3


def _fake_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/sessions/s1/chat",
            "headers": [(b"x-connection-id", b"conn-1")],
        }
    )


async def test_close_before_first_body_iteration_cleans_scope_without_task_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    previous_debug = loop.get_debug()
    loop.set_debug(True)
    lease = _RecordingLease()
    agent = _CloseBeforeIterationAgent(Session(
        id="s1",
        user_id="test-user",
        status=SessionStatus.RUNNING,
        execution_mode="foreground",
        execution_phase="running",
    ))
    supervisor = _CloseBeforeIterationSupervisor(agent)

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        fake_acquire_connection_limit,
    )
    before = set(session_routes._PENDING_AUTO_DEGRADE_TASKS)
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

    try:
        assert response.client_close_handler_callable is not None
        await response.client_close_handler_callable({"type": "http.disconnect"})
        await asyncio.sleep(0.02)
        created = set(session_routes._PENDING_AUTO_DEGRADE_TASKS) - before
        assert supervisor.scope.exited is True
        assert lease.released is True
        assert created == set()
    finally:
        leaked = set(session_routes._PENDING_AUTO_DEGRADE_TASKS) - before
        for task in leaked:
            task.cancel()
        await asyncio.gather(*leaked, return_exceptions=True)
        loop.set_debug(previous_debug)


async def test_cancelled_close_handler_still_cleans_then_auto_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    agent = _IdleChatAgentService()
    supervisor = _BlockingCleanupSupervisor()
    degraded = asyncio.Event()

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    async def fake_auto_degrade(*args: Any, **kwargs: Any) -> None:
        assert supervisor.scope.exited is True
        assert lease.released is True
        degraded.set()

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        fake_acquire_connection_limit,
    )
    monkeypatch.setattr(session_routes, "_do_auto_degrade", fake_auto_degrade)
    before = set(session_routes._PENDING_AUTO_DEGRADE_TASKS)
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

    assert response.client_close_handler_callable is not None
    handler_task = asyncio.create_task(
        response.client_close_handler_callable({"type": "http.disconnect"})
    )
    await supervisor.scope.exit_entered.wait()
    handler_task.cancel()
    supervisor.scope.allow_exit.set()
    with pytest.raises(asyncio.CancelledError):
        await handler_task

    await asyncio.wait_for(degraded.wait(), timeout=0.5)
    await asyncio.sleep(0)
    assert set(session_routes._PENDING_AUTO_DEGRADE_TASKS) - before == set()


async def test_shutdown_drain_cancels_and_consumes_auto_degrade_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _blocked_disconnect_workflow() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(session_routes, "_AUTO_DEGRADE_SHUTDOWN_WAIT_SECONDS", 0.0)
    task = asyncio.create_task(
        _blocked_disconnect_workflow(),
        name="test-auto-degrade-shutdown-drain",
    )
    session_routes._track_auto_degrade_task(task)
    await started.wait()

    await session_routes.drain_auto_degrade_tasks()
    await asyncio.sleep(0)

    assert task.done()
    assert task.cancelled()
    assert cancelled.is_set()
    assert task not in session_routes._PENDING_AUTO_DEGRADE_TASKS


async def test_shutdown_drain_consumes_already_failed_auto_degrade_task(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def _failed_disconnect_workflow() -> None:
        raise RuntimeError("disconnect cleanup failed")

    task = asyncio.create_task(
        _failed_disconnect_workflow(),
        name="test-failed-auto-degrade-shutdown-drain",
    )
    await asyncio.sleep(0)
    assert task.done()
    session_routes._PENDING_AUTO_DEGRADE_TASKS.add(task)
    try:
        await session_routes.drain_auto_degrade_tasks()
    finally:
        session_routes._PENDING_AUTO_DEGRADE_TASKS.discard(task)

    assert "auto-degrade task failed during shutdown" in caplog.text


async def test_chat_owner_conflict_suggests_takeover(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    supervisor = _ConflictSupervisor()

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        fake_acquire_connection_limit,
    )

    response = await session_routes.chat(
        session_id="s1",
        request=ChatRequest(message="hi"),
        fastapi_request=_fake_request(),
        current_user=_fake_user(),
        agent_service=_IdleChatAgentService(),
        session_service=_AllowSessionService(),
        supervisor=supervisor,
        redis_client=object(),
    )

    frame = await response.body_iterator.__anext__()
    payload = json.loads(frame.data)

    assert frame.event == "owner_conflict"
    assert payload["payload"]["suggested_action"] == "request_takeover"
    await response.body_iterator.aclose()
    assert supervisor.scope.exited is True
    assert lease.released is True


async def test_chat_http_disconnect_callback_detaches_auto_degrade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    agent = _IdleChatAgentService()
    supervisor = _RecordingSupervisor()
    auto_degrade_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    created_tasks: list[asyncio.Task[None]] = []

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    def fake_auto_degrade(*args: Any, **kwargs: Any):
        auto_degrade_calls.append((args, dict(kwargs)))

        async def _noop() -> None:
            return None

        return _noop()

    def fake_track_auto_degrade_task(task: asyncio.Task[None]) -> None:
        created_tasks.append(task)

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
    monkeypatch.setattr(
        session_routes,
        "_track_auto_degrade_task",
        fake_track_auto_degrade_task,
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

    assert response.client_close_handler_callable is not None
    await response.client_close_handler_callable({"type": "http.disconnect"})
    await asyncio.gather(*created_tasks)

    assert len(auto_degrade_calls) == 1
    args, kwargs = auto_degrade_calls[0]
    assert args == ("s1", "test-user", agent, supervisor)
    assert kwargs == {}
    assert len(created_tasks) == 1
    assert created_tasks[0].done()
    assert supervisor.scope.exited is True
    assert lease.released is True


async def test_chat_cancelled_stream_detaches_auto_degrade_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _RecordingLease()
    agent = _CancellingChatAgentService()
    supervisor = _RecordingSupervisor()
    auto_degrade_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    created_tasks: list[asyncio.Task[None]] = []

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    def fake_auto_degrade(*args: Any, **kwargs: Any):
        auto_degrade_calls.append((args, dict(kwargs)))

        async def _noop() -> None:
            return None

        return _noop()

    def fake_track_auto_degrade_task(task: asyncio.Task[None]) -> None:
        created_tasks.append(task)

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
    monkeypatch.setattr(
        session_routes,
        "_track_auto_degrade_task",
        fake_track_auto_degrade_task,
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
    await asyncio.gather(*created_tasks)

    assert len(auto_degrade_calls) == 1
    args, kwargs = auto_degrade_calls[0]
    assert args == ("s1", "test-user", agent, supervisor)
    assert kwargs == {}
    assert len(created_tasks) == 1
    assert created_tasks[0].done()
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
    created_tasks: list[asyncio.Task[None]] = []

    async def fake_acquire_connection_limit(**kwargs: Any) -> _RecordingLease:
        return lease

    def fake_auto_degrade(*args: Any, **kwargs: Any):
        auto_degrade_calls.append((args, dict(kwargs)))

        async def _noop() -> None:
            return None

        return _noop()

    def fake_track_auto_degrade_task(task: asyncio.Task[None]) -> None:
        created_tasks.append(task)

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
    monkeypatch.setattr(
        session_routes,
        "_track_auto_degrade_task",
        fake_track_auto_degrade_task,
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
    await asyncio.gather(*created_tasks)

    assert len(auto_degrade_calls) == 1
    args, kwargs = auto_degrade_calls[0]
    assert args == ("s1", "test-user", agent, supervisor)
    assert kwargs == {}
    assert len(created_tasks) == 1
    assert created_tasks[0].done()
    assert supervisor.scope.exited is True
    assert supervisor.scope.exit_exc_type is None
    assert lease.released is True
