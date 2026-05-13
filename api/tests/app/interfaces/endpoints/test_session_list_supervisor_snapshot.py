from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import pytest
from sse_starlette import ServerSentEvent

from app.domain.models.session import Session
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.endpoints import session_routes

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _user() -> User:
    return User(
        id="user-1",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def _background_session() -> Session:
    return Session(
        id="bg-1",
        title="后台任务",
        user_id="user-1",
        latest_message="处理中",
        latest_message_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        status="running",
        execution_mode="background",
        execution_phase="running",
        background_reason="explicit",
        retry_budget_remaining=2,
    )


def _foreground_session() -> Session:
    return Session(
        id="fg-1",
        title="前台任务",
        user_id="user-1",
        execution_mode="foreground",
    )


def _snapshot(session: Session) -> dict[str, Any]:
    return {
        "execution_mode": session.execution_mode,
        "execution_phase": session.execution_phase,
        "background_reason": session.background_reason,
        "expires_at": None,
        "retry_budget_remaining": session.retry_budget_remaining,
        "suspended_reason": None,
        "terminal_reason": None,
        "last_progress_at": None,
        "is_alive": True,
        "cancellation_state": "none",
    }


class _SessionService:
    async def get_all_sessions(self, user_id: str, is_admin: bool) -> list[Session]:
        return [_background_session(), _foreground_session()]

    async def get_session(self, **kwargs: Any) -> Session:
        return _background_session()


class _AgentService:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def build_supervisor_snapshot(self, session: Session) -> dict[str, Any]:
        self.calls.append(session.id)
        return _snapshot(session)


class _Lease:
    def start_heartbeat(self) -> None:
        pass

    async def release(self) -> None:
        pass


async def test_get_all_sessions_includes_background_supervisor_snapshot() -> None:
    agent_service = _AgentService()

    response = await session_routes.get_all_sessions(
        current_user=_user(),
        session_service=_SessionService(),
        agent_service=agent_service,
    )

    items = response.data.sessions
    assert items[0].supervisor_snapshot is not None
    assert items[0].supervisor_snapshot.model_dump(mode="json") == _snapshot(
        _background_session()
    )
    assert items[1].supervisor_snapshot is None
    assert agent_service.calls == ["bg-1"]


async def test_get_session_includes_background_supervisor_snapshot() -> None:
    agent_service = _AgentService()

    response = await session_routes.get_session(
        session_id="bg-1",
        current_user=_user(),
        session_service=_SessionService(),
        agent_service=agent_service,
    )

    assert response.data.supervisor_snapshot is not None
    assert response.data.supervisor_snapshot.model_dump(mode="json") == _snapshot(
        _background_session()
    )
    assert agent_service.calls == ["bg-1"]


async def test_stream_sessions_includes_background_supervisor_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_service = _AgentService()
    captured: dict[str, AsyncGenerator[ServerSentEvent, None]] = {}

    async def _acquire_connection_limit(**kwargs: Any) -> _Lease:
        return _Lease()

    class _EventSourceResponse:
        def __init__(
            self,
            generator: AsyncGenerator[ServerSentEvent, None],
            *,
            headers: dict[str, str],
        ) -> None:
            captured["generator"] = generator
            self.headers = headers

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _acquire_connection_limit,
    )
    monkeypatch.setattr(session_routes, "EventSourceResponse", _EventSourceResponse)

    await session_routes.stream_sessions(
        current_user=_user(),
        session_service=_SessionService(),
        agent_service=agent_service,
        redis_client=object(),
    )

    event = await captured["generator"].__anext__()
    body = json.loads(event.data)

    items = body["sessions"]
    assert items[0]["supervisor_snapshot"] == _snapshot(_background_session())
    assert items[1]["supervisor_snapshot"] is None
    assert agent_service.calls == ["bg-1"]
