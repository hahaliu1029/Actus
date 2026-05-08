from __future__ import annotations

from typing import Any

import httpx
import pytest
from app.domain.models.event import MessageEvent
from app.domain.models.session import SessionStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read
from app.interfaces.service_dependencies import get_agent_service
from app.main import app

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


async def _noop_rate_limit() -> None:
    return None


class _FakeAgentService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.result: dict[str, Any] = {
            "events": [],
            "session_status": SessionStatus.RUNNING,
            "has_more": False,
            # B3-core PR-1 §3.3 — endpoint reads these new keys
            "last_seq": 0,
            "supervisor_snapshot": None,
        }

    async def get_events_since(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_events_since", kwargs))
        return self.result


async def _request(
    url: str,
    *,
    fake_agent_service: _FakeAgentService,
) -> httpx.Response:
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.get(url)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_read, None)


async def test_get_events_since_with_since_param() -> None:
    """增量事件端点带 since 参数时正确透传并返回 SSE 格式事件"""
    e = MessageEvent(role="assistant", message="hello")
    e.id = "evt-2"
    fake_service = _FakeAgentService()
    fake_service.result = {
        "events": [e],
        "session_status": SessionStatus.RUNNING,
        "has_more": False,
        # B3-core PR-1 §3.3 — endpoint reads these new keys
        "last_seq": 0,
        "supervisor_snapshot": None,
    }

    response = await _request(
        "/api/sessions/s1/events?since=evt-1",
        fake_agent_service=fake_service,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert len(body["data"]["events"]) == 1
    assert body["data"]["session_status"] == "running"
    assert body["data"]["has_more"] is False
    # 验证 service 收到正确参数
    assert fake_service.calls == [
        (
            "get_events_since",
            {
                "session_id": "s1",
                "since_event_id": "evt-1",
                "user_id": "test-user",
                "is_admin": False,
                "since_seq": None,  # B3-core PR-1 §3.3 — additive kwarg
            },
        )
    ]


async def test_get_events_since_without_since_param() -> None:
    """不带 since 参数时 service 收到 since_event_id=None（全量恢复场景）"""
    fake_service = _FakeAgentService()

    response = await _request(
        "/api/sessions/s1/events",
        fake_agent_service=fake_service,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    # 验证 since_event_id 为 None
    assert fake_service.calls == [
        (
            "get_events_since",
            {
                "session_id": "s1",
                "since_event_id": None,
                "user_id": "test-user",
                "is_admin": False,
                "since_seq": None,  # B3-core PR-1 §3.3 — additive kwarg
            },
        )
    ]


async def test_get_events_since_passes_user_auth() -> None:
    """端点正确透传当前用户身份到 service 层"""
    fake_service = _FakeAgentService()

    await _request(
        "/api/sessions/s1/events?since=evt-1",
        fake_agent_service=fake_service,
    )

    call_kwargs = fake_service.calls[0][1]
    assert call_kwargs["user_id"] == "test-user"
    assert call_kwargs["is_admin"] is False
