from __future__ import annotations

from typing import Any

import httpx
import pytest
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read, rate_limit_write
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
        self.start_request_status = "started"

    async def get_takeover(self, **kwargs):
        self.calls.append(("get_takeover", kwargs))
        return {
            "status": "takeover_pending",
            "takeover_id": "tk_001",
            "request_status": "requested",
            "reason": "agent_request",
            "scope": "shell",
            "handoff_mode": None,
            "expires_at": 1_772_222_222,
        }

    async def start_takeover(self, **kwargs):
        self.calls.append(("start_takeover", kwargs))
        if self.start_request_status == "starting":
            return {
                "status": "running",
                "request_status": "starting",
                "scope": kwargs["scope"],
                "takeover_id": "tk_starting",
                "reason": None,
                "expires_at": 1_772_222_222,
            }
        return {
            "status": "takeover",
            "request_status": "started",
            "scope": kwargs["scope"],
            "takeover_id": "tk_002",
            "reason": None,
            "expires_at": 1_772_222_222,
        }

    async def renew_takeover(self, **kwargs):
        self.calls.append(("renew_takeover", kwargs))
        return {
            "status": "takeover",
            "request_status": "renewed",
            "takeover_id": kwargs["takeover_id"],
            "expires_at": 1_772_333_333,
        }

    async def reject_takeover(self, **kwargs):
        self.calls.append(("reject_takeover", kwargs))
        return {
            "status": "running",
            "reason": kwargs["decision"],
        }

    async def end_takeover(self, **kwargs):
        self.calls.append(("end_takeover", kwargs))
        return {
            "status": "completed",
            "handoff_mode": kwargs["handoff_mode"],
        }

    async def reopen_takeover(self, **kwargs):
        self.calls.append(("reopen_takeover", kwargs))
        return {
            "status": "takeover_pending",
            "request_status": "reopened",
            "reason": None,
            "remaining_seconds": 240.5,
        }


async def _request(
    method: str,
    url: str,
    *,
    fake_agent_service: _FakeAgentService,
    payload: dict | None = None,
) -> httpx.Response:
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, url, json=payload)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_read, None)
        app.dependency_overrides.pop(rate_limit_write, None)


async def test_get_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "GET",
        "/api/sessions/s1/takeover",
        fake_agent_service=fake_service,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"]["status"] == "takeover_pending"
    assert body["data"]["takeover_id"] == "tk_001"
    assert body["data"]["expires_at"] == 1_772_222_222
    assert fake_service.calls == [
        (
            "get_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_start_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/start",
        fake_agent_service=fake_service,
        payload={"scope": "browser"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"]["status"] == "takeover"
    assert body["data"]["request_status"] == "started"
    assert body["data"]["scope"] == "browser"
    assert body["data"]["expires_at"] == 1_772_222_222
    assert fake_service.calls == [
        (
            "start_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "scope": "browser",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_start_takeover_route_returns_202_when_starting() -> None:
    fake_service = _FakeAgentService()
    fake_service.start_request_status = "starting"
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/start",
        fake_agent_service=fake_service,
        payload={"scope": "shell"},
    )
    body = response.json()

    assert response.status_code == 202
    assert body["data"]["status"] == "running"
    assert body["data"]["request_status"] == "starting"
    assert body["data"]["takeover_id"] == "tk_starting"
    assert body["data"]["expires_at"] == 1_772_222_222


async def test_renew_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/renew",
        fake_agent_service=fake_service,
        payload={"takeover_id": "tk_renew_001"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"]["status"] == "takeover"
    assert body["data"]["request_status"] == "renewed"
    assert body["data"]["takeover_id"] == "tk_renew_001"
    assert body["data"]["expires_at"] == 1_772_333_333
    assert fake_service.calls == [
        (
            "renew_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "takeover_id": "tk_renew_001",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_reject_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/reject",
        fake_agent_service=fake_service,
        payload={"decision": "continue"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["status"] == "running"
    assert body["data"]["reason"] == "continue"
    assert fake_service.calls == [
        (
            "reject_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "decision": "continue",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_end_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/end",
        fake_agent_service=fake_service,
        payload={"handoff_mode": "complete"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["status"] == "completed"
    assert body["data"]["handoff_mode"] == "complete"
    assert fake_service.calls == [
        (
            "end_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "handoff_mode": "complete",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_end_takeover_route_uses_continue_as_default() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/end",
        fake_agent_service=fake_service,
        payload={},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["status"] == "completed"
    assert fake_service.calls == [
        (
            "end_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "handoff_mode": "continue",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]


async def test_reopen_takeover_route() -> None:
    fake_service = _FakeAgentService()
    response = await _request(
        "POST",
        "/api/sessions/s1/takeover/reopen",
        fake_agent_service=fake_service,
        payload={},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"]["status"] == "takeover_pending"
    assert body["data"]["request_status"] == "reopened"
    assert body["data"]["reason"] is None
    assert body["data"]["remaining_seconds"] == 240.5
    assert fake_service.calls == [
        (
            "reopen_takeover",
            {
                "session_id": "s1",
                "user_id": "test-user",
                "is_admin": False,
                "user_role": "user",
            },
        )
    ]
