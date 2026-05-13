from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.application.errors.exceptions import BadRequestError, ConflictError
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_write
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
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    async def retry_from_suspend(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {
            "status": "running",
            "request_status": "resumed",
            "retry_budget_remaining": 1,
            "expires_at": 1_776_666_666,
        }


async def test_retry_from_suspend_route() -> None:
    fake_service = _FakeAgentService()
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: fake_service
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.post("/api/sessions/s1/retry-from-suspend")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_write, None)

    body = response.json()
    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"] == {
        "status": "running",
        "request_status": "resumed",
        "retry_budget_remaining": 1,
        "expires_at": 1_776_666_666,
    }
    assert fake_service.calls == [
        {
            "session_id": "s1",
            "user_id": "test-user",
            "is_admin": False,
            "user_role": "user",
        }
    ]


async def test_retry_from_suspend_route_preserves_400_and_409_errors() -> None:
    for error, expected_status in (
        (BadRequestError("当前会话不是后台任务"), 400),
        (ConflictError("后台任务状态已变化，请刷新后重试"), 409),
    ):
        fake_service = _FakeAgentService()
        fake_service.error = error
        app.dependency_overrides[get_current_user] = _fake_user
        app.dependency_overrides[get_agent_service] = lambda: fake_service
        app.dependency_overrides[rate_limit_write] = _noop_rate_limit
        try:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
            ) as client:
                response = await client.post("/api/sessions/s1/retry-from-suspend")
        finally:
            app.dependency_overrides.pop(get_current_user, None)
            app.dependency_overrides.pop(get_agent_service, None)
            app.dependency_overrides.pop(rate_limit_write, None)

        assert response.status_code == expected_status
