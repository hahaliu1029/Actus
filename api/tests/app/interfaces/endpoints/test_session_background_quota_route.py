from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read
from app.interfaces.service_dependencies import get_supervisor
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


class _Supervisor:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def get_background_quota(self, user_id: str) -> dict[str, Any]:
        self.calls.append(user_id)
        return {
            "system_used": 3,
            "system_limit": 100,
            "user_used": 2,
            "user_limit": 5,
        }


async def test_background_quota_route() -> None:
    supervisor = _Supervisor()
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_supervisor] = lambda: supervisor
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            response = await client.get("/api/sessions/background-quota")
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_supervisor, None)
        app.dependency_overrides.pop(rate_limit_read, None)

    body = response.json()
    assert response.status_code == 200
    assert body["data"] == {
        "system_used": 3,
        "system_limit": 100,
        "user_used": 2,
        "user_limit": 5,
    }
    assert supervisor.calls == ["test-user"]
