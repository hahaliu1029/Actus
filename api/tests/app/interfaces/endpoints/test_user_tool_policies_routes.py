"""Route tests for /v2/user/tool-policies CRUD using dependency_overrides.

Pattern copied from test_skill_v2_policy_routes.py:43-62.
No real DB; FakeService is in-memory.
"""
from __future__ import annotations

from typing import Optional

import httpx
import pytest

from app.application.services.user_tool_approval_policy_service import (
    UserToolApprovalPolicyService,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.domain.models.user_tool_approval_policy import (
    UserToolApprovalPolicy,
)
from app.domain.repositories.user_tool_approval_policy_repository import (
    UserToolApprovalPolicyRepository,
)
from app.infrastructure.storage.postgres import get_db_session
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import (
    get_user_tool_approval_policy_service,
)
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _InMemoryRepo(UserToolApprovalPolicyRepository):
    def __init__(self) -> None:
        self._data: dict[tuple[str, str], UserToolApprovalPolicy] = {}

    async def get(
        self, user_id: str, tool_name: str
    ) -> Optional[UserToolApprovalPolicy]:
        return self._data.get((user_id, tool_name))

    async def list_by_user(
        self, user_id: str
    ) -> list[UserToolApprovalPolicy]:
        return [
            p
            for (uid, _), p in self._data.items()
            if uid == user_id
        ]

    async def upsert(
        self, policy: UserToolApprovalPolicy
    ) -> UserToolApprovalPolicy:
        self._data[(policy.user_id, policy.tool_name)] = policy
        return policy

    async def delete(self, user_id: str, tool_name: str) -> bool:
        return self._data.pop((user_id, tool_name), None) is not None


class _FakeDbSession:
    """No-op session — handlers call .commit(); we skip persistence since
    FakeService is purely in-memory and bypasses ORM entirely."""

    async def commit(self) -> None:
        pass


def _fake_user(user_id: str = "user-a") -> User:
    return User(
        id=user_id,
        username=f"u_{user_id}",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


async def _request(
    method: str,
    url: str,
    *,
    fake_service: UserToolApprovalPolicyService,
    user: Optional[User] = None,
    json: Optional[dict] = None,
) -> httpx.Response:
    app.dependency_overrides[get_current_user] = lambda: user or _fake_user()
    app.dependency_overrides[get_user_tool_approval_policy_service] = (
        lambda: fake_service
    )
    app.dependency_overrides[get_db_session] = lambda: _FakeDbSession()
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, url, json=json)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_user_tool_approval_policy_service, None)
        app.dependency_overrides.pop(get_db_session, None)


def _fresh_service() -> UserToolApprovalPolicyService:
    return UserToolApprovalPolicyService(_InMemoryRepo())


async def test_get_list_empty() -> None:
    svc = _fresh_service()
    r = await _request("GET", "/api/v2/user/tool-policies", fake_service=svc)
    assert r.status_code == 200
    assert r.json()["data"] == {"policies": []}


async def test_put_then_get() -> None:
    svc = _fresh_service()
    r1 = await _request(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
        json={"policy": "ask"},
    )
    assert r1.status_code == 200
    assert r1.json()["data"]["policy"] == "ask"
    r2 = await _request(
        "GET",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
    )
    assert r2.status_code == 200
    assert r2.json()["data"]["policy"] == "ask"


async def test_put_overwrites() -> None:
    svc = _fresh_service()
    await _request(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
        json={"policy": "ask"},
    )
    await _request(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
        json={"policy": "auto"},
    )
    r = await _request(
        "GET",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
    )
    assert r.json()["data"]["policy"] == "auto"


async def test_get_missing_returns_404() -> None:
    svc = _fresh_service()
    r = await _request(
        "GET",
        "/api/v2/user/tool-policies/never-set-tool",
        fake_service=svc,
    )
    assert r.status_code == 404


async def test_delete_is_idempotent() -> None:
    svc = _fresh_service()
    r1 = await _request(
        "DELETE",
        "/api/v2/user/tool-policies/never-set-tool",
        fake_service=svc,
    )
    assert r1.status_code == 200

    await _request(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
        json={"policy": "ask"},
    )
    r2 = await _request(
        "DELETE",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
    )
    assert r2.status_code == 200
    r3 = await _request(
        "GET",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
    )
    assert r3.status_code == 404


async def test_put_rejects_invalid_policy_value() -> None:
    svc = _fresh_service()
    r = await _request(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        fake_service=svc,
        json={"policy": "allow"},  # 不在 enum
    )
    assert r.status_code == 422


async def test_put_accepts_mcp_tool_name_with_hyphen() -> None:
    svc = _fresh_service()
    r = await _request(
        "PUT",
        "/api/v2/user/tool-policies/mcp_amap-maps_maps_weather",
        fake_service=svc,
        json={"policy": "auto"},
    )
    assert r.status_code == 200


async def test_list_returns_multiple_policies() -> None:
    svc = _fresh_service()
    for tool, pol in [
        ("shell_execute", "ask"),
        ("mcp_github_create_issue", "auto"),
        ("brainstorm_skill", "deny"),
    ]:
        await _request(
            "PUT",
            f"/api/v2/user/tool-policies/{tool}",
            fake_service=svc,
            json={"policy": pol},
        )
    r = await _request(
        "GET", "/api/v2/user/tool-policies", fake_service=svc
    )
    assert r.status_code == 200
    names = {p["tool_name"] for p in r.json()["data"]["policies"]}
    assert names == {
        "shell_execute",
        "mcp_github_create_issue",
        "brainstorm_skill",
    }
