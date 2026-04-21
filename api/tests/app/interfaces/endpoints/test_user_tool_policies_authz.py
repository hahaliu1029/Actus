"""Authz tests: user A cannot see user B's policies (FakeService isolates by user_id).

Same dependency_overrides + ASGITransport pattern as test_user_tool_policies_routes.py.
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
    async def commit(self) -> None:
        pass


def _user(uid: str) -> User:
    return User(
        id=uid,
        username=f"u_{uid}",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


async def _req(
    method: str,
    url: str,
    *,
    svc: UserToolApprovalPolicyService,
    as_user: User,
    json: Optional[dict] = None,
) -> httpx.Response:
    app.dependency_overrides[get_current_user] = lambda: as_user
    app.dependency_overrides[get_user_tool_approval_policy_service] = (
        lambda: svc
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


async def test_user_list_only_shows_own_policies() -> None:
    svc = UserToolApprovalPolicyService(_InMemoryRepo())
    user_a, user_b = _user("user-a"), _user("user-b")

    await _req(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        svc=svc,
        as_user=user_a,
        json={"policy": "ask"},
    )

    r = await _req(
        "GET",
        "/api/v2/user/tool-policies",
        svc=svc,
        as_user=user_b,
    )
    assert r.status_code == 200
    names = {p["tool_name"] for p in r.json()["data"]["policies"]}
    assert "shell_execute" not in names


async def test_user_get_other_user_policy_returns_404() -> None:
    svc = UserToolApprovalPolicyService(_InMemoryRepo())
    user_a, user_b = _user("user-a"), _user("user-b")

    await _req(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        svc=svc,
        as_user=user_a,
        json={"policy": "ask"},
    )
    r = await _req(
        "GET",
        "/api/v2/user/tool-policies/shell_execute",
        svc=svc,
        as_user=user_b,
    )
    assert r.status_code == 404


async def test_put_ignores_user_id_in_body() -> None:
    """Body 中的任何 user_id 字段都被忽略，落表永远用 CurrentUser.id。"""
    svc = UserToolApprovalPolicyService(_InMemoryRepo())
    user_a, user_b = _user("user-a"), _user("user-b")

    r = await _req(
        "PUT",
        "/api/v2/user/tool-policies/shell_execute",
        svc=svc,
        as_user=user_a,
        json={"policy": "ask", "user_id": user_b.id},  # malicious body field
    )
    assert r.status_code == 200

    # user B 视角看不到这条
    r_b = await _req(
        "GET",
        "/api/v2/user/tool-policies/shell_execute",
        svc=svc,
        as_user=user_b,
    )
    assert r_b.status_code == 404
