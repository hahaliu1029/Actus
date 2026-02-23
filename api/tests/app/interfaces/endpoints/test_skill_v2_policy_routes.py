from __future__ import annotations

import httpx
import pytest

from app.domain.models.app_config import SkillRiskMode, SkillRiskPolicy
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import get_app_config_service
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeAppConfigService:
    def __init__(self) -> None:
        self._policy = SkillRiskPolicy(mode=SkillRiskMode.OFF)

    async def get_skill_risk_policy(self) -> SkillRiskPolicy:
        return self._policy

    async def update_skill_risk_policy(
        self, policy: SkillRiskPolicy
    ) -> SkillRiskPolicy:
        self._policy = policy
        return self._policy


def _fake_user(role: UserRole) -> User:
    return User(
        id="test-user",
        username="tester",
        role=role,
        status=UserStatus.ACTIVE,
    )


async def _request(
    method: str,
    url: str,
    *,
    role: UserRole,
    fake_service: _FakeAppConfigService,
    json: dict | None = None,
) -> httpx.Response:
    app.dependency_overrides[get_current_user] = lambda: _fake_user(role)
    app.dependency_overrides[get_app_config_service] = lambda: fake_service
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.request(method, url, json=json)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_app_config_service, None)


async def test_skill_policy_get_is_available_for_logged_in_user() -> None:
    fake_service = _FakeAppConfigService()
    response = await _request(
        "GET",
        "/api/v2/skills/policy",
        role=UserRole.USER,
        fake_service=fake_service,
    )
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["mode"] == "off"


async def test_skill_policy_post_is_forbidden_for_non_admin() -> None:
    fake_service = _FakeAppConfigService()
    response = await _request(
        "POST",
        "/api/v2/skills/policy",
        role=UserRole.USER,
        fake_service=fake_service,
        json={"mode": "enforce_confirmation"},
    )

    assert response.status_code == 403


async def test_skill_policy_admin_can_update_and_read_back() -> None:
    fake_service = _FakeAppConfigService()
    update_response = await _request(
        "POST",
        "/api/v2/skills/policy",
        role=UserRole.SUPER_ADMIN,
        fake_service=fake_service,
        json={"mode": "enforce_confirmation"},
    )
    assert update_response.status_code == 200
    assert update_response.json()["data"]["mode"] == "enforce_confirmation"

    get_response = await _request(
        "GET",
        "/api/v2/skills/policy",
        role=UserRole.SUPER_ADMIN,
        fake_service=fake_service,
    )
    assert get_response.status_code == 200
    assert get_response.json()["data"]["mode"] == "enforce_confirmation"
