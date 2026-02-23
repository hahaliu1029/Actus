from __future__ import annotations

import asyncio
import json

from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.endpoints.skill_routes import list_skills
from app.interfaces.endpoints.user_routes import get_skill_tools


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


def test_legacy_skill_admin_routes_return_410() -> None:
    response = asyncio.run(list_skills(admin_user=_fake_user()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 410
    assert body["msg"] == "SKILL_API_MOVED"
    assert body["data"]["migrate_to"] == "/v2/skills"


def test_legacy_user_skill_routes_return_410() -> None:
    response = asyncio.run(get_skill_tools(current_user=_fake_user()))
    body = json.loads(response.body.decode("utf-8"))

    assert response.status_code == 410
    assert body["msg"] == "SKILL_API_MOVED"
    assert body["data"]["migrate_to"] == "/v2/user/tools/skills"
