from __future__ import annotations

import httpx
import pytest

from app.domain.models.skill_creator import SkillCreationProgress, SkillCreationResult
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import get_skill_creator_service
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_admin_user() -> User:
    return User(
        id="admin-user",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


class _FakeCreatorService:
    async def create(self, *, description: str, sandbox=None, installed_by: str = ""):
        del description, sandbox, installed_by
        yield SkillCreationProgress(step="analyzing", message="正在分析...")
        yield SkillCreationResult(
            skill_id="test--abc",
            skill_name="test",
            tools=["run"],
            files_count=3,
            summary="ok",
        )


async def test_create_returns_sse_stream() -> None:
    app.dependency_overrides[get_current_user] = _fake_admin_user
    app.dependency_overrides[get_skill_creator_service] = lambda: _FakeCreatorService()

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v2/skills/create",
                json={"description": "创建一个测试 skill"},
            )
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_skill_creator_service, None)

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
    assert "event: progress" in response.text
    assert "event: complete" in response.text


async def test_create_requires_description() -> None:
    app.dependency_overrides[get_current_user] = _fake_admin_user
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v2/skills/create", json={})
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 422
