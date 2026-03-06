from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.application.errors.exceptions import NotFoundError
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_admin() -> User:
    return User(
        id="test-admin",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


def _make_skill() -> Skill:
    return Skill(
        id="skill-001",
        slug="demo-skill",
        name="Demo Skill",
        description="A demo skill for testing",
        version="1.0.0",
        source_type=SkillSourceType.LOCAL,
        source_ref="/tmp/demo-skill",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "tools": [
                {
                    "name": "run_code",
                    "description": "Run code in sandbox",
                    "parameters": {"type": "object"},
                    "required": ["code"],
                }
            ],
            "skill_md": "# Demo Skill\nThis is a demo.",
            "activation": {"auto": True},
            "policy": {"sandbox": True},
            "security": {"allow_network": False},
            "bundle_file_count": 3,
            "context_ref_count": 1,
            "last_sync_at": "2026-01-01T00:00:00",
        },
        enabled=True,
        installed_by="user-1",
        created_at=datetime(2026, 1, 1, 0, 0, 0),
        updated_at=datetime(2026, 1, 2, 0, 0, 0),
    )


async def test_get_skill_detail_returns_full_data() -> None:
    skill = _make_skill()
    fake_service = AsyncMock()
    fake_service.get_skill = AsyncMock(return_value=skill)

    app.dependency_overrides[get_current_user] = _fake_admin

    with patch(
        "app.interfaces.endpoints.skill_v2_routes._build_skill_service",
        return_value=fake_service,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/v2/skills/demo-skill--abcd1234")

    app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 200
    body = response.json()
    data = body["data"]

    assert data["id"] == "skill-001"
    assert data["slug"] == "demo-skill"
    assert data["name"] == "Demo Skill"
    assert data["description"] == "A demo skill for testing"
    assert data["version"] == "1.0.0"
    assert data["source_type"] == "local"
    assert data["source_ref"] == "/tmp/demo-skill"
    assert data["runtime_type"] == "native"
    assert data["enabled"] is True
    assert data["installed_by"] == "user-1"
    assert data["created_at"] == "2026-01-01T00:00:00"
    assert data["updated_at"] == "2026-01-02T00:00:00"
    assert data["bundle_file_count"] == 3
    assert data["context_ref_count"] == 1
    assert data["last_sync_at"] == "2026-01-01T00:00:00"

    # tools
    assert len(data["tools"]) == 1
    tool = data["tools"][0]
    assert tool["name"] == "run_code"
    assert tool["description"] == "Run code in sandbox"
    assert tool["parameters"] == {"type": "object"}
    assert tool["required"] == ["code"]

    # skill_md
    assert data["skill_md"] == "# Demo Skill\nThis is a demo."

    # activation / policy / security
    assert data["activation"] == {"auto": True}
    assert data["policy"] == {"sandbox": True}
    assert data["security"] == {"allow_network": False}

    # bundle_files defaults to empty (no bundle_index.json on disk)
    assert data["bundle_files"] == []


async def test_get_skill_detail_not_found() -> None:
    fake_service = AsyncMock()
    fake_service.get_skill = AsyncMock(
        side_effect=NotFoundError("Skill 不存在")
    )

    app.dependency_overrides[get_current_user] = _fake_admin

    with patch(
        "app.interfaces.endpoints.skill_v2_routes._build_skill_service",
        return_value=fake_service,
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.get("/api/v2/skills/nonexistent-skill")

    app.dependency_overrides.pop(get_current_user, None)

    assert response.status_code == 404
