from __future__ import annotations

import pytest

from app.application.errors.exceptions import NotFoundError, ValidationError
from app.application.services.skill_service import SkillService
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _InMemorySkillRepository:
    def __init__(self) -> None:
        self._items: dict[str, Skill] = {}

    async def list(self) -> list[Skill]:
        return list(self._items.values())

    async def list_enabled(self) -> list[Skill]:
        return [item for item in self._items.values() if item.enabled]

    async def get_by_id(self, skill_id: str) -> Skill | None:
        return self._items.get(skill_id)

    async def get_by_slug(self, slug: str) -> Skill | None:
        for item in self._items.values():
            if item.slug == slug:
                return item
        return None

    async def upsert(self, skill: Skill) -> Skill:
        self._items[skill.id] = skill
        return skill

    async def delete(self, skill_id: str) -> bool:
        return self._items.pop(skill_id, None) is not None


async def test_install_skill_requires_tools_manifest() -> None:
    service = SkillService(_InMemorySkillRepository())

    with pytest.raises(ValidationError):
        await service.install_skill(
            source_type=SkillSourceType.GITHUB,
            source_ref="owner/repo",
            manifest={
                "name": "Demo Skill",
                "runtime_type": SkillRuntimeType.NATIVE.value,
            },
            skill_md="# demo",
            installed_by="admin-1",
        )


async def test_install_skill_creates_and_updates_by_slug() -> None:
    service = SkillService(_InMemorySkillRepository())

    created = await service.install_skill(
        source_type=SkillSourceType.GITHUB,
        source_ref="owner/repo",
        manifest={
            "name": "Demo Skill",
            "runtime_type": SkillRuntimeType.NATIVE.value,
            "tools": [
                {
                    "name": "demo_native_tool",
                    "description": "run demo",
                    "parameters": {"path": {"type": "string"}},
                    "required": ["path"],
                    "entry": {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": "echo hello",
                    },
                }
            ],
        },
        skill_md="# demo",
        installed_by="admin-1",
    )

    updated = await service.install_skill(
        source_type=SkillSourceType.GITHUB,
        source_ref="owner/repo-v2",
        manifest={
            "name": "Demo Skill",
            "runtime_type": SkillRuntimeType.NATIVE.value,
            "version": "2.0.0",
            "tools": [
                {
                    "name": "demo_native_tool",
                    "description": "run demo 2",
                    "parameters": {"path": {"type": "string"}},
                    "required": ["path"],
                    "entry": {
                        "exec_dir": "/home/ubuntu/workspace",
                        "command": "echo world",
                    },
                }
            ],
        },
        skill_md="# demo v2",
        installed_by="admin-1",
    )

    assert updated.id == created.id
    assert updated.version == "2.0.0"
    assert updated.source_ref == "owner/repo-v2"


async def test_set_skill_enabled_raises_when_missing() -> None:
    service = SkillService(_InMemorySkillRepository())

    with pytest.raises(NotFoundError):
        await service.set_skill_enabled("missing", False)


async def test_discovery_returns_known_catalog() -> None:
    service = SkillService(_InMemorySkillRepository())

    mcp_items = await service.discover_mcp_skills()
    github_items = await service.discover_github_skills()

    assert len(mcp_items) > 0
    assert len(github_items) > 0
