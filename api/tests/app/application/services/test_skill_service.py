from __future__ import annotations

from pathlib import Path

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


async def test_install_skill_accepts_skill_md_without_manifest() -> None:
    service = SkillService(_InMemorySkillRepository())

    skill = await service.install_skill(
        source_type=SkillSourceType.GITHUB,
        source_ref="owner/repo",
        manifest={},
        skill_md="""---
name: Demo Skill
description: skill from markdown only
---
# Demo
""",
        installed_by="admin-1",
    )

    assert skill.name == "Demo Skill"
    assert skill.runtime_type == SkillRuntimeType.NATIVE
    assert skill.manifest["name"] == "Demo Skill"
    assert "skill_md" in skill.manifest


async def test_install_skill_parses_yaml_frontmatter_tools() -> None:
    service = SkillService(_InMemorySkillRepository())

    skill = await service.install_skill(
        source_type=SkillSourceType.LOCAL,
        source_ref="local:/tmp/demo-skill",
        manifest={},
        skill_md="""---
name: Demo Skill
description: yaml frontmatter with nested tools
runtime_type: native
tools:
  - name: run_demo
    description: run demo command
    parameters:
      query:
        type: string
    required:
      - query
    entry:
      exec_dir: /home/ubuntu/workspace
      command: echo demo
---
# Demo Skill
""",
        installed_by="admin-1",
    )

    tools = skill.manifest.get("tools")
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert tools[0]["name"] == "run_demo"


async def test_install_skill_loads_local_directory_bundle(tmp_path: Path) -> None:
    service = SkillService(_InMemorySkillRepository())
    skill_dir = tmp_path / "pptx"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: PPTX Skill\ndescription: build slide decks\n---\n# PPTX\nSee [guide](references/guide.md)\n",
        encoding="utf-8",
    )
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)
    (skill_dir / "references" / "guide.md").write_text(
        "Use python-pptx templates.",
        encoding="utf-8",
    )

    skill = await service.install_skill(
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_dir.as_posix()}",
        manifest={},
        skill_md="",
        installed_by="admin-1",
    )

    assert skill.name == "PPTX Skill"
    assert skill.manifest["bundle_file_count"] >= 2
    assert skill.manifest["context_ref_count"] == 1
    assert "references/guide.md" in skill.manifest["context_refs"]
    assert "reference:references/guide.md" in skill.manifest["context_blob"]


async def test_install_skill_prefers_skill_md_override_over_source(tmp_path: Path) -> None:
    service = SkillService(_InMemorySkillRepository())
    skill_dir = tmp_path / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: Source Name\n---\n# Source Skill\n",
        encoding="utf-8",
    )

    skill = await service.install_skill(
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_dir.as_posix()}",
        manifest={},
        skill_md="---\nname: Override Name\n---\n# Override Skill\n",
        installed_by="admin-1",
    )

    assert skill.name == "Override Name"
    assert "Override Skill" in skill.manifest["skill_md"]


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


async def test_install_skill_rejects_high_risk_command() -> None:
    service = SkillService(_InMemorySkillRepository())

    with pytest.raises(ValidationError):
        await service.install_skill(
            source_type=SkillSourceType.LOCAL,
            source_ref="local:/tmp/demo",
            manifest={
                "name": "Danger Skill",
                "runtime_type": SkillRuntimeType.NATIVE.value,
                "tools": [
                    {
                        "name": "danger",
                        "description": "danger",
                        "parameters": {},
                        "required": [],
                        "entry": {
                            "exec_dir": "/home/ubuntu/workspace",
                            "command": "rm -rf /",
                        },
                    }
                ],
            },
            skill_md="# danger",
            installed_by="admin-1",
        )


async def test_install_skill_rejects_legacy_source_type() -> None:
    service = SkillService(_InMemorySkillRepository())

    with pytest.raises(ValidationError):
        await service.install_skill(
            source_type=SkillSourceType.MCP_REGISTRY,
            source_ref="mcp:legacy",
            manifest={
                "name": "Legacy Skill",
                "runtime_type": SkillRuntimeType.NATIVE.value,
                "tools": [
                    {
                        "name": "run",
                        "description": "run",
                        "parameters": {},
                        "required": [],
                        "entry": {
                            "exec_dir": "/home/ubuntu/workspace",
                            "command": "echo ok",
                        },
                    }
                ],
            },
            skill_md="# legacy",
            installed_by="admin-1",
        )
