from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.infrastructure.repositories.file_skill_repository import FileSkillRepository

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_skill(skill_id: str) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=f"Skill {skill_id}",
        description="demo",
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "name": f"Skill {skill_id}",
            "runtime_type": "native",
            "tools": [{"name": "run", "description": "run"}],
            "skill_md": "# Demo",
        },
        enabled=True,
        installed_by="admin-1",
    )


async def test_file_skill_repository_upsert_and_list(tmp_path: Path) -> None:
    repo = FileSkillRepository(root_dir=tmp_path)
    skill = _build_skill("demo-skill--1234abcd")

    await repo.upsert(skill)
    listed = await repo.list()

    assert len(listed) == 1
    assert listed[0].id == skill.id
    assert listed[0].source_type == SkillSourceType.LOCAL


async def test_file_skill_repository_writes_expected_files(tmp_path: Path) -> None:
    repo = FileSkillRepository(root_dir=tmp_path)
    skill = _build_skill("demo-skill--1234abcd")

    await repo.upsert(skill)

    skill_dir = tmp_path / skill.id
    assert (skill_dir / "meta.json").exists()
    assert (skill_dir / "manifest.json").exists()
    assert (skill_dir / "SKILL.md").exists()

    meta = json.loads((skill_dir / "meta.json").read_text())
    assert meta["id"] == skill.id
    assert meta["source_type"] == "local"


async def test_file_skill_repository_writes_bundle_files(tmp_path: Path) -> None:
    repo = FileSkillRepository(root_dir=tmp_path)
    skill = _build_skill("bundle-skill--1234abcd")
    skill.manifest["_bundle_files"] = {
        "SKILL.md": b"# bundle skill",
        "references/guide.md": b"guide",
        "assets/icon.bin": b"\x00\x01",
    }

    await repo.upsert(skill)
    skill_dir = tmp_path / skill.id

    assert (skill_dir / "bundle" / "SKILL.md").exists()
    assert (skill_dir / "bundle" / "references" / "guide.md").exists()
    assert (skill_dir / "bundle" / "assets" / "icon.bin").exists()
    assert (skill_dir / "bundle_index.json").exists()
