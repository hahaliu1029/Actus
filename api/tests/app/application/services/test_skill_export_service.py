from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.skill_export_service import SkillExportService
from app.interfaces.schemas.skill import SkillExportFormat

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_skill_on_disk(tmp_path: Path, skill_id: str = "test-skill--abc12345", runtime_type: str = "native", slug: str = "test-skill", skill_md: str = "---\nname: test-skill\ndescription: A test skill\n---\n# Test Skill\n\nInstructions here.\n") -> str:
    """在 tmp_path 下构建一个完整的 skill 目录并返回 skill_id。"""
    skill_dir = tmp_path / skill_id
    skill_dir.mkdir(exist_ok=True)

    meta = {
        "id": skill_id,
        "slug": slug,
        "name": "Test Skill",
        "description": "A test skill",
        "version": "0.1.0",
        "source_type": "local",
        "source_ref": "local:/tmp/test",
        "runtime_type": runtime_type,
        "enabled": True,
        "installed_by": "admin",
        "created_at": "2026-03-10T00:00:00",
        "updated_at": "2026-03-10T00:00:00",
    }
    manifest = {
        "name": "Test Skill",
        "description": "A test skill",
        "runtime_type": runtime_type,
        "version": "0.1.0",
        "tools": [{"name": "run_test", "description": "run test", "parameters": {}, "required": [], "entry": {"command": "python bundle/run_test.py"}}],
        "activation": {},
        "policy": {"risk_level": "low"},
        "security": {},
        "skill_md": skill_md,
    }

    (skill_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (skill_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (skill_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")

    if runtime_type == "native":
        bundle_dir = skill_dir / "bundle"
        bundle_dir.mkdir(exist_ok=True)
        (bundle_dir / "run_test.py").write_text("print('hello')", encoding="utf-8")
        bundle_index = [{"path": "run_test.py", "size": 14, "sha256": "abc", "is_text": False}]
        (skill_dir / "bundle_index.json").write_text(json.dumps(bundle_index), encoding="utf-8")

    return skill_id


# ---- Actus format tests ----

async def test_export_actus_format(tmp_path: Path) -> None:
    skill_id = _make_skill_on_disk(tmp_path)
    service = SkillExportService(skills_root_dir=tmp_path)

    zip_bytes, filename = await service.export_skill(skill_id, SkillExportFormat.ACTUS)

    assert filename == "test-skill-actus.zip"
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "test-skill/meta.json" in names
        assert "test-skill/manifest.json" in names
        assert "test-skill/SKILL.md" in names
        assert "test-skill/bundle/run_test.py" in names
        assert "test-skill/bundle_index.json" in names


async def test_export_nonexistent_skill_raises(tmp_path: Path) -> None:
    service = SkillExportService(skills_root_dir=tmp_path)
    with pytest.raises(NotFoundError):
        await service.export_skill("nonexistent", SkillExportFormat.ACTUS)


# ---- Agent Skills format tests ----

async def test_export_agent_skills_format(tmp_path: Path) -> None:
    skill_id = _make_skill_on_disk(tmp_path)
    service = SkillExportService(skills_root_dir=tmp_path)

    zip_bytes, filename = await service.export_skill(skill_id, SkillExportFormat.AGENT_SKILLS)

    assert filename == "test-skill-agent-skills.zip"
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "test-skill/SKILL.md" in names
        assert "test-skill/scripts/run_test.py" in names
        assert "test-skill/meta.json" not in names
        assert "test-skill/manifest.json" not in names

        skill_md = zf.read("test-skill/SKILL.md").decode("utf-8")
        assert "name: test-skill" in skill_md
        assert "description: A test skill" in skill_md


async def test_export_agent_skills_mcp_skill(tmp_path: Path) -> None:
    skill_id = _make_skill_on_disk(
        tmp_path,
        skill_id="mcp-skill--abc12345",
        runtime_type="mcp",
        slug="mcp-skill",
        skill_md="---\nname: mcp-skill\ndescription: An MCP skill\n---\n# MCP Skill\n",
    )
    service = SkillExportService(skills_root_dir=tmp_path)

    zip_bytes, _ = await service.export_skill(skill_id, SkillExportFormat.AGENT_SKILLS)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        assert "mcp-skill/SKILL.md" in names
        skill_md = zf.read("mcp-skill/SKILL.md").decode("utf-8")
        assert "compatibility:" in skill_md
        assert "MCP" in skill_md


async def test_export_agent_skills_empty_skill_md(tmp_path: Path) -> None:
    skill_id = _make_skill_on_disk(
        tmp_path,
        skill_id="empty-md--abc12345",
        slug="empty-md",
        skill_md="",
    )
    # Override meta to match
    skill_dir = tmp_path / skill_id
    meta = json.loads((skill_dir / "meta.json").read_text(encoding="utf-8"))
    meta["slug"] = "empty-md"
    meta["name"] = "Empty MD Skill"
    meta["description"] = "A skill with no SKILL.md content"
    (skill_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    manifest = json.loads((skill_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["name"] = "Empty MD Skill"
    manifest["description"] = "A skill with no SKILL.md content"
    manifest["skill_md"] = ""
    (skill_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    service = SkillExportService(skills_root_dir=tmp_path)
    zip_bytes, _ = await service.export_skill(skill_id, SkillExportFormat.AGENT_SKILLS)

    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        skill_md = zf.read("empty-md/SKILL.md").decode("utf-8")
        assert "name: empty-md" in skill_md
        assert "A skill with no SKILL.md content" in skill_md
