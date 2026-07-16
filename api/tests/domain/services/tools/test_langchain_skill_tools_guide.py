"""Tests for get_skill_guide tool with resource layer support."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from app.application.services.sandbox_accessors import EagerSandboxAccessor
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.langchain_skill_tools import create_skill_guide_tool
from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_skill(
    slug="xlsx", name="XLSX Processor", skill_id="abc123",
    description="Process Excel", skill_md="# Guide\nUse openpyxl.",
):
    skill = MagicMock()
    skill.slug = slug
    skill.name = name
    skill.id = skill_id
    skill.description = description
    skill.manifest = {"skill_md": f"---\nname: {slug}\n---\n{skill_md}"}
    return skill


class TestGetSkillGuideWithResources:
    """Test that get_skill_guide returns resource info when file listing is available."""

    async def test_guide_includes_sandbox_path_and_files(self):
        skill = _make_skill()
        pool = [skill]
        listings = {"abc123": ["main.py", "scripts/run.sh", "references/guide.md"]}

        tool = create_skill_guide_tool(
            skill_pool_ref=lambda: pool,
            file_listings_ref=lambda: listings,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )

        result = await tool.ainvoke({"skill_slug": "xlsx"})
        assert "/home/ubuntu/workspace/.skills/abc123/" in result
        assert "main.py" in result
        assert "scripts/run.sh" in result
        assert "references/guide.md" in result
        assert "file_read" in result

    async def test_guide_without_file_listing(self):
        """When no file listing is cached, no resource section is appended."""
        skill = _make_skill()
        tool = create_skill_guide_tool(
            skill_pool_ref=lambda: [skill],
            file_listings_ref=lambda: {},
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )
        result = await tool.ainvoke({"skill_slug": "xlsx"})
        assert "# XLSX Processor" in result
        assert "Skill Resources" not in result

    async def test_guide_no_listings_ref(self):
        """When file_listings_ref is None (backward compat), no resource section."""
        skill = _make_skill()
        tool = create_skill_guide_tool(skill_pool_ref=lambda: [skill])
        result = await tool.ainvoke({"skill_slug": "xlsx"})
        assert "# XLSX Processor" in result
        assert "Skill Resources" not in result


# ---------------------------------------------------------------------------
# End-to-end: sync manager cache -> guide tool -> resource info
# ---------------------------------------------------------------------------


def _make_real_skill(skill_id: str, slug: str, name: str, skill_md: str) -> Skill:
    return Skill(
        id=skill_id, slug=slug, name=name, description=name,
        source_type=SkillSourceType.LOCAL, source_ref=f"local:{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={"bundle_file_count": 2, "skill_md": skill_md,
                  "runtime_type": "native", "tools": []},
        enabled=True,
    )


class _SimpleSandbox:
    """Minimal sandbox mock for sync tests. Accepts all uploads."""

    def __init__(self):
        self.files: dict[str, bytes] = {}

    async def upload_file(self, file_data, filepath, filename=None):
        self.files[filepath] = file_data.read()
        return ToolResult(success=True)

    async def write_file(self, filepath, content, **kw):
        self.files[filepath] = content.encode() if isinstance(content, str) else content
        return ToolResult(success=True)

    async def check_file_exists(self, filepath):
        return ToolResult(success=True, data={"exists": filepath in self.files})

    async def read_file(self, filepath, **kw):
        if filepath not in self.files:
            return ToolResult(success=False)
        return ToolResult(success=True, data={"content": self.files[filepath].decode()})


class TestGetSkillGuideEndToEnd:
    """End-to-end: sync manager cache -> guide tool -> resource info."""

    async def test_full_flow(self, tmp_path):
        """Simulate full flow: sync skill -> get guide with resources."""
        skills_root = tmp_path / "skills"
        bundle_dir = skills_root / "test-id" / "bundle"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "main.py").write_text("print('hi')")
        (bundle_dir / "README.md").write_text("# Readme")

        sandbox = _SimpleSandbox()
        manager = SkillBundleSyncManager(
            sandbox_accessor=EagerSandboxAccessor(sandbox), skills_root_dir=skills_root,
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )

        skill = _make_real_skill("test-id", "test-skill", "Test Skill", "# Test\nDo things.")
        await manager.prepare_startup_sync([skill], [skill])
        await manager.await_initial_sync()

        tool = create_skill_guide_tool(
            skill_pool_ref=lambda: [skill],
            file_listings_ref=lambda: manager.get_file_listing_all(),
            sandbox_skill_root="/home/ubuntu/workspace/.skills",
        )

        result = await tool.ainvoke({"skill_slug": "test-skill"})
        assert "# Test Skill" in result
        assert "Do things." in result
        assert "/home/ubuntu/workspace/.skills/test-id/" in result
        assert "main.py" in result
        assert "README.md" in result
