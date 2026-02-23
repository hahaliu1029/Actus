from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.tool_result import ToolResult
from app.domain.services.tools.skill_bundle_sync import SkillBundleSyncManager

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.upload_paths: list[str] = []
        self.fail_upload_paths: set[str] = set()

    async def upload_file(self, file_data, filepath: str, filename: str | None = None) -> ToolResult:
        if filepath in self.fail_upload_paths:
            return ToolResult(success=False, message="upload failed")

        raw = file_data.read()
        if isinstance(raw, str):
            raw = raw.encode("utf-8")

        self.files[filepath] = raw
        self.upload_paths.append(filepath)
        return ToolResult(success=True, data={"filepath": filepath, "filename": filename})

    async def write_file(
        self,
        filepath: str,
        content: str,
        append: bool = False,
        leading_newline: bool = False,
        trailing_newline: bool = False,
        sudo: bool = False,
    ) -> ToolResult:
        if leading_newline:
            content = "\n" + content
        if trailing_newline:
            content = content + "\n"
        if append and filepath in self.files:
            self.files[filepath] = self.files[filepath] + content.encode("utf-8")
        else:
            self.files[filepath] = content.encode("utf-8")
        return ToolResult(success=True, data={"filepath": filepath})

    async def check_file_exists(self, filepath: str) -> ToolResult:
        return ToolResult(success=True, data={"exists": filepath in self.files})

    async def read_file(
        self,
        filepath: str,
        start_line: int | None = None,
        end_line: int | None = None,
        sudo: bool = False,
        max_length: int = 10000,
    ) -> ToolResult:
        if filepath not in self.files:
            return ToolResult(success=False, message="not found")
        content = self.files[filepath].decode("utf-8")
        return ToolResult(success=True, data={"filepath": filepath, "content": content[:max_length]})



def _build_native_skill(skill_id: str, *, version: str, bundle_file_count: int = 2) -> Skill:
    return Skill(
        id=skill_id,
        slug=skill_id,
        name=skill_id,
        description="demo",
        source_type=SkillSourceType.LOCAL,
        source_ref=f"local:{skill_id}",
        runtime_type=SkillRuntimeType.NATIVE,
        manifest={
            "runtime_type": "native",
            "tools": [],
            "bundle_file_count": bundle_file_count,
            "last_sync_at": version,
        },
        enabled=True,
    )



def _write_bundle(skills_root: Path, skill_id: str) -> None:
    bundle_dir = skills_root / skill_id / "bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "SKILL.md").write_text("# Demo", encoding="utf-8")
    (bundle_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (bundle_dir / "scripts" / "run.py").write_text("print('ok')", encoding="utf-8")


async def test_initial_sync_uploads_bundle_and_writes_marker(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager = SkillBundleSyncManager(
        sandbox=sandbox,
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )

    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager.await_initial_sync()

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id)

    assert error is None
    assert sandbox_dir == f"/home/ubuntu/workspace/.skills/{skill.id}"
    assert f"{sandbox_dir}/SKILL.md" in sandbox.upload_paths
    assert f"{sandbox_dir}/scripts/run.py" in sandbox.upload_paths
    assert f"{sandbox_dir}/.actus-sync.json" in sandbox.files


async def test_same_version_marker_skips_reupload(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager_first = SkillBundleSyncManager(
        sandbox=sandbox,
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager_first.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager_first.await_initial_sync()
    first_upload_count = len(sandbox.upload_paths)

    manager_second = SkillBundleSyncManager(
        sandbox=sandbox,
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager_second.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager_second.await_initial_sync()

    assert len(sandbox.upload_paths) == first_upload_count


async def test_concurrent_ensure_ready_syncs_once(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    manager = SkillBundleSyncManager(
        sandbox=sandbox,
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[])

    await asyncio.gather(
        manager.ensure_ready_for_invoke(skill.id),
        manager.ensure_ready_for_invoke(skill.id),
    )

    assert sandbox.upload_paths.count(f"/home/ubuntu/workspace/.skills/{skill.id}/SKILL.md") == 1
    assert sandbox.upload_paths.count(f"/home/ubuntu/workspace/.skills/{skill.id}/scripts/run.py") == 1


async def test_sync_failure_is_reported_to_invoke_path(tmp_path: Path) -> None:
    sandbox = _FakeSandbox()
    skill = _build_native_skill("pptx--1234", version="v1")
    _write_bundle(tmp_path, skill.id)

    sandbox.fail_upload_paths.add(f"/home/ubuntu/workspace/.skills/{skill.id}/scripts/run.py")

    manager = SkillBundleSyncManager(
        sandbox=sandbox,
        skills_root_dir=tmp_path,
        sandbox_skill_root="/home/ubuntu/workspace/.skills",
    )
    await manager.prepare_startup_sync(skill_pool=[skill], initial_selected=[skill])
    await manager.await_initial_sync()

    sandbox_dir, error = await manager.ensure_ready_for_invoke(skill.id)

    assert sandbox_dir is None
    assert error
    assert "上传文件失败" in error
