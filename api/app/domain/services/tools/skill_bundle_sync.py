"""Skill bundle synchronization manager for sandbox runtime."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from app.domain.external.sandbox import Sandbox
from app.domain.models.skill import Skill, SkillRuntimeType

logger = logging.getLogger(__name__)

SYNC_STATUS = Literal["pending", "running", "success", "failed"]
SYNC_MARKER_FILENAME = ".actus-sync.json"
DEFAULT_BACKGROUND_CONCURRENCY = 4


@dataclass
class SkillSyncState:
    status: SYNC_STATUS = "pending"
    version: str = ""
    sandbox_dir: str = ""
    error: str | None = None
    task: asyncio.Task[str | None] | None = None


class SkillBundleSyncManager:
    """Sync skill bundle files from API filesystem into sandbox filesystem."""

    def __init__(
        self,
        sandbox: Sandbox,
        skills_root_dir: str | Path,
        sandbox_skill_root: str,
        background_concurrency: int = DEFAULT_BACKGROUND_CONCURRENCY,
    ) -> None:
        self._sandbox = sandbox
        self._skills_root_dir = Path(skills_root_dir)
        self._sandbox_skill_root = str(sandbox_skill_root).rstrip("/")
        self._background_concurrency = max(1, int(background_concurrency or 1))
        self._skill_pool: dict[str, Skill] = {}
        self._sync_states: dict[str, SkillSyncState] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._initial_tasks: list[asyncio.Task[str | None]] = []
        self._background_task: asyncio.Task[None] | None = None
        self._background_skills: list[Skill] = []

    async def prepare_startup_sync(
        self,
        skill_pool: list[Skill],
        initial_selected: list[Skill],
    ) -> None:
        """Prepare startup synchronization tasks.

        Initial selected skills are synchronized in foreground (blocking).
        Remaining syncable skills are prepared for background sync.
        """
        self._skill_pool = {skill.id: skill for skill in skill_pool}
        selected_ids = {skill.id for skill in initial_selected}

        self._initial_tasks = []
        self._background_skills = []

        for skill in skill_pool:
            if not self._needs_sync(skill):
                continue
            if skill.id in selected_ids:
                task = self._ensure_sync_task(skill)
                self._initial_tasks.append(task)
            else:
                self._background_skills.append(skill)

    async def await_initial_sync(self) -> None:
        """Wait startup foreground sync tasks to complete."""
        if not self._initial_tasks:
            return
        results = await asyncio.gather(*self._initial_tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.warning("前台Skill bundle同步任务异常: %s", str(result))

    def start_background_sync(self) -> None:
        """Start background synchronization for remaining skills."""
        if self._background_task or not self._background_skills:
            return
        self._background_task = asyncio.create_task(self._run_background_sync())

    async def ensure_ready_for_invoke(
        self,
        skill_id: str,
        *,
        skill: Skill | None = None,
    ) -> tuple[str | None, str | None]:
        """Ensure a skill bundle is synchronized before native invoke."""
        resolved_skill = skill or self._skill_pool.get(skill_id)
        if resolved_skill is None:
            return None, f"Skill[{skill_id}]未在当前会话同步池中"

        if not self._needs_sync(resolved_skill):
            return None, None

        self._skill_pool[resolved_skill.id] = resolved_skill
        task = self._ensure_sync_task(resolved_skill)
        await task

        state = self._sync_states.get(resolved_skill.id)
        if not state:
            return None, f"Skill[{resolved_skill.id}]同步状态缺失"
        if state.status == "failed":
            return None, state.error or f"Skill[{resolved_skill.id}]同步失败"
        if state.status != "success":
            return None, f"Skill[{resolved_skill.id}]同步未完成"
        return state.sandbox_dir or None, None

    async def cleanup(self) -> None:
        """Cancel and cleanup pending synchronization tasks."""
        tasks: list[asyncio.Task] = []
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            tasks.append(self._background_task)

        for state in self._sync_states.values():
            if state.task and not state.task.done():
                state.task.cancel()
                tasks.append(state.task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        self._background_task = None
        self._initial_tasks = []
        self._background_skills = []

    def _ensure_sync_task(self, skill: Skill) -> asyncio.Task[str | None]:
        state = self._sync_states.get(skill.id)
        if state and state.task:
            return state.task

        version = self._version_of(skill)
        state = SkillSyncState(status="pending", version=version)
        task = asyncio.create_task(self._sync_skill(skill))
        state.task = task
        self._sync_states[skill.id] = state
        return task

    async def _sync_skill(self, skill: Skill) -> str | None:
        lock = self._locks.setdefault(skill.id, asyncio.Lock())
        state = self._sync_states.setdefault(
            skill.id, SkillSyncState(version=self._version_of(skill))
        )

        async with lock:
            state.status = "running"
            state.error = None
            try:
                sandbox_dir = await self._sync_bundle(skill)
                state.status = "success"
                state.sandbox_dir = sandbox_dir or ""
                return sandbox_dir
            except Exception as e:  # noqa: BLE001
                state.status = "failed"
                state.error = str(e)
                logger.warning("Skill bundle同步失败(skill=%s): %s", skill.id, str(e))
                return None

    async def _run_background_sync(self) -> None:
        semaphore = asyncio.Semaphore(self._background_concurrency)

        async def _worker(skill: Skill) -> None:
            async with semaphore:
                task = self._ensure_sync_task(skill)
                await task

        try:
            await asyncio.gather(
                *[_worker(skill) for skill in self._background_skills],
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._background_task = None

    async def _sync_bundle(self, skill: Skill) -> str | None:
        bundle_count = self._bundle_file_count(skill)
        if bundle_count <= 0:
            return None

        sandbox_skill_dir = f"{self._sandbox_skill_root}/{skill.id}"
        marker_path = f"{sandbox_skill_dir}/{SYNC_MARKER_FILENAME}"
        version = self._version_of(skill)
        marker_version = await self._read_marker_version(marker_path)
        if marker_version and marker_version == version:
            return sandbox_skill_dir

        bundle_dir = self._skills_root_dir / skill.id / "bundle"
        if not bundle_dir.exists() or not bundle_dir.is_dir():
            raise RuntimeError(f"Skill[{skill.id}] bundle目录不存在: {bundle_dir}")

        files = sorted(path for path in bundle_dir.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Skill[{skill.id}] bundle为空，无法同步")

        for source_path in files:
            rel_path = source_path.relative_to(bundle_dir).as_posix()
            target_path = f"{sandbox_skill_dir}/{rel_path}"
            with source_path.open("rb") as fp:
                result = await self._sandbox.upload_file(
                    file_data=fp,
                    filepath=target_path,
                    filename=source_path.name,
                )
            if not result.success:
                raise RuntimeError(
                    f"上传文件失败: {rel_path} ({result.message or 'unknown error'})"
                )

        marker = json.dumps(
            {
                "skill_id": skill.id,
                "version": version,
                "synced_at": datetime.now().isoformat(),
                "bundle_file_count": len(files),
            },
            ensure_ascii=False,
        )
        marker_result = await self._sandbox.write_file(
            filepath=marker_path,
            content=marker,
        )
        if not marker_result.success:
            raise RuntimeError(
                f"写入同步标记失败: {marker_result.message or 'unknown error'}"
            )
        return sandbox_skill_dir

    async def _read_marker_version(self, marker_path: str) -> str:
        exists_result = await self._sandbox.check_file_exists(marker_path)
        if not exists_result.success:
            return ""
        data = exists_result.data if isinstance(exists_result.data, dict) else {}
        if not data.get("exists"):
            return ""

        read_result = await self._sandbox.read_file(filepath=marker_path, max_length=4096)
        if not read_result.success:
            return ""
        read_data = read_result.data if isinstance(read_result.data, dict) else {}
        content = str(read_data.get("content") or "").strip()
        if not content:
            return ""
        try:
            payload = json.loads(content)
        except Exception:
            return ""
        return str(payload.get("version") or "")

    @staticmethod
    def _needs_sync(skill: Skill) -> bool:
        return (
            skill.runtime_type == SkillRuntimeType.NATIVE
            and SkillBundleSyncManager._bundle_file_count(skill) > 0
        )

    @staticmethod
    def _bundle_file_count(skill: Skill) -> int:
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        raw = manifest.get("bundle_file_count")
        try:
            return int(raw or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _version_of(skill: Skill) -> str:
        manifest = skill.manifest if isinstance(skill.manifest, dict) else {}
        return str(manifest.get("last_sync_at") or "static")
