from __future__ import annotations

"""Skill metadata index service with simple cache invalidation."""

import asyncio
from pathlib import Path

from app.domain.models.skill import Skill
from app.domain.repositories.skill_repository import SkillRepository


class SkillIndexService:
    """基于目录版本号缓存已启用 Skill 元数据。"""

    def __init__(
        self,
        skill_repository: SkillRepository,
        skills_root: str | Path,
    ) -> None:
        self._skill_repository = skill_repository
        self._skills_root = Path(skills_root)
        self._version: float = -1.0
        self._cached_enabled_skills: list[Skill] = []
        self._lock = asyncio.Lock()

    async def list_enabled_skills(self) -> list[Skill]:
        async with self._lock:
            current = await asyncio.to_thread(self._compute_version)
            if current != self._version:
                self._cached_enabled_skills = await self._skill_repository.list_enabled()
                self._version = current
            return list(self._cached_enabled_skills)

    def _compute_version(self) -> float:
        if not self._skills_root.exists():
            return 0.0

        latest = self._skills_root.stat().st_mtime
        for path in self._skills_root.rglob("*"):
            try:
                latest = max(latest, path.stat().st_mtime)
            except FileNotFoundError:
                continue
        return latest
