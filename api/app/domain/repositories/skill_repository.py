from __future__ import annotations

"""Skill 仓储接口"""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.skill import Skill


class SkillRepository(ABC):
    """Skill 数据仓储抽象接口"""

    @abstractmethod
    async def list(self) -> list[Skill]:
        """获取全部 Skill"""
        pass

    @abstractmethod
    async def list_enabled(self) -> list[Skill]:
        """获取全部启用状态的 Skill"""
        pass

    @abstractmethod
    async def get_by_id(self, skill_id: str) -> Optional[Skill]:
        """根据 ID 获取 Skill"""
        pass

    @abstractmethod
    async def get_by_slug(self, slug: str) -> Optional[Skill]:
        """根据 slug 获取 Skill"""
        pass

    @abstractmethod
    async def upsert(self, skill: Skill) -> Skill:
        """创建或更新 Skill"""
        pass

    @abstractmethod
    async def delete(self, skill_id: str) -> bool:
        """删除 Skill"""
        pass
