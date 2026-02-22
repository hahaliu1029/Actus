from __future__ import annotations

"""Skill 仓储实现"""

from app.domain.models.skill import Skill
from app.domain.repositories.skill_repository import SkillRepository
from app.infrastructure.models.skill import SkillModel
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBSkillRepository(SkillRepository):
    """基于数据库的 Skill 仓储实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def list(self) -> list[Skill]:
        stmt = select(SkillModel).order_by(SkillModel.created_at.desc())
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def list_enabled(self) -> list[Skill]:
        stmt = (
            select(SkillModel)
            .where(SkillModel.enabled.is_(True))
            .order_by(SkillModel.created_at.desc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def get_by_id(self, skill_id: str) -> Skill | None:
        stmt = select(SkillModel).where(SkillModel.id == skill_id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_slug(self, slug: str) -> Skill | None:
        stmt = select(SkillModel).where(SkillModel.slug == slug)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def upsert(self, skill: Skill) -> Skill:
        stmt = select(SkillModel).where(SkillModel.id == skill.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if not record:
            slug_stmt = select(SkillModel).where(SkillModel.slug == skill.slug)
            slug_result = await self.db_session.execute(slug_stmt)
            record = slug_result.scalar_one_or_none()

        if record:
            record.update_from_domain(skill)
            await self.db_session.flush()
            return record.to_domain()

        created = SkillModel.from_domain(skill)
        self.db_session.add(created)
        await self.db_session.flush()
        return created.to_domain()

    async def delete(self, skill_id: str) -> bool:
        stmt = delete(SkillModel).where(SkillModel.id == skill_id)
        result = await self.db_session.execute(stmt)
        return (result.rowcount or 0) > 0
