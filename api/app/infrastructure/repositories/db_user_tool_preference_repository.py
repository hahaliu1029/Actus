"""用户工具偏好仓储实现"""

from typing import Optional

from app.domain.models.user_tool_preference import ToolType, UserToolPreference
from app.domain.repositories.user_tool_preference_repository import (
    UserToolPreferenceRepository,
)
from app.infrastructure.models.user_tool_preference import UserToolPreferenceModel
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBUserToolPreferenceRepository(UserToolPreferenceRepository):
    """基于数据库的用户工具偏好仓储实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓储初始化"""
        self.db_session = db_session

    async def create(self, preference: UserToolPreference) -> UserToolPreference:
        """创建用户工具偏好"""
        record = UserToolPreferenceModel.from_domain(preference)
        self.db_session.add(record)
        await self.db_session.flush()
        return record.to_domain()

    async def get_by_id(self, preference_id: str) -> Optional[UserToolPreference]:
        """根据 ID 获取用户工具偏好"""
        stmt = select(UserToolPreferenceModel).where(
            UserToolPreferenceModel.id == preference_id
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_user_and_tool(
        self, user_id: str, tool_type: ToolType, tool_id: str
    ) -> Optional[UserToolPreference]:
        """根据用户 ID、工具类型和工具 ID 获取偏好"""
        stmt = select(UserToolPreferenceModel).where(
            and_(
                UserToolPreferenceModel.user_id == user_id,
                UserToolPreferenceModel.tool_type == tool_type.value,
                UserToolPreferenceModel.tool_id == tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_user_id(
        self, user_id: str, tool_type: Optional[ToolType] = None
    ) -> list[UserToolPreference]:
        """获取用户的所有工具偏好，可选按类型过滤"""
        if tool_type:
            stmt = select(UserToolPreferenceModel).where(
                and_(
                    UserToolPreferenceModel.user_id == user_id,
                    UserToolPreferenceModel.tool_type == tool_type.value,
                )
            )
        else:
            stmt = select(UserToolPreferenceModel).where(
                UserToolPreferenceModel.user_id == user_id
            )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def upsert(self, preference: UserToolPreference) -> UserToolPreference:
        """创建或更新用户工具偏好"""
        # 查找已存在的记录
        stmt = select(UserToolPreferenceModel).where(
            and_(
                UserToolPreferenceModel.user_id == preference.user_id,
                UserToolPreferenceModel.tool_type == preference.tool_type.value,
                UserToolPreferenceModel.tool_id == preference.tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            # 更新
            record.update_from_domain(preference)
            await self.db_session.flush()
            return record.to_domain()
        else:
            # 创建
            return await self.create(preference)

    async def delete(self, preference_id: str) -> bool:
        """删除用户工具偏好"""
        stmt = select(UserToolPreferenceModel).where(
            UserToolPreferenceModel.id == preference_id
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            await self.db_session.delete(record)
            return True
        return False

    async def delete_by_tool(self, tool_type: ToolType, tool_id: str) -> int:
        """删除指定工具的所有用户偏好（当工具被删除时）"""
        stmt = delete(UserToolPreferenceModel).where(
            and_(
                UserToolPreferenceModel.tool_type == tool_type.value,
                UserToolPreferenceModel.tool_id == tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        return result.rowcount
