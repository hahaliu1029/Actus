"""用户工具扩展启用仓储实现。"""

from typing import Optional

from app.domain.models.user_tool_enablement import ToolType, UserToolEnablement
from app.domain.repositories.user_tool_enablement_repository import (
    UserToolEnablementRepository,
)
from app.infrastructure.models.user_tool_enablement import UserToolEnablementModel
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBUserToolEnablementRepository(UserToolEnablementRepository):
    """基于数据库的用户工具扩展启用仓储实现。"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓储初始化。"""
        self.db_session = db_session

    async def create(self, enablement: UserToolEnablement) -> UserToolEnablement:
        """创建用户工具扩展启用记录。"""
        record = UserToolEnablementModel.from_domain(enablement)
        self.db_session.add(record)
        await self.db_session.flush()
        return record.to_domain()

    async def get_by_id(self, enablement_id: str) -> Optional[UserToolEnablement]:
        """根据 ID 获取用户工具扩展启用记录。"""
        stmt = select(UserToolEnablementModel).where(
            UserToolEnablementModel.id == enablement_id
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_user_and_tool(
        self, user_id: str, tool_type: ToolType, tool_id: str
    ) -> Optional[UserToolEnablement]:
        """根据用户 ID、工具类型和工具 ID 获取启用记录。"""
        stmt = select(UserToolEnablementModel).where(
            and_(
                UserToolEnablementModel.user_id == user_id,
                UserToolEnablementModel.tool_type == tool_type.value,
                UserToolEnablementModel.tool_id == tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def get_by_user_id(
        self, user_id: str, tool_type: Optional[ToolType] = None
    ) -> list[UserToolEnablement]:
        """获取用户的所有工具扩展启用记录，可选按类型过滤。"""
        if tool_type:
            stmt = select(UserToolEnablementModel).where(
                and_(
                    UserToolEnablementModel.user_id == user_id,
                    UserToolEnablementModel.tool_type == tool_type.value,
                )
            )
        else:
            stmt = select(UserToolEnablementModel).where(
                UserToolEnablementModel.user_id == user_id
            )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def upsert(self, enablement: UserToolEnablement) -> UserToolEnablement:
        """创建或更新用户工具扩展启用记录。"""
        # 查找已存在的记录
        stmt = select(UserToolEnablementModel).where(
            and_(
                UserToolEnablementModel.user_id == enablement.user_id,
                UserToolEnablementModel.tool_type == enablement.tool_type.value,
                UserToolEnablementModel.tool_id == enablement.tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            # 更新
            record.update_from_domain(enablement)
            await self.db_session.flush()
            return record.to_domain()
        else:
            # 创建
            return await self.create(enablement)

    async def delete(self, enablement_id: str) -> bool:
        """删除用户工具扩展启用记录。"""
        stmt = select(UserToolEnablementModel).where(
            UserToolEnablementModel.id == enablement_id
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        if record:
            await self.db_session.delete(record)
            return True
        return False

    async def delete_by_tool(self, tool_type: ToolType, tool_id: str) -> int:
        """删除指定工具的所有用户扩展启用记录（当工具被删除时）。"""
        stmt = delete(UserToolEnablementModel).where(
            and_(
                UserToolEnablementModel.tool_type == tool_type.value,
                UserToolEnablementModel.tool_id == tool_id,
            )
        )
        result = await self.db_session.execute(stmt)
        return result.rowcount
