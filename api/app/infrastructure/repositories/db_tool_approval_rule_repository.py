"""工具审批规则仓储实现"""

from app.domain.models.tool_approval_rule import ToolApprovalRule
from app.domain.repositories.tool_approval_rule_repository import (
    ToolApprovalRuleRepository,
)
from app.infrastructure.models.tool_approval_rule import ToolApprovalRuleModel
from sqlalchemy import and_, delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class DBToolApprovalRuleRepository(ToolApprovalRuleRepository):
    """基于数据库的工具审批规则仓储实现"""

    def __init__(self, db_session: AsyncSession) -> None:
        """构造函数，完成数据仓储初始化"""
        self.db_session = db_session

    async def find_by_user_and_tool(
        self, user_id: str, tool_name: str
    ) -> list[ToolApprovalRule]:
        """根据用户 ID 和工具名称查询审批规则"""
        stmt = select(ToolApprovalRuleModel).where(
            and_(
                ToolApprovalRuleModel.user_id == user_id,
                ToolApprovalRuleModel.tool_name == tool_name,
            )
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]

    async def create(self, rule: ToolApprovalRule) -> ToolApprovalRule:
        """创建工具审批规则"""
        record = ToolApprovalRuleModel.from_domain(rule)
        self.db_session.add(record)
        await self.db_session.flush()
        return record.to_domain()

    async def delete(self, rule_id: str) -> None:
        """删除工具审批规则"""
        stmt = delete(ToolApprovalRuleModel).where(
            ToolApprovalRuleModel.id == rule_id
        )
        await self.db_session.execute(stmt)

    async def find_by_user(self, user_id: str) -> list[ToolApprovalRule]:
        """根据用户 ID 查询所有审批规则"""
        stmt = select(ToolApprovalRuleModel).where(
            ToolApprovalRuleModel.user_id == user_id
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [record.to_domain() for record in records]
