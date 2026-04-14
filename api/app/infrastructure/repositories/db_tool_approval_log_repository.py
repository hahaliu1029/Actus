"""工具审批日志仓储实现（审计只写）"""

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.repositories.tool_approval_log_repository import ToolApprovalLogRepository
from app.infrastructure.models.tool_approval_log import ToolApprovalLogModel


class DBToolApprovalLogRepository(ToolApprovalLogRepository):
    """基于数据库的工具审批日志仓储实现（审计只写）"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create(
        self,
        user_id: str,
        session_id: str,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: str,
        action: str,
        scope: str,
        approved_by: str,
    ) -> None:
        """写入一条审批日志记录"""
        record = ToolApprovalLogModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            session_id=session_id,
            tool_name=tool_name,
            tool_args=tool_args,
            risk_level=risk_level,
            action=action,
            scope=scope,
            approved_by=approved_by,
        )
        self.db_session.add(record)
        await self.db_session.flush()
