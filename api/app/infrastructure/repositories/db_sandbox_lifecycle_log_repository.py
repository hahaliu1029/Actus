"""Sandbox 生命周期审计日志仓储实现（审计只写）"""

import uuid
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.repositories.sandbox_lifecycle_log_repository import (
    SandboxLifecycleLogRepository,
)
from app.infrastructure.models.sandbox_lifecycle_log import (
    SandboxLifecycleLogModel,
)


class DBSandboxLifecycleLogRepository(SandboxLifecycleLogRepository):
    """基于数据库的 Sandbox 生命周期审计日志仓储实现（审计只写）"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create(
        self,
        session_id: str,
        old_state: str,
        new_state: str,
        generation: int,
        sandbox_id: Optional[str] = None,
        reason: Optional[str] = None,
        triggered_by: Optional[str] = None,
    ) -> None:
        """写入一条生命周期审计日志记录"""
        record = SandboxLifecycleLogModel(
            id=str(uuid.uuid4()),
            session_id=session_id,
            old_state=old_state,
            new_state=new_state,
            generation=generation,
            sandbox_id=sandbox_id,
            reason=reason,
            triggered_by=triggered_by,
        )
        self.db_session.add(record)
