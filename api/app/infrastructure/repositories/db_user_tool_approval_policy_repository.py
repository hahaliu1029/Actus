"""SQLAlchemy 仓储实现：UserToolApprovalPolicy。"""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from app.domain.repositories.user_tool_approval_policy_repository import (
    UserToolApprovalPolicyRepository,
)
from app.infrastructure.models.user_tool_approval_policy import (
    UserToolApprovalPolicyModel,
)


class DBUserToolApprovalPolicyRepository(UserToolApprovalPolicyRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def get(
        self, user_id: str, tool_name: str
    ) -> Optional[UserToolApprovalPolicy]:
        stmt = select(UserToolApprovalPolicyModel).where(
            and_(
                UserToolApprovalPolicyModel.user_id == user_id,
                UserToolApprovalPolicyModel.tool_name == tool_name,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        return record.to_domain() if record else None

    async def list_by_user(
        self, user_id: str
    ) -> list[UserToolApprovalPolicy]:
        stmt = (
            select(UserToolApprovalPolicyModel)
            .where(UserToolApprovalPolicyModel.user_id == user_id)
            .order_by(UserToolApprovalPolicyModel.tool_name.asc())
        )
        result = await self.db_session.execute(stmt)
        records = result.scalars().all()
        return [r.to_domain() for r in records]

    async def upsert(
        self, policy: UserToolApprovalPolicy
    ) -> UserToolApprovalPolicy:
        # 原子 upsert：INSERT ... ON CONFLICT DO UPDATE。单条 SQL 语句下
        # 并发 PUT 到同一 (user_id, tool_name) 不会双读 miss + 二次插入
        # 撞 UNIQUE。冲突时保留原 id/created_at，仅刷新 policy/updated_at。
        now = datetime.now()
        insert_stmt = pg_insert(UserToolApprovalPolicyModel).values(
            id=policy.id,
            user_id=policy.user_id,
            tool_name=policy.tool_name,
            policy=policy.policy.value,
            created_at=policy.created_at,
            updated_at=now,
        )
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=["user_id", "tool_name"],
            set_={
                "policy": policy.policy.value,
                "updated_at": now,
            },
        ).returning(
            UserToolApprovalPolicyModel.id,
            UserToolApprovalPolicyModel.user_id,
            UserToolApprovalPolicyModel.tool_name,
            UserToolApprovalPolicyModel.policy,
            UserToolApprovalPolicyModel.created_at,
            UserToolApprovalPolicyModel.updated_at,
        )
        result = await self.db_session.execute(upsert_stmt)
        await self.db_session.flush()
        row = result.one()
        return UserToolApprovalPolicy(
            id=row.id,
            user_id=row.user_id,
            tool_name=row.tool_name,
            policy=ApprovalPolicy(row.policy),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def delete(self, user_id: str, tool_name: str) -> bool:
        stmt = select(UserToolApprovalPolicyModel).where(
            and_(
                UserToolApprovalPolicyModel.user_id == user_id,
                UserToolApprovalPolicyModel.tool_name == tool_name,
            )
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return False
        await self.db_session.delete(record)
        await self.db_session.flush()
        return True
