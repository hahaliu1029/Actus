"""R5 CS4 grants 仓储的 SQLAlchemy 实现。

职责：把 ORM ``ToolApprovalGrantModel`` 翻译成 domain ``ApprovalGrant``，
不暴露 SQLAlchemy / IntegrityError 以外的 infra 细节。Writer 负责捕获
IntegrityError 并走 ``find_by_confirmation_id`` / ``find_smart_approve_dedup``
幂等回读分支。
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import and_, delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.approval_grant import ApprovalDecision, ApprovalGrant
from app.domain.repositories.approval_grant_repository import ApprovalGrantRepository
from app.domain.services.approval_grant_policy import to_naive_utc
from app.infrastructure.models.tool_approval_grant import ToolApprovalGrantModel


def _model_to_domain(model: ToolApprovalGrantModel) -> ApprovalGrant:
    """ORM → domain 单向映射。"""
    return ApprovalGrant(
        decision_id=model.decision_id,
        user_id=model.user_id,
        session_id=model.session_id,
        tool_name=model.tool_name,
        tool_source=model.tool_source,
        arg_digest=model.arg_digest,
        primary_arg=model.primary_arg,
        dir_arg=model.dir_arg,
        scope=model.scope,
        effect=model.effect,
        source_type=model.source_type,
        confirmation_id=model.confirmation_id,
        expires_at=model.expires_at,
        created_at=model.created_at,
    )


class DBApprovalGrantRepository(ApprovalGrantRepository):
    """Postgres 实现。UNIQUE 冲突由 DB 抛 IntegrityError，Writer 负责处理。"""

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create(self, decision: ApprovalDecision) -> str:
        decision_id = str(uuid.uuid4())
        record = ToolApprovalGrantModel(
            decision_id=decision_id,
            user_id=decision.user_id,
            session_id=decision.session_id,
            tool_name=decision.tool_name,
            tool_source=decision.tool_source,
            arg_digest=decision.arg_digest,
            primary_arg=decision.primary_arg,
            dir_arg=decision.dir_arg,
            scope=decision.scope,
            effect=decision.effect,
            source_type=decision.source_type,
            confirmation_id=decision.confirmation_id,
            expires_at=to_naive_utc(decision.expires_at),
        )
        self.db_session.add(record)
        # flush 触发 UNIQUE 约束；IntegrityError 在此抛出、由 writer 捕获
        await self.db_session.flush()
        return decision_id

    async def find_by_confirmation_id(
        self, confirmation_id: str,
    ) -> Optional[ApprovalGrant]:
        stmt = select(ToolApprovalGrantModel).where(
            ToolApprovalGrantModel.confirmation_id == confirmation_id
        )
        result = await self.db_session.execute(stmt)
        row = result.scalar_one_or_none()
        return _model_to_domain(row) if row is not None else None

    async def find_active_grants(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
    ) -> list[ApprovalGrant]:
        """返回 (user, tool) 下未过期的 grants：全部 ``always`` + 指定 session 的 ``session`` 行。"""
        # DB 列是 TIMESTAMP WITHOUT TIME ZONE，用 naive UTC 对齐比较
        now = to_naive_utc(datetime.now(timezone.utc))
        # scope 过滤：always 一律返；session 要求 session_id 匹配（None 时仅查 always）
        scope_filter = ToolApprovalGrantModel.scope == "always"
        if session_id is not None:
            scope_filter = or_(
                scope_filter,
                and_(
                    ToolApprovalGrantModel.scope == "session",
                    ToolApprovalGrantModel.session_id == session_id,
                ),
            )
        stmt = select(ToolApprovalGrantModel).where(
            and_(
                ToolApprovalGrantModel.user_id == user_id,
                ToolApprovalGrantModel.tool_name == tool_name,
                scope_filter,
                or_(
                    ToolApprovalGrantModel.expires_at.is_(None),
                    ToolApprovalGrantModel.expires_at > now,
                ),
            )
        )
        result = await self.db_session.execute(stmt)
        return [_model_to_domain(row) for row in result.scalars().all()]

    async def find_smart_approve_dedup(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
        arg_digest: str,
        effect: str,
    ) -> Optional[ApprovalGrant]:
        """匹配 partial UNIQUE ``ux_tool_approval_grants_smart_approve_dedup`` 的读回。"""
        conds = [
            ToolApprovalGrantModel.user_id == user_id,
            ToolApprovalGrantModel.tool_name == tool_name,
            ToolApprovalGrantModel.arg_digest == arg_digest,
            ToolApprovalGrantModel.effect == effect,
            ToolApprovalGrantModel.confirmation_id.is_(None),
        ]
        if session_id is None:
            conds.append(ToolApprovalGrantModel.session_id.is_(None))
        else:
            conds.append(ToolApprovalGrantModel.session_id == session_id)
        stmt = select(ToolApprovalGrantModel).where(and_(*conds))
        result = await self.db_session.execute(stmt)
        row = result.scalar_one_or_none()
        return _model_to_domain(row) if row is not None else None

    async def delete(self, decision_id: str) -> None:
        await self.db_session.execute(
            delete(ToolApprovalGrantModel).where(
                ToolApprovalGrantModel.decision_id == decision_id
            )
        )
        await self.db_session.flush()
