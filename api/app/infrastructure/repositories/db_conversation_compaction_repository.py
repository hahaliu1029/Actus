"""DB impl of ConversationCompactionRepository using ON CONFLICT DO NOTHING.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.conversation_compaction import ConversationCompaction
from app.domain.repositories.conversation_compaction_repository import (
    ConversationCompactionRepository,
)
from app.infrastructure.models.conversation_compaction import (
    ConversationCompactionModel,
)


def _model_to_domain(m: ConversationCompactionModel) -> ConversationCompaction:
    return ConversationCompaction(
        id=m.id,
        compaction_id=m.compaction_id,
        session_id=m.session_id,
        summary=m.summary,
        summary_tokens=m.summary_tokens,
        first_visible_event_id=m.first_visible_event_id,
        last_visible_event_id=m.last_visible_event_id,
        pre_compact_checkpoint_id=m.pre_compact_checkpoint_id,
        operations=list(m.operations or []),
        parent_compaction_id=m.parent_compaction_id,
        tokens_before_total=m.tokens_before_total,
        tokens_after_total=m.tokens_after_total,
        messages_removed_total=m.messages_removed_total,
        created_at=m.created_at,
    )


class DBConversationCompactionRepository(ConversationCompactionRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    async def create_or_get(self, record: ConversationCompaction) -> ConversationCompaction:
        stmt = (
            pg_insert(ConversationCompactionModel)
            .values(
                id=record.id,
                compaction_id=record.compaction_id,
                session_id=record.session_id,
                summary=record.summary,
                summary_tokens=record.summary_tokens,
                first_visible_event_id=record.first_visible_event_id,
                last_visible_event_id=record.last_visible_event_id,
                pre_compact_checkpoint_id=record.pre_compact_checkpoint_id,
                operations=record.operations,
                parent_compaction_id=record.parent_compaction_id,
                tokens_before_total=record.tokens_before_total,
                tokens_after_total=record.tokens_after_total,
                messages_removed_total=record.messages_removed_total,
                created_at=record.created_at,
            )
            .on_conflict_do_nothing(index_elements=["compaction_id"])
            .returning(ConversationCompactionModel.id)
        )
        result = await self.db_session.execute(stmt)
        inserted_id = result.scalar_one_or_none()
        if inserted_id is not None:
            return record  # New insert, return as-is

        # Conflict — fetch existing
        existing_stmt = select(ConversationCompactionModel).where(
            ConversationCompactionModel.compaction_id == record.compaction_id
        )
        existing = (await self.db_session.execute(existing_stmt)).scalar_one()
        return _model_to_domain(existing)

    async def list_for_session(self, session_id: str) -> list[ConversationCompaction]:
        stmt = (
            select(ConversationCompactionModel)
            .where(ConversationCompactionModel.session_id == session_id)
            .order_by(ConversationCompactionModel.created_at.desc())
        )
        rows = (await self.db_session.execute(stmt)).scalars().all()
        return [_model_to_domain(m) for m in rows]

    async def get_by_id(
        self,
        session_id: str,
        compaction_id: str,
    ) -> ConversationCompaction | None:
        stmt = select(ConversationCompactionModel).where(
            ConversationCompactionModel.session_id == session_id,
            ConversationCompactionModel.compaction_id == compaction_id,
        )
        m = (await self.db_session.execute(stmt)).scalar_one_or_none()
        return _model_to_domain(m) if m is not None else None
