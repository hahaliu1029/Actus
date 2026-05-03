"""SQLAlchemy ORM model for conversation_compactions table.

Spec: docs/superpowers/specs/2026-05-02-b6-compaction-metadata-persistence-design.md
"""
from __future__ import annotations

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

from app.infrastructure.models.base import Base


class ConversationCompactionModel(Base):
    __tablename__ = "conversation_compactions"

    id = Column(UUID(as_uuid=True), primary_key=True)
    compaction_id = Column(String(16), nullable=False, unique=True)
    session_id = Column(
        String(255),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    summary = Column(Text(), nullable=False)
    summary_tokens = Column(Integer(), nullable=False)
    first_visible_event_id = Column(String(64), nullable=True)
    last_visible_event_id = Column(String(64), nullable=True)
    pre_compact_checkpoint_id = Column(String(255), nullable=True)
    operations = Column(JSONB(), nullable=False)
    parent_compaction_id = Column(String(16), nullable=True)
    tokens_before_total = Column(Integer(), nullable=False)
    tokens_after_total = Column(Integer(), nullable=False)
    messages_removed_total = Column(Integer(), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    __table_args__ = (
        CheckConstraint(
            "tokens_after_total <= tokens_before_total",
            name="ck_compaction_tokens_monotonic",
        ),
        CheckConstraint(
            "jsonb_array_length(operations) >= 1",
            name="ck_compaction_operations_nonempty",
        ),
        Index(
            "ix_compaction_session_created",
            "session_id",
            text("created_at DESC"),
        ),
    )
