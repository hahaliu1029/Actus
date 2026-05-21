"""ORM mapping for mailbox_envelope_audit table (C3 spec §5.8)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import DateTime, Index, Integer, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.infrastructure.models.base import Base


class MailboxEnvelopeAuditModel(Base):
    """Mailbox envelope audit row — consumer-side terminal-dedup authority.

    Spec §5.8 Layer 2 — PK (parent_session_id, envelope_id) provides
    consumer-side idempotency; the dedup question becomes a single PK
    lookup. ``processed_at`` distinguishes inflight vs final-acked rows.
    """

    __tablename__ = "mailbox_envelope_audit"
    # C3 PR-1 (codex P2): mirror the indexes declared in the alembic migration
    # ``c3_add_mailbox_envelope_audit`` so they're present on
    # ``Base.metadata`` for autogenerate runs and create_all-based test
    # fixtures. Without these on the ORM, future autogenerate diffs would
    # propose dropping them.
    __table_args__ = (
        Index(
            "ix_mailbox_envelope_audit_child",
            "child_session_id",
        ),
        Index(
            "ix_mailbox_envelope_audit_unprocessed",
            "parent_session_id",
            "received_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )

    # C3 PR-1 (codex round 15 P2): mirror the migration widening of
    # ``parent_session_id`` and ``child_session_id`` to ``String(255)`` so
    # they match ``sessions.id`` (api/app/infrastructure/models/session.py
    # :55-60). ``envelope_id`` stays at 64 — ULIDs are 26 chars.
    parent_session_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    envelope_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    child_session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)
    producer_role: Mapped[str] = mapped_column(String(32), nullable=False)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )
    processing_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    processed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    audit_payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    reclaim_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    last_error: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
