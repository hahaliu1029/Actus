"""B4 M0 Phase C: cost_records ORM table.

Mirrors the DDL produced by migration ``b4m0_add_cost_records``. The CHECK
constraint on ``cost_status`` is declared in ``__table_args__`` so that
``Base.metadata.create_all()`` (used by integration tests) produces the same
shape as a migrated DB.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


_COST_STATUS_ALLOWED = (
    "cost_status IN ('actual', 'estimated', 'partial', 'unknown')"
)


class CostRecordModel(Base):
    """SQLAlchemy ORM model for cost_records."""

    __tablename__ = "cost_records"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_cost_records"),
        CheckConstraint(
            _COST_STATUS_ALLOWED,
            name="ck_cost_records_cost_status_allowed",
        ),
        Index(
            "uq_cost_records_run_id",
            "run_id",
            unique=True,
        ),
        Index("ix_cost_records_session_id", "session_id"),
        Index("ix_cost_records_user_id", "user_id"),
    )

    id: Mapped[str] = mapped_column(
        String(64),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    node_name: Mapped[str] = mapped_column(String(128), nullable=False)
    step_ix: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    attempt_ix: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)

    input_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    output_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_read_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    cache_write_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    reasoning_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    total_usd: Mapped[Decimal] = mapped_column(
        Numeric(28, 10), nullable=False, server_default=text("0")
    )
    pricing_version: Mapped[str] = mapped_column(String(32), nullable=False)
    cost_status: Mapped[str] = mapped_column(String(16), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
