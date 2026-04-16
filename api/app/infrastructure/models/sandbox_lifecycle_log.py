"""Sandbox 生命周期审计日志 ORM 模型（仅写入，用于审计）

PR2 §10.5: tool_approval_log 风格的审计表。
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class SandboxLifecycleLogModel(Base):
    """Sandbox 生命周期审计日志 ORM 模型（审计只写）"""

    __tablename__ = "sandbox_lifecycle_log"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_sandbox_lifecycle_log_id"),
        Index("ix_sandbox_lifecycle_log_session_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    old_state: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    new_state: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    generation: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    sandbox_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    reason: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    triggered_by: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
