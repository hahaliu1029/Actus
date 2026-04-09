"""工具审批日志 ORM 模型（仅写入，用于审计）"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    JSON,
    PrimaryKeyConstraint,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ToolApprovalLogModel(Base):
    """工具审批日志 ORM 模型（审计只写）"""

    __tablename__ = "tool_approval_log"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_tool_approval_log_id"),
        Index("ix_tool_approval_log_session_id", "session_id"),
    )

    id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    tool_args: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )
    risk_level: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    scope: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    approved_by: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
