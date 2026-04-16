"""记忆管理审计日志 ORM 模型（仅写入，用于审计）

参照 ToolApprovalLogModel 风格。
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    PrimaryKeyConstraint,
    String,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MemoryAuditLogModel(Base):
    """记忆管理审计日志 ORM 模型（审计只写）"""

    __tablename__ = "memory_audit_log"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_memory_audit_log_id"),
        Index("ix_memory_audit_log_user_id", "user_id"),
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
    )
    chunk_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        comment="单条操作的目标 chunk；bulk/delete_all 场景为 None，改用 chunk_ids",
    )
    chunk_ids: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True,
        comment="bulk_delete 时记录实际被删除的 chunk id 列表（仅本用户拥有的）",
    )
    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    old_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )
    new_snapshot: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )
    affected_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        # now() 微秒精度，与 memory_chunks 保持一致，确保并发写入可稳定排序
        server_default=text("now()"),
    )
