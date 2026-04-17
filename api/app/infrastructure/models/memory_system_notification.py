"""``memory_system_notifications`` ORM 模型。"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    PrimaryKeyConstraint,
    String,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class MemorySystemNotificationModel(Base):
    """post-flow 用户通知 ORM 模型。

    表结构见 migration ``m3_memory_system_notifications``。这里的 Index
    声明和 migration 保持一一对应，保证 ``Base.metadata.create_all``
    （测试路径）与 Alembic upgrade（生产路径）建出的库 schema 一致。
    """

    __tablename__ = "memory_system_notifications"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_memory_system_notifications"),
        Index(
            "ix_memory_system_notifications_user_unread",
            "user_id",
            "created_at",
            postgresql_where=text("read_at IS NULL"),
        ),
        Index(
            "ix_memory_system_notifications_expires_at",
            "expires_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(255), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
    read_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("now()"),
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("(now() + INTERVAL '30 days')"),
    )
