"""用户工具扩展启用 ORM 模型。"""

import uuid
from datetime import datetime

from app.domain.models.user_tool_enablement import ToolType, UserToolEnablement
from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class UserToolEnablementModel(Base):
    """用户工具扩展启用数据 ORM 模型。"""

    __tablename__ = "user_tool_enablements"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_user_tool_enablements_id"),
        UniqueConstraint(
            "user_id", "tool_type", "tool_id", name="uq_user_tool_enablements_user_tool"
        ),
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
    tool_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    tool_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default=text("true"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, enablement: UserToolEnablement) -> "UserToolEnablementModel":
        """从领域模型创建 ORM 模型。"""
        return cls(
            id=enablement.id,
            user_id=enablement.user_id,
            tool_type=enablement.tool_type.value,
            tool_id=enablement.tool_id,
            enabled=enablement.enabled,
            created_at=enablement.created_at,
            updated_at=enablement.updated_at,
        )

    def to_domain(self) -> UserToolEnablement:
        """将 ORM 模型转换为领域模型。"""
        return UserToolEnablement(
            id=self.id,
            user_id=self.user_id,
            tool_type=ToolType(self.tool_type),
            tool_id=self.tool_id,
            enabled=self.enabled,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, enablement: UserToolEnablement) -> None:
        """从领域模型更新数据。"""
        self.enabled = enablement.enabled
        self.updated_at = datetime.now()
