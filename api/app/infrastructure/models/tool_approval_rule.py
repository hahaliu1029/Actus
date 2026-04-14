"""工具审批规则 ORM 模型"""

import uuid
from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.domain.models.tool_approval_rule import ToolApprovalRule
from .base import Base


class ToolApprovalRuleModel(Base):
    """工具审批规则 ORM 模型"""

    __tablename__ = "tool_approval_rules"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_tool_approval_rules_id"),
        UniqueConstraint(
            "user_id",
            "tool_name",
            "command_pattern",
            "dir_pattern",
            name="uq_tool_approval_rules_user_tool_cmd_dir",
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
    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    rule: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    command_pattern: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    dir_pattern: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        server_default=text("''"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        onupdate=datetime.now,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )

    @classmethod
    def from_domain(cls, rule: ToolApprovalRule) -> "ToolApprovalRuleModel":
        """从领域模型创建 ORM 模型"""
        return cls(
            id=rule.id,
            user_id=rule.user_id,
            tool_name=rule.tool_name,
            rule=rule.rule,
            command_pattern=rule.command_pattern,
            dir_pattern=rule.dir_pattern,
            created_at=rule.created_at,
            updated_at=rule.updated_at,
        )

    def to_domain(self) -> ToolApprovalRule:
        """将 ORM 模型转换为领域模型"""
        return ToolApprovalRule(
            id=self.id,
            user_id=self.user_id,
            tool_name=self.tool_name,
            rule=self.rule,
            command_pattern=self.command_pattern,
            dir_pattern=self.dir_pattern,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )
