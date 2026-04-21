"""用户工具审批偏好 ORM 模型。"""

import uuid
from datetime import datetime

from app.domain.models.user_tool_approval_policy import (
    ApprovalPolicy,
    UserToolApprovalPolicy,
)
from sqlalchemy import (
    DateTime,
    ForeignKey,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class UserToolApprovalPolicyModel(Base):
    """R6 §4.2 — 显式覆盖表；无 server_default on policy。"""

    __tablename__ = "user_tool_approval_policies"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="pk_user_tool_approval_policies_id"),
        UniqueConstraint(
            "user_id",
            "tool_name",
            name="uq_user_tool_approval_policies_user_tool",
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
    policy: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
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
    def from_domain(cls, policy: UserToolApprovalPolicy) -> "UserToolApprovalPolicyModel":
        return cls(
            id=policy.id,
            user_id=policy.user_id,
            tool_name=policy.tool_name,
            policy=policy.policy.value,
            created_at=policy.created_at,
            updated_at=policy.updated_at,
        )

    def to_domain(self) -> UserToolApprovalPolicy:
        return UserToolApprovalPolicy(
            id=self.id,
            user_id=self.user_id,
            tool_name=self.tool_name,
            policy=ApprovalPolicy(self.policy),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def update_from_domain(self, policy: UserToolApprovalPolicy) -> None:
        self.policy = policy.policy.value
        self.updated_at = datetime.now()
