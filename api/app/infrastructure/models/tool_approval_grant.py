"""R5 CS4: tool_approval_grants ORM model.

Phase 1 scope: 13 columns (no policy_id / revoked_at / reason_code /
source_actor_id / schema_version — those are Phase 2 PermissionEngine prework
and will be added via ALTER TABLE ADD COLUMN when needed).
"""

from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    PrimaryKeyConstraint,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class ToolApprovalGrantModel(Base):
    """R5 CS4 grants ORM.

    核心机制：
    - ``confirmation_id UNIQUE`` 作原子 single-flight claim（NULL 不参与 UNIQUE）
    - Partial UNIQUE ``ux_tool_approval_grants_smart_approve_dedup`` 覆盖
      SmartApprove (``confirmation_id IS NULL``) 路径的去重
    - ``effect='deny'`` 也走这张表（Reader Phase 1 不 surface，但审计证据保留）
    """

    __tablename__ = "tool_approval_grants"
    __table_args__ = (
        PrimaryKeyConstraint("decision_id", name="pk_tool_approval_grants"),
        UniqueConstraint(
            "confirmation_id",
            name="uq_tool_approval_grants_confirmation_id",
        ),
        CheckConstraint(
            "scope IN ('session','always')",
            name="ck_tool_approval_grants_scope",
        ),
        CheckConstraint(
            "effect IN ('approve','deny')",
            name="ck_tool_approval_grants_effect",
        ),
        CheckConstraint(
            "tool_source IN ('native','mcp','a2a','skill')",
            name="ck_tool_approval_grants_tool_source",
        ),
        CheckConstraint(
            "(scope='always' AND session_id IS NULL)"
            " OR (scope='session' AND session_id IS NOT NULL)",
            name="ck_tool_approval_grants_session_scope_has_session_id",
        ),
        CheckConstraint(
            "(scope='always' AND expires_at IS NULL)"
            " OR (scope='session' AND expires_at IS NOT NULL)",
            name="ck_tool_approval_grants_expires_at_only_for_session",
        ),
        Index(
            "ix_tool_approval_grants_active",
            "user_id",
            "tool_name",
        ),
        Index(
            "ix_tool_approval_grants_session",
            "session_id",
            "tool_name",
            postgresql_where=text("scope='session'"),
        ),
        Index(
            "ux_tool_approval_grants_smart_approve_dedup",
            "user_id",
            "session_id",
            "tool_name",
            "arg_digest",
            "effect",
            unique=True,
            postgresql_where=text("confirmation_id IS NULL"),
        ),
    )

    decision_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        primary_key=True,
    )
    user_id: Mapped[str] = mapped_column(
        String(255),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    tool_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    tool_source: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    arg_digest: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )
    primary_arg: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        server_default=text("''"),
    )
    dir_arg: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        server_default=text("''"),
    )
    scope: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    effect: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
    )
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    confirmation_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP(0)"),
    )
