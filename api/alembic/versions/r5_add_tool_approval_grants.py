"""R5 CS4: add tool_approval_grants + extend tool_approval_log.decision_id

Revision ID: r5_add_tool_approval_grants
Revises: m3_memory_system_notifications
Create Date: 2026-04-20 10:00:00.000000

DDL-only. Backfill lives in app.cli.backfill_approval_grants (manual, not
part of the alembic chain — see design doc §Distribution Plan step 3).

手工触发：``uv run python -m app.cli.backfill_approval_grants``
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "r5_add_tool_approval_grants"
down_revision: Union[str, Sequence[str], None] = "m3_memory_system_notifications"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_approval_grants",
        sa.Column("decision_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=True),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("tool_source", sa.String(16), nullable=False),
        sa.Column("arg_digest", sa.String(64), nullable=False),
        sa.Column(
            "primary_arg",
            sa.String(512),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "dir_arg",
            sa.String(512),
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column("scope", sa.String(16), nullable=False),
        sa.Column("effect", sa.String(16), nullable=False),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("confirmation_id", sa.String(255), nullable=True),
        sa.Column("expires_at", sa.DateTime, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("decision_id", name="pk_tool_approval_grants"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_tool_approval_grants_user_id",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "confirmation_id",
            name="uq_tool_approval_grants_confirmation_id",
        ),
        sa.CheckConstraint(
            "scope IN ('session','always')",
            name="ck_tool_approval_grants_scope",
        ),
        sa.CheckConstraint(
            "effect IN ('approve','deny')",
            name="ck_tool_approval_grants_effect",
        ),
        sa.CheckConstraint(
            "tool_source IN ('native','mcp','a2a','skill')",
            name="ck_tool_approval_grants_tool_source",
        ),
        sa.CheckConstraint(
            "(scope='always' AND session_id IS NULL)"
            " OR (scope='session' AND session_id IS NOT NULL)",
            name="ck_tool_approval_grants_session_scope_has_session_id",
        ),
        sa.CheckConstraint(
            "(scope='always' AND expires_at IS NULL)"
            " OR (scope='session' AND expires_at IS NOT NULL)",
            name="ck_tool_approval_grants_expires_at_only_for_session",
        ),
    )

    op.create_index(
        "ix_tool_approval_grants_active",
        "tool_approval_grants",
        ["user_id", "tool_name"],
    )

    op.create_index(
        "ix_tool_approval_grants_session",
        "tool_approval_grants",
        ["session_id", "tool_name"],
        postgresql_where=sa.text("scope='session'"),
    )

    # Partial UNIQUE: SmartApprove (confirmation_id IS NULL) dedup.
    # Main UNIQUE(confirmation_id) 不约束 NULL 路径（Postgres NULL 语义），
    # 这条 partial UNIQUE 覆盖 (user, session, tool, arg_digest, effect) 去重。
    op.create_index(
        "ux_tool_approval_grants_smart_approve_dedup",
        "tool_approval_grants",
        ["user_id", "session_id", "tool_name", "arg_digest", "effect"],
        unique=True,
        postgresql_where=sa.text("confirmation_id IS NULL"),
    )

    # Extend tool_approval_log with decision_id FK (symmetric rollback via delete_by_decision_id).
    op.add_column(
        "tool_approval_log",
        sa.Column("decision_id", sa.String(36), nullable=True),
    )
    op.create_foreign_key(
        "fk_tool_approval_log_decision_id",
        "tool_approval_log",
        "tool_approval_grants",
        ["decision_id"],
        ["decision_id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_tool_approval_log_decision_id",
        "tool_approval_log",
        ["decision_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tool_approval_log_decision_id", table_name="tool_approval_log")
    op.drop_constraint(
        "fk_tool_approval_log_decision_id",
        "tool_approval_log",
        type_="foreignkey",
    )
    op.drop_column("tool_approval_log", "decision_id")

    op.drop_index(
        "ux_tool_approval_grants_smart_approve_dedup",
        table_name="tool_approval_grants",
    )
    op.drop_index(
        "ix_tool_approval_grants_session",
        table_name="tool_approval_grants",
    )
    op.drop_index(
        "ix_tool_approval_grants_active",
        table_name="tool_approval_grants",
    )
    op.drop_table("tool_approval_grants")
