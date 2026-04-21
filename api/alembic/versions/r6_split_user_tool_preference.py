"""split UserToolPreference: rename old table + create ApprovalPolicy table

Revision ID: r6_user_tool_split
Revises: r5_add_tool_approval_grants
Create Date: 2026-04-21 00:00:00.000000

R6 §7: rename `user_tool_preferences` → `user_tool_enablements`
(metadata-only ALTER TABLE) + create `user_tool_approval_policies`.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "r6_user_tool_split"
down_revision: Union[str, Sequence[str], None] = "r5_add_tool_approval_grants"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. rename 物理表
    op.rename_table("user_tool_preferences", "user_tool_enablements")

    # 2. rename 全部 named constraint / index（PK + UNIQUE + FK + INDEX）
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT pk_user_tool_preferences_id "
        "TO pk_user_tool_enablements_id"
    )
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT uq_user_tool_preferences_user_tool "
        "TO uq_user_tool_enablements_user_tool"
    )
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT fk_user_tool_preferences_user_id_users "
        "TO fk_user_tool_enablements_user_id_users"
    )
    op.execute(
        "ALTER INDEX ix_user_tool_preferences_user_id "
        "RENAME TO ix_user_tool_enablements_user_id"
    )

    # 3. 新建 user_tool_approval_policies
    op.create_table(
        "user_tool_approval_policies",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("policy", sa.String(16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime,
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_user_tool_approval_policies_id"),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_user_tool_approval_policies_user_id_users",
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "user_id",
            "tool_name",
            name="uq_user_tool_approval_policies_user_tool",
        ),
    )
    op.create_index(
        "ix_user_tool_approval_policies_user_id",
        "user_tool_approval_policies",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_tool_approval_policies_user_id",
        table_name="user_tool_approval_policies",
    )
    op.drop_table("user_tool_approval_policies")
    op.execute(
        "ALTER INDEX ix_user_tool_enablements_user_id "
        "RENAME TO ix_user_tool_preferences_user_id"
    )
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT fk_user_tool_enablements_user_id_users "
        "TO fk_user_tool_preferences_user_id_users"
    )
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT uq_user_tool_enablements_user_tool "
        "TO uq_user_tool_preferences_user_tool"
    )
    op.execute(
        "ALTER TABLE user_tool_enablements "
        "RENAME CONSTRAINT pk_user_tool_enablements_id "
        "TO pk_user_tool_preferences_id"
    )
    op.rename_table("user_tool_enablements", "user_tool_preferences")
