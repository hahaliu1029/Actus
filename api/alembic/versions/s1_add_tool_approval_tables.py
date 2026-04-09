"""add tool approval rules and log tables

Revision ID: s1_tool_approval
Revises: f1a2b3c4d5e6
Create Date: 2026-04-09 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "s1_tool_approval"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "tool_approval_rules",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("rule", sa.String(32), nullable=False),
        sa.Column("command_pattern", sa.String(512), nullable=False),
        sa.Column("dir_pattern", sa.String(512), nullable=False, server_default=sa.text("''")),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.Column("updated_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.PrimaryKeyConstraint("id", name="pk_tool_approval_rules_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_tool_approval_rules_user_id", ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "tool_name", "command_pattern", "dir_pattern", name="uq_tool_approval_rules_user_tool_pattern"),
    )
    op.create_index("ix_tool_approval_rules_user_id", "tool_approval_rules", ["user_id"])

    op.create_table(
        "tool_approval_log",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("tool_name", sa.String(255), nullable=False),
        sa.Column("tool_args", sa.JSON, nullable=True),
        sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("scope", sa.String(32), nullable=False),
        sa.Column("approved_by", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime, nullable=False, server_default=sa.text("CURRENT_TIMESTAMP(0)")),
        sa.PrimaryKeyConstraint("id", name="pk_tool_approval_log_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_tool_approval_log_user_id", ondelete="CASCADE"),
    )
    op.create_index("ix_tool_approval_log_user_id", "tool_approval_log", ["user_id"])
    op.create_index("ix_tool_approval_log_session_id", "tool_approval_log", ["session_id"])


def downgrade() -> None:
    op.drop_table("tool_approval_log")
    op.drop_table("tool_approval_rules")
