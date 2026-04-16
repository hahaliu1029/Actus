"""add sandbox_lifecycle_log audit table

Revision ID: s3_sandbox_lifecycle_log
Revises: s2_sandbox_binding
Create Date: 2026-04-16 14:00:00.000000

PR2 §10.5: K8s-style terminal-immutable audit trail for sandbox binding
state transitions. Mirrors tool_approval_log pattern.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "s3_sandbox_lifecycle_log"
down_revision: Union[str, None] = "s2_sandbox_binding"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sandbox_lifecycle_log",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(255), nullable=False),
        sa.Column("old_state", sa.String(32), nullable=False),
        sa.Column("new_state", sa.String(32), nullable=False),
        sa.Column("generation", sa.Integer, nullable=False),
        sa.Column("sandbox_id", sa.String(255), nullable=True),
        sa.Column("reason", sa.String(64), nullable=True),
        sa.Column("triggered_by", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_sandbox_lifecycle_log_id"),
    )
    op.create_index(
        "ix_sandbox_lifecycle_log_session_id",
        "sandbox_lifecycle_log",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_sandbox_lifecycle_log_session_id",
        table_name="sandbox_lifecycle_log",
    )
    op.drop_table("sandbox_lifecycle_log")
