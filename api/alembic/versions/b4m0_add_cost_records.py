"""B4 M0: add cost_records table for per-session LLM cost ledger.

Revision ID: b4m0_cost_records
Revises: r6_user_tool_split
Create Date: 2026-04-24 14:00:00.000000

One row per completed LLM call. Indexed on session_id for GET /cost path,
unique on run_id so retries cannot double-bill, CHECK on cost_status so
invalid statuses are rejected at INSERT time (design Issue 2C).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b4m0_cost_records"
down_revision: Union[str, Sequence[str], None] = "r6_user_tool_split"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_COST_STATUS_ALLOWED = (
    "cost_status IN ('actual', 'estimated', 'partial', 'unknown')"
)


def upgrade() -> None:
    op.create_table(
        "cost_records",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(255),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("node_name", sa.String(128), nullable=False),
        sa.Column(
            "step_ix",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "attempt_ix",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column(
            "input_tokens",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "output_tokens",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cache_read_tokens",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "cache_write_tokens",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "reasoning_tokens",
            sa.Integer,
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "total_usd",
            sa.Numeric(28, 10),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("pricing_version", sa.String(32), nullable=False),
        sa.Column("cost_status", sa.String(16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            _COST_STATUS_ALLOWED,
            name="ck_cost_records_cost_status_allowed",
        ),
    )
    op.create_index(
        "uq_cost_records_run_id",
        "cost_records",
        ["run_id"],
        unique=True,
    )
    op.create_index(
        "ix_cost_records_session_id",
        "cost_records",
        ["session_id"],
    )
    op.create_index(
        "ix_cost_records_user_id",
        "cost_records",
        ["user_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_cost_records_user_id", table_name="cost_records")
    op.drop_index("ix_cost_records_session_id", table_name="cost_records")
    op.drop_index("uq_cost_records_run_id", table_name="cost_records")
    op.drop_table("cost_records")
