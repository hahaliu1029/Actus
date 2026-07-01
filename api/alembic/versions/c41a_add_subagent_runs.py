"""C4.1a: add subagent_runs observation table.

Revision ID: c41a_add_subagent_runs
Revises: s3pr1_add_session_depth_lineage
Create Date: 2026-07-01

Spec ref: docs/superpowers/specs/2026-07-01-c4.1a-subagent-run-record-design.md §4.

观测面表：每行 = 一次 subagent run 的投影结果（coordinator child / research
child / 未来 REMOTE）。列扁平化自 SubagentRunResult；JSONB artifacts。无 FK
（observation + REMOTE-ready）。幂等键 UNIQUE(child_session_id)（全局唯一；
Postgres UNIQUE 容多 NULL → REMOTE 的 NULL child 不 dedup，由 C4.1b 定义）。

Downgrade（DESTRUCTIVE）：drop index + table。观测数据丢失可接受（纯观测底座，
不参与执行/恢复）。
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c41a_add_subagent_runs"
down_revision = "s3pr1_add_session_depth_lineage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "subagent_runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("runtime", sa.String(length=16), nullable=False),
        sa.Column("lifecycle_state", sa.String(length=32), nullable=False),
        sa.Column("terminal_outcome", sa.String(length=32), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column("parent_session_id", sa.String(length=255), nullable=False),
        sa.Column("child_session_id", sa.String(length=255), nullable=True),
        sa.Column("source_ref", sa.String(length=255), nullable=True),
        sa.Column(
            "cost_authoritative", sa.Boolean(),
            nullable=False, server_default=sa.text("false"),
        ),
        sa.Column("cost_total_input_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_total_output_tokens", sa.Integer(), nullable=True),
        sa.Column("cost_total_usd", sa.Float(), nullable=True),
        sa.Column("cost_tool_call_count", sa.Integer(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("duration_source", sa.String(length=32), nullable=False),
        sa.Column(
            "artifacts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False, server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "created_at", sa.TIMESTAMP(timezone=True),
            nullable=False, server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "child_session_id", name="uq_subagent_runs_child_session_id",
        ),
    )
    op.create_index(
        "ix_subagent_runs_parent_session_id", "subagent_runs", ["parent_session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_subagent_runs_parent_session_id", table_name="subagent_runs",
    )
    op.drop_table("subagent_runs")
