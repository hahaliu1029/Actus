"""add memory_audit_log table + composite indexes

Revision ID: m1_add_memory_audit_log
Revises: r3_skill_risk_columns
Create Date: 2026-04-16 12:00:00.000000

Memory management audit trail. Write-only table, mirrors tool_approval_log pattern.
Also adds composite indexes to memory_chunks and memory_audit_log to support
the /v2/memories list endpoint (updated_at DESC pagination) and future audit
history queries (created_at DESC).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "m1_add_memory_audit_log"
down_revision: Union[str, None] = "r3_skill_risk_columns"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_audit_log",
        sa.Column("id", sa.String(255), nullable=False),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_id", sa.String(255), nullable=True),
        sa.Column("chunk_ids", sa.JSON, nullable=True),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("old_snapshot", sa.JSON, nullable=True),
        sa.Column("new_snapshot", sa.JSON, nullable=True),
        sa.Column("affected_count", sa.Integer, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            # 使用 now()（微秒精度），与 memory_chunks 保持一致；
            # CURRENT_TIMESTAMP(0) 只有秒级精度，并发写入无法按时间稳定排序
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_memory_audit_log_id"),
    )
    op.create_index(
        "ix_memory_audit_log_user_id",
        "memory_audit_log",
        ["user_id"],
    )
    # 复合索引：支持按用户翻查审计历史（created_at DESC）
    op.create_index(
        "ix_memory_audit_log_user_created_at",
        "memory_audit_log",
        ["user_id", sa.text("created_at DESC")],
    )
    # 复合索引：/v2/memories 列表分页按 (updated_at DESC, id DESC) 排序。
    # id DESC 是 updated_at 并列时的确定性 tie-breaker（MemoryFlushService 给同一
    # batch 复用同一 now()，并列非常常见）；把 id 也建进索引以保持 ORDER BY 走索引顺序，
    # 避免深翻页时退化成全扫 + 内存排序。
    op.create_index(
        "ix_memory_chunks_user_updated_at",
        "memory_chunks",
        ["user_id", sa.text("updated_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_chunks_user_updated_at",
        table_name="memory_chunks",
    )
    op.drop_index(
        "ix_memory_audit_log_user_created_at",
        table_name="memory_audit_log",
    )
    op.drop_index(
        "ix_memory_audit_log_user_id",
        table_name="memory_audit_log",
    )
    op.drop_table("memory_audit_log")
