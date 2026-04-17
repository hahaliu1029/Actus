"""add memory_system_notifications table for post-flow user notifications

Revision ID: m3_memory_system_notifications
Revises: m2_add_memory_category_and_audit
Create Date: 2026-04-18 00:00:00.000000

Memory System Redesign M1 PR-4+8 — background-flush 完成后用户态通知。
gate circuit breaker OPEN / quota 超限 / fs 永久失败等事件在 task 已结束
（live SSE sink 已释放）时发生，需要持久化到 DB 由前端 polling `/unread`
消费。因为是 post-flow 纪实，而不是 session-scoped 事件，也不和
``memory_audit_log`` 合并——audit log 是 "做了什么" 的幂等写入记录，
notification 是 "要告诉用户什么" 的可读状态机（有 read_at 生命周期）。

设计取舍：
- ``expires_at`` 默认 ``NOW() + 30d``，上层 purge 任务按列扫；比 TTL 表
  更可控（hot table 维护成本低）。老行保留 30d 足够用户打开 app
  看到过去的通知，再长就是噪声。
- Partial index 只覆盖 ``read_at IS NULL`` 的行——未读 list 是 polling
  每 30s 打的热路径（设计 §196 "前端每 30 秒轮询 /unread 返未读数"），
  索引小 + planner 一眼选中；已读行只在 detail drawer 打开时才全量取。
- ``user_id`` 走 ``users.id`` FK + ``ondelete="CASCADE"``，和
  ``memory_chunks`` / ``memory_audit_log`` 保持一致：用户删号时连带
  清掉所有通知，避免孤儿行堆积。
- 不加 ``event_type`` enum CHECK constraint——PR-4+8 只用 gate_paused /
  quota_exceeded / fs_permanent_failure 三类，但后续 milestone（M4
  trust_score feedback）可能新增 event_type；单点枚举约束会让每次加
  事件类型都要发 migration，代价不匹配收益。运行时由 Pydantic schema
  做白名单校验。
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "m3_memory_system_notifications"
down_revision: Union[str, None] = "m2_add_memory_category_and_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "memory_system_notifications",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column(
            "user_id",
            sa.String(255),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "read_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("(now() + INTERVAL '30 days')"),
        ),
    )
    # Polling-heat path: unread-only list for a user, newest first.
    # Partial index keeps size proportional to unread backlog, not total.
    op.create_index(
        "ix_memory_system_notifications_user_unread",
        "memory_system_notifications",
        ["user_id", "created_at"],
        postgresql_where=sa.text("read_at IS NULL"),
    )
    # Separate btree for expiry cleanup batch jobs. Kept outside partial
    # index so it covers all rows (including already-read ones past their
    # TTL).
    op.create_index(
        "ix_memory_system_notifications_expires_at",
        "memory_system_notifications",
        ["expires_at"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_system_notifications_expires_at",
        table_name="memory_system_notifications",
    )
    op.drop_index(
        "ix_memory_system_notifications_user_unread",
        table_name="memory_system_notifications",
    )
    op.drop_table("memory_system_notifications")
