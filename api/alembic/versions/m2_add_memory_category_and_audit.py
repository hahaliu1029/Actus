"""add memory_chunks category/auto_promoted_at/fs_synced/pinned + CHECK constraints

Revision ID: m2_add_memory_category_and_audit
Revises: m1_add_memory_audit_log
Create Date: 2026-04-17 16:00:00.000000

Memory System Redesign M1 PR-1 — first-class type + filesystem backing + pin
semantics. Adds four new columns to ``memory_chunks`` with the following
invariants enforced at the database layer:

- ``category`` is **NULLABLE** on purpose. Historical rows written before M1
  stay at ``NULL`` forever; new rows (PR-2 onward) MUST set category explicitly
  to one of ``user | rule | fact``. Never backfill legacy rows — doing so
  would silently mislabel session_flush chunks as ``fact`` and poison future
  retrieval.
- ``pinned`` may only be true when ``category = 'user'``. Enforced via CHECK
  constraint so any new category or write path that forgets this rule fails
  at INSERT time rather than corrupts the ranker silently.
- ``source`` gains a CHECK constraint covering the M1-allowed values only.
  Legacy rows that predate this migration all carry ``session_flush``
  (the pre-M1 default), which passes the constraint as-is. ``file`` is
  deliberately excluded until M3 adds file-import support.
- ``fs_synced`` defaults to ``true`` for legacy rows (no file backing to
  reconcile, they pre-date the file-store) and defaults to ``false`` server-
  side for new rows so FsReconciler picks them up on first write failure.

A partial index on ``fs_synced = false`` backs the reconciler's boot scan so
it stays O(pending) rather than O(total). ``pinned`` gets a partial index
used by the M2 user_profile prompt section (pinned rows are always retained
first regardless of recency).
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "m2_add_memory_category_and_audit"
down_revision: Union[str, None] = "m1_add_memory_audit_log"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# CHECK constraint expressions — kept as constants so upgrade and downgrade
# reference the same SQL text without drift.
_SOURCE_ALLOWED = "source IN ('session_flush', 'manual', 'memory_save')"
_CATEGORY_ALLOWED = "category IS NULL OR category IN ('user', 'rule', 'fact')"
_PINNED_ONLY_FOR_USER = "pinned = false OR category = 'user'"


def upgrade() -> None:
    # 1. 新列：category (nullable) / auto_promoted_at (nullable) /
    #    fs_synced (not null, backfill 成 true 代表"无需同步") /
    #    pinned (not null, default false)
    #
    # fs_synced 的两步处理：
    #   (a) add_column(..., server_default=false) 会把 PG 默认策略应用到所有
    #       已有行——PostgreSQL 的 ALTER TABLE ADD COLUMN DEFAULT 会先把
    #       默认值物化到每条历史行上，所以在这一步后所有行 fs_synced=false；
    #   (b) 紧接着的 UPDATE 把这些历史行翻成 true：它们是 M1 前写入、没有
    #       对应文件的"遗留行"，FsReconciler 不应扫到它们。
    # 两步分开是因为 Alembic 的 add_column 里无法直接表达 "新行默认 false，
    # 旧行标 true" 的分叉语义。
    op.add_column(
        "memory_chunks",
        sa.Column("category", sa.String(16), nullable=True),
    )
    op.add_column(
        "memory_chunks",
        sa.Column("auto_promoted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "memory_chunks",
        sa.Column(
            "fs_synced",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "memory_chunks",
        sa.Column(
            "pinned",
            sa.Boolean,
            nullable=False,
            server_default=sa.text("false"),
        ),
    )

    # 2. 历史行标记为已同步——它们没有对应 file，reconciler 不应扫到。
    op.execute(
        "UPDATE memory_chunks SET fs_synced = true WHERE fs_synced = false"
    )

    # 3. CHECK constraints
    op.create_check_constraint(
        "ck_memory_chunks_source_allowed",
        "memory_chunks",
        _SOURCE_ALLOWED,
    )
    op.create_check_constraint(
        "ck_memory_chunks_category_allowed",
        "memory_chunks",
        _CATEGORY_ALLOWED,
    )
    op.create_check_constraint(
        "ck_memory_chunks_pinned_only_for_user",
        "memory_chunks",
        _PINNED_ONLY_FOR_USER,
    )

    # 4. 支持索引
    # 4a. FsReconciler 的 partial index——只扫 fs_synced=false 的行。
    #
    # 列顺序选 (updated_at, id) 而非 (user_id, updated_at)，理由：
    # - 启动全局 boot scan: ``WHERE fs_synced=false ORDER BY updated_at ASC,
    #   id ASC LIMIT N`` → 本索引的 leading columns 与 ORDER BY 一致，planner
    #   可直接索引顺序扫描 + LIMIT，真的做到 O(LIMIT) 而非 O(total)
    # - 懒式 per-user walk: ``WHERE fs_synced=false AND user_id=? ORDER BY
    #   updated_at ASC`` → 扫整个 partial index（仅 pending 行，数量小） +
    #   filter user_id，仍是 O(pending_total) 而非 O(total_memory_chunks)
    # 反过来把 user_id 放前导列会让 boot scan 退化到全表扫 + 排序。
    op.create_index(
        "ix_memory_chunks_fs_synced_pending",
        "memory_chunks",
        ["updated_at", "id"],
        postgresql_where=sa.text("fs_synced = false"),
    )
    # 4b. category filter 常用，带 user_id 复合索引
    op.create_index(
        "ix_memory_chunks_user_category",
        "memory_chunks",
        ["user_id", "category"],
    )
    # 4c. pinned user memory 的快查询（M2 user_profile section 要优先保留）
    op.create_index(
        "ix_memory_chunks_user_pinned",
        "memory_chunks",
        ["user_id"],
        postgresql_where=sa.text("pinned = true"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_chunks_user_pinned",
        table_name="memory_chunks",
    )
    op.drop_index(
        "ix_memory_chunks_user_category",
        table_name="memory_chunks",
    )
    op.drop_index(
        "ix_memory_chunks_fs_synced_pending",
        table_name="memory_chunks",
    )
    op.drop_constraint(
        "ck_memory_chunks_pinned_only_for_user",
        "memory_chunks",
        type_="check",
    )
    op.drop_constraint(
        "ck_memory_chunks_category_allowed",
        "memory_chunks",
        type_="check",
    )
    op.drop_constraint(
        "ck_memory_chunks_source_allowed",
        "memory_chunks",
        type_="check",
    )
    op.drop_column("memory_chunks", "pinned")
    op.drop_column("memory_chunks", "fs_synced")
    op.drop_column("memory_chunks", "auto_promoted_at")
    op.drop_column("memory_chunks", "category")
