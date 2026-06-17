"""C2 PR-5: add coordinator_apply_audit table.

Revision ID: c2pr5_apply_audit
Revises: c2pr1_coordinator_columns
Create Date: 2026-05-25

Spec ref: §12.4 P0-8 (audit table for PatchApplier) + r6 (partial unique
success index).

The table records every coordinator PatchApplier invocation — one row per
``coordinator_run_id`` apply attempt. The two status-flavor columns are
distinct (codex R1 P1#4 clarification):

**``status`` column** (lifecycle + terminal apply outcome). 32-char
VARCHAR with no CHECK constraint — applier writes ``ApplyStatus.value``:

- ``in_progress`` — applier opened the audit row, snapshot/preflight
  in flight. There can be at most one in_progress row per (run_id) at
  any time (Redis lock enforces process-level, NOT table-level).
- ``success`` — all entries applied + snapshots discarded. The
  ``ux_apply_audit_run_success`` partial unique index pins ≤1 success
  row per run_id forever (idempotency guard for crash-recovery in
  PR-7).
- ``digest_drift`` — preflight saw modify/delete base_digest mismatch
  vs. sandbox compute_digest.
- ``file_missing`` — preflight modify/delete on a non-existent path.
- ``file_exists`` — preflight add on an already-existing path.
- ``target_special_file`` — preflight modify/delete found the target is a
  direct special inode (FIFO/socket/block/char) via ``check_path`` before
  the read; rejected without snapshot/read/write (S1b 2a). No schema change
  (free-text String(32); 19 chars).
- ``post_write_digest_mismatch`` — post-write verify re-read the
  sandbox file and got a digest != ``new_digest`` (sandbox-side
  truncation / corruption).
- ``minio_fetch_failed`` — minio_client.get_bytes raised mid-apply.
- ``write_io_error`` — sandbox write raised mid-apply (OSError or
  similar).
- ``apply_aborted`` — cancel_event fired before/during apply; rollback
  may have run if any writes had already happened.
- ``rollback_partial`` — apply failed AND the subsequent rollback
  could not restore every snapshot/added-file; HealthEvent emitted;
  operator manual recovery required. This is a terminal override of
  the original failure cause; the original cause is preserved in
  ``failed_at_path`` + ``failed_reason``.

**``plan_hash`` column** [codex R4 P1] — SHA-256 hex of the
canonical PatchApplyPlan JSON, captured at ``insert_in_progress``.
Used by PR-7 rehydrate to detect "this retry is replaying the same
plan that already partially applied" (idempotency key). Nullable in
PR-5 for backward compat; PR-7 follow-up tightens to NOT NULL.

**``rollback_status`` column** (orthogonal to ``status``; NULL when
no rollback was attempted):

- ``complete`` — every snapshot/added-file in ``applied`` was undone
  successfully.
- ``partial`` — at least one undo failed. The applier overrides
  ``status`` to ``rollback_partial`` and emits a HealthEvent.

Two indexes:

- ``ix_apply_audit_run`` — non-unique lookup ``(coordinator_run_id)``
  for "show all apply attempts for this run". Used by the PR-7 rehydrate
  path to find the latest attempt + by the SSE replay path.
- ``ux_apply_audit_run_success`` — UNIQUE
  ``(coordinator_run_id) WHERE status = 'success'``. Guarantees that
  even if a buggy applier replays an apply (e.g. after pod crash, before
  PR-7 dedup catches it), the DB rejects a second ``status='success'``
  row. Failed / in_progress / rollback_* rows can repeat freely.

**Downgrade contract (DESTRUCTIVE):** ``downgrade()`` drops the table
unconditionally. Any historical audit data is lost on rollback — that's
acceptable because the table is operational (debugging post-incident,
not source-of-truth for live state).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c2pr5_apply_audit"
down_revision = "c2pr1_coordinator_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "coordinator_apply_audit",
        sa.Column(
            "id", sa.BigInteger,
            primary_key=True, autoincrement=True,
        ),
        sa.Column(
            "coordinator_run_id", sa.String(length=320),
            nullable=False,
        ),
        sa.Column(
            "parent_session_id", sa.String(length=255),
            nullable=False,
        ),
        # [codex R4 P1 — PR-7 idempotency] SHA-256 hex (64 chars) over
        # the canonical-JSON form of the PatchApplyPlan. PR-7 rehydrate
        # uses this to detect "is this retry the same plan that already
        # ran?" without re-comparing every FilePatchEntry. Nullable for
        # PR-5 backward compat (early rows from local dev may not have
        # it); PR-7 will tighten to NOT NULL in a follow-up migration
        # once historical rows are backfilled or accepted as lossy.
        sa.Column("plan_hash", sa.String(length=64), nullable=True),
        # 32 chars covers every current value (see module docstring for
        # the full taxonomy — in_progress + 9 ApplyStatus.value entries
        # with the longest being post_write_digest_mismatch at 26 chars).
        # Future ApplyStatus additions under 32 chars don't need a widen.
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "file_count", sa.Integer,
            nullable=False, server_default="0",
        ),
        sa.Column(
            "total_bytes", sa.BigInteger,
            nullable=False, server_default="0",
        ),
        # Path length aligns with the path validator's spec ceiling (2048
        # chars accommodates deep monorepo layouts).
        sa.Column(
            "failed_at_path", sa.String(length=2048), nullable=True,
        ),
        sa.Column(
            "failed_reason", sa.String(length=256), nullable=True,
        ),
        sa.Column(
            "rollback_status", sa.String(length=32), nullable=True,
        ),
        sa.Column(
            "rollback_failed_paths",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "started_at", sa.TIMESTAMP(timezone=True), nullable=False,
        ),
        sa.Column(
            "finished_at", sa.TIMESTAMP(timezone=True), nullable=True,
        ),
        sa.Column("duration_ms", sa.Integer, nullable=True),
        sa.Column(
            "applied_files",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "plan_files_preview",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "diagnostics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.create_index(
        "ix_apply_audit_run", "coordinator_apply_audit",
        ["coordinator_run_id"],
    )
    op.create_index(
        "ux_apply_audit_run_success", "coordinator_apply_audit",
        ["coordinator_run_id"],
        unique=True,
        postgresql_where=sa.text("status = 'success'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ux_apply_audit_run_success", table_name="coordinator_apply_audit",
    )
    op.drop_index(
        "ix_apply_audit_run", table_name="coordinator_apply_audit",
    )
    op.drop_table("coordinator_apply_audit")
