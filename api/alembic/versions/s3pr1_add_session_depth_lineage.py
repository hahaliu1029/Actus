"""C2-full S3 PR-1: add session depth + root_session_id lineage columns.

Revision ID: s3pr1_add_session_depth_lineage  (31 chars — fits alembic_version varchar(32))
Revises: c2b1_mailbox_running_child_idx
Create Date: 2026-06-23

Persisted lineage backbone (design §4.1):
- depth INTEGER NOT NULL DEFAULT 0 — root=0, child=parent.depth+1. The constant
  server-default makes this a metadata-only ADD COLUMN on PG17 (no table rewrite).
- root_session_id VARCHAR(255) NULL — NULL ⇔ self-is-root; else the true tree
  root id. VARCHAR(255) matches sessions.id / parent_session_id width so the ORM
  mirror (String(255)) and alembic --autogenerate agree (no drift).

Backfill (every existing row is depth≤1 — see F0.8): roots
(parent_session_id IS NULL) keep (0, NULL) via the column default; subagents get
depth=1, root_session_id=parent_session_id. Scoping the UPDATE to
parent_session_id IS NOT NULL locks subagent rows only; idempotent / re-runnable.

NOT in PR-1 (design §4.1 steps 4-6):
- No index on root_session_id — no in-scope reader issues WHERE root_session_id=…;
  deferred to S3-enable so PR-1 pays no index-build lock.
- No FK on root_session_id — denormalized cache; the parent_session_id FK already
  enforces lineage integrity.
- No depth CHECK — the app cap is authoritative.

CI integration test asserts the post-backfill invariant (depth=0) ⇔
(parent_session_id IS NULL). Downgrade drops both columns (additive; the
parent_session_id FK remains the lineage source of truth).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "s3pr1_add_session_depth_lineage"
down_revision = "c2b1_mailbox_running_child_idx"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("depth", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "sessions",
        sa.Column("root_session_id", sa.String(length=255), nullable=True),
    )
    # Backfill subagent rows only — roots are already correct via the default.
    op.execute(
        "UPDATE sessions "
        "SET depth = 1, root_session_id = parent_session_id "
        "WHERE parent_session_id IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_column("sessions", "root_session_id")
    op.drop_column("sessions", "depth")
