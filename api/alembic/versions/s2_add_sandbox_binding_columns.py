"""add sandbox binding lifecycle columns to sessions

Revision ID: s2_sandbox_binding
Revises: s1_tool_approval
Create Date: 2026-04-16 00:00:00.000000

Migration strategy per §9.4 of sandbox-lifecycle-design spec:
- PENDING + sandbox_id NULL → UNBOUND
- PENDING + sandbox_id NOT NULL → DESTROYED (destroyed_at = updated_at)
- RUNNING/WAITING/TAKEOVER/TAKEOVER_PENDING/FINISHING + sandbox_id NOT NULL → ACTIVE
- COMPLETED/TIMED_OUT + sandbox_id NOT NULL → SUSPENDED (§16 decision: allow reopen)
- COMPLETED/TIMED_OUT + sandbox_id NULL → DESTROYED (destroyed_at = completed_at or updated_at)
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers
revision: str = "s2_sandbox_binding"
down_revision: Union[str, Sequence[str], None] = "s1_tool_approval"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Add new columns with defaults
    op.add_column(
        "sessions",
        sa.Column(
            "sandbox_state",
            sa.String(32),
            nullable=False,
            server_default=sa.text("'unbound'::character varying"),
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "sandbox_generation",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "sessions",
        sa.Column("sandbox_created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("sandbox_destroyed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("sandbox_destroy_reason", sa.String(64), nullable=True),
    )

    # Step 2: Backfill existing rows based on session status + sandbox_id

    # RUNNING / WAITING / TAKEOVER / TAKEOVER_PENDING / FINISHING with sandbox → ACTIVE
    op.execute(
        sa.text("""
            UPDATE sessions
            SET sandbox_state = 'active',
                sandbox_generation = 1,
                sandbox_created_at = created_at
            WHERE status IN ('running', 'waiting', 'takeover', 'takeover_pending', 'finishing')
              AND sandbox_id IS NOT NULL
        """)
    )

    # COMPLETED / TIMED_OUT with sandbox → SUSPENDED (§16: allow reopen)
    op.execute(
        sa.text("""
            UPDATE sessions
            SET sandbox_state = 'suspended',
                sandbox_generation = 1,
                sandbox_created_at = created_at
            WHERE status IN ('completed', 'timed_out')
              AND sandbox_id IS NOT NULL
        """)
    )

    # COMPLETED / TIMED_OUT without sandbox → DESTROYED
    op.execute(
        sa.text("""
            UPDATE sessions
            SET sandbox_state = 'destroyed',
                sandbox_generation = 1,
                sandbox_destroyed_at = COALESCE(completed_at, updated_at),
                sandbox_destroy_reason = 'session_delete'
            WHERE status IN ('completed', 'timed_out')
              AND sandbox_id IS NULL
        """)
    )

    # PENDING with sandbox (anomalous) → DESTROYED
    op.execute(
        sa.text("""
            UPDATE sessions
            SET sandbox_state = 'destroyed',
                sandbox_generation = 1,
                sandbox_destroyed_at = updated_at,
                sandbox_destroy_reason = 'reconcile_orphan'
            WHERE status = 'pending'
              AND sandbox_id IS NOT NULL
        """)
    )

    # PENDING without sandbox → UNBOUND (already the server_default, no-op)

    # Step 3: Add index for reconcile_orphans scan
    op.create_index(
        "ix_sessions_sandbox_state",
        "sessions",
        ["sandbox_state"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_sandbox_state", table_name="sessions")
    op.drop_column("sessions", "sandbox_destroy_reason")
    op.drop_column("sessions", "sandbox_destroyed_at")
    op.drop_column("sessions", "sandbox_created_at")
    op.drop_column("sessions", "sandbox_generation")
    op.drop_column("sessions", "sandbox_state")
