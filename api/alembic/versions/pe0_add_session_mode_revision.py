"""PE-0: add sessions.mode_revision strict monotonic counter

Revision ID: pe0_mode_rev
Revises: b3p2_supervisor_columns
Create Date: 2026-05-14

Adds a BIGINT counter, default 0. SessionStateMachine.transition()
performs UPDATE sessions SET status=:new, mode_revision=mode_revision+1
WHERE id=:sid AND status=:from_state (atomic CAS).

PermissionEngine reads (mode, mode_revision) before and after slow
stages (SmartApprove LLM) and raises PolicyConflict if mode_revision
changed during evaluation — captures RUNNING -> TAKEOVER -> RUNNING
round-trip races that value-comparison on status alone misses.
"""

from alembic import op
import sqlalchemy as sa

revision = "pe0_mode_rev"
down_revision = "b3p2_supervisor_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column(
            "mode_revision",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "mode_revision")
