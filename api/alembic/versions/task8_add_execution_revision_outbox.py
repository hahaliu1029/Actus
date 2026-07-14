"""Task 8: add generation-safe execution revision and durable pending event.

Revision ID: task8_exec_revision
Revises: d1a_add_extension_registry
Create Date: 2026-07-14
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "task8_exec_revision"
down_revision = "d1a_add_extension_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column(
            "execution_revision",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "pending_execution_event",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("sessions", "pending_execution_event")
    op.drop_column("sessions", "execution_revision")
