"""add R3 skill risk metadata columns

Revision ID: r3_skill_risk_columns
Revises: s3_sandbox_lifecycle_log
Create Date: 2026-04-16 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "r3_skill_risk_columns"
down_revision: Union[str, Sequence[str], None] = "s3_sandbox_lifecycle_log"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add trust_origin, scan_report, force_approved_hash to skills table."""
    op.add_column(
        "skills",
        sa.Column(
            "trust_origin",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'user_installed'"),
        ),
    )
    op.add_column(
        "skills",
        sa.Column(
            "scan_report",
            postgresql.JSONB(),
            nullable=True,
            server_default=sa.text("NULL"),
        ),
    )
    op.add_column(
        "skills",
        sa.Column(
            "force_approved_hash",
            sa.String(length=128),
            nullable=True,
            server_default=sa.text("NULL"),
        ),
    )


def downgrade() -> None:
    """Remove R3 skill risk metadata columns."""
    op.drop_column("skills", "force_approved_hash")
    op.drop_column("skills", "scan_report")
    op.drop_column("skills", "trust_origin")
