"""add skills table

Revision ID: c4fbe9d1a8c1
Revises: 3a5b7c9d1e2f
Create Date: 2026-02-22 22:15:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "c4fbe9d1a8c1"
down_revision: Union[str, Sequence[str], None] = "3a5b7c9d1e2f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "skills",
        sa.Column("id", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column(
            "description",
            sa.String(length=1024),
            nullable=False,
            server_default=sa.text("''::character varying"),
        ),
        sa.Column(
            "version",
            sa.String(length=64),
            nullable=False,
            server_default=sa.text("'0.1.0'"),
        ),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_ref", sa.String(length=512), nullable=False),
        sa.Column("runtime_type", sa.String(length=32), nullable=False),
        sa.Column(
            "manifest",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("installed_by", sa.String(length=255), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP(0)"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_skills_id"),
        sa.UniqueConstraint("slug", name="uq_skills_slug"),
        sa.ForeignKeyConstraint(
            ["installed_by"],
            ["users.id"],
            name="fk_skills_installed_by_users",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_skills_slug", "skills", ["slug"], unique=False)
    op.create_index("ix_skills_installed_by", "skills", ["installed_by"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_skills_installed_by", table_name="skills")
    op.drop_index("ix_skills_slug", table_name="skills")
    op.drop_table("skills")
