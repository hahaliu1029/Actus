"""phase1_minimal_add_sample_session_id

Revision ID: p1m_sample_session_id
Revises: pe0_mode_rev
Create Date: 2026-05-18

Phase 1 minimal subagent feature: adds sample_session_id nullable FK
to sessions table for child-session relationship.

FK semantics: ondelete=RESTRICT — parent delete is blocked while children
exist; prevents orphan sessions that would bypass frontend selector filter
(useFilteredSessionsForList filters by sample_session_id is null).

Partial index on (sample_session_id) WHERE sample_session_id IS NOT NULL
for efficient child queries; the column is mostly NULL.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "p1m_sample_session_id"
down_revision = "pe0_mode_rev"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("sample_session_id", sa.String(length=255), nullable=True),
    )
    op.create_foreign_key(
        "fk_sessions_sample_session_id_sessions",
        "sessions",
        "sessions",
        ["sample_session_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_sessions_sample_session_id",
        "sessions",
        ["sample_session_id"],
        postgresql_where=sa.text("sample_session_id IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index("ix_sessions_sample_session_id", table_name="sessions")
    op.drop_constraint(
        "fk_sessions_sample_session_id_sessions",
        "sessions",
        type_="foreignkey",
    )
    op.drop_column("sessions", "sample_session_id")
