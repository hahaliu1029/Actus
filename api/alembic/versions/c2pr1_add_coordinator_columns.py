"""C2 PR-1: add coordinator columns to sessions.

Revision ID: c2pr1_coordinator_columns
Revises: c3pr6_retire_legacy_ctrl_plane
Create Date: 2026-05-25

Spec §4.7 + §12.4 P0-6 (JSONB attempts) + P1-5 (length alignment):

- ``coordinator_run_id VARCHAR(320)`` — identifier shape
  ``f"{session_id}:{step_id_hash16}:a{attempt_ix}"``. The 320 cap is the
  P1-5 sum of ``sessions.id`` (VARCHAR(255)) + ``":" + 16-hex hash + ":a" +
  decimal attempt index``. NULL on rows that pre-date the coordinator (and
  on non-coordinator children).
- ``work_unit_id VARCHAR(64)`` — identifier shape
  ``f"{step_id_hash16}.a{attempt_ix}.{i}"``. 64 chars is comfortable for
  the 16-hex hash + ".a" + small decimals + "." + small index.
- ``coordinator_attempts JSONB NOT NULL DEFAULT '{}'::jsonb`` — per-step
  attempt counter map ``{step_id_hash16: attempt_ix}``. JSONB is the
  P0-6 choice over TEXT so consumers can use ``jsonb_set`` / ``->>`` for
  per-step counter bump without read-modify-write round-trips.

Two partial indexes:

- ``ix_sessions_coordinator_run`` — non-unique lookup
  ``(parent_session_id, coordinator_run_id, work_unit_id)`` for "list all
  work units of a coordinator run". Partial WHERE
  ``coordinator_run_id IS NOT NULL`` so the index stays small on the vast
  majority of rows that aren't coordinator children.
- ``ux_sessions_coordinator_wu`` — UNIQUE
  ``(parent_session_id, coordinator_run_id, work_unit_id)`` partial WHERE
  ``coordinator_run_id IS NOT NULL AND work_unit_id IS NOT NULL``. This is
  the idempotent retry guard: a second coordinator step that tries to
  spawn a duplicate ``(run_id, work_unit_id)`` under the same parent gets
  blocked at the DB layer, even if the application-level dedup misfires.

CHECK constraint widening:

The existing ``ck_sessions_tool_filter_preset`` from t12 only allowed
``'subagent_research'``. C2 PR-1's planner emits coordinator step tasks
under a new ``'coordinator_step'`` preset (different tool allowlist than
subagent_research), so we DROP + RECREATE the CHECK to allow both
values. Downgrade restores the narrow t12 form.

**Downgrade contract (DESTRUCTIVE):** ``downgrade()`` drops the columns
unconditionally. Any live coordinator run loses its persisted
``(run_id, work_unit_id, attempts)`` state on rollback. Operators must
terminate or COMPLETE active coordinator runs before downgrading. The
narrowed CHECK ``IN ('subagent_research')`` will also reject any
``tool_filter_preset='coordinator_step'`` rows still present — the
downgrade DROP CONSTRAINT + ADD CONSTRAINT round-trip validates against
existing data, so the operator must additionally null those rows out
first. This matches the destructive-downgrade semantics established by
t12 / p1m.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "c2pr1_coordinator_columns"
down_revision = "c3pr6_retire_legacy_ctrl_plane"
branch_labels = None
depends_on = None


_CHECK_NAME = "ck_sessions_tool_filter_preset"


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column("coordinator_run_id", sa.String(length=320), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column("work_unit_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "sessions",
        sa.Column(
            "coordinator_attempts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    # Partial indexes — narrow WHERE clauses keep them sparse on the
    # majority of rows that aren't coordinator children. See module
    # docstring for the lookup vs. uniqueness rationale.
    op.create_index(
        "ix_sessions_coordinator_run",
        "sessions",
        ["parent_session_id", "coordinator_run_id", "work_unit_id"],
        unique=False,
        postgresql_where=sa.text("coordinator_run_id IS NOT NULL"),
    )
    op.create_index(
        "ux_sessions_coordinator_wu",
        "sessions",
        ["parent_session_id", "coordinator_run_id", "work_unit_id"],
        unique=True,
        postgresql_where=sa.text(
            "coordinator_run_id IS NOT NULL "
            "AND work_unit_id IS NOT NULL"
        ),
    )

    # Widen the t12 CHECK from {'subagent_research'} → {'subagent_research',
    # 'coordinator_step'}. Raw SQL (not ``op.create_check_constraint``) on
    # purpose: SQLAlchemy's naming_convention (api/app/infrastructure/models/
    # base.py:5-11 — ``ck_%(table_name)s_%(constraint_name)s``) would double-
    # prefix to ``ck_sessions_ck_sessions_tool_filter_preset``. The
    # t12 migration uses the same raw-SQL approach for the same reason.
    op.execute(
        f"ALTER TABLE sessions DROP CONSTRAINT IF EXISTS {_CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE sessions ADD CONSTRAINT {_CHECK_NAME} "
        "CHECK (tool_filter_preset IS NULL "
        "OR tool_filter_preset IN ('subagent_research', 'coordinator_step'))"
    )


def downgrade() -> None:
    # Restore narrow t12 CHECK first, then drop indexes + columns in reverse.
    op.execute(
        f"ALTER TABLE sessions DROP CONSTRAINT IF EXISTS {_CHECK_NAME}"
    )
    op.execute(
        f"ALTER TABLE sessions ADD CONSTRAINT {_CHECK_NAME} "
        "CHECK (tool_filter_preset IS NULL "
        "OR tool_filter_preset IN ('subagent_research'))"
    )
    op.drop_index("ux_sessions_coordinator_wu", table_name="sessions")
    op.drop_index("ix_sessions_coordinator_run", table_name="sessions")
    op.drop_column("sessions", "coordinator_attempts")
    op.drop_column("sessions", "work_unit_id")
    op.drop_column("sessions", "coordinator_run_id")
