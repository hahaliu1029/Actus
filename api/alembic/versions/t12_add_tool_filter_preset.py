"""t12_add_tool_filter_preset

Revision ID: t12_tool_filter_preset
Revises: p1m_sample_session_id
Create Date: 2026-05-19

Phase 1 PR-X / T12 — pod-restart resilience for child-session ``tool_filter``.

Adds ``sessions.tool_filter_preset`` (nullable VARCHAR(64)) so that when a
child session task is reconstructed after a pod restart (resume / FINISHING
/ orphan / preflight paths in ``AgentService._create_task``), the runtime can
re-derive the in-memory ``tool_filter`` allowlist from a persisted preset
name. Plain ``execution_mode`` could not carry this without widening its
two-valued Literal and re-auditing 12+ ``== "background"`` branch points; a
dedicated additive column is the safer surface.

Currently the only valid preset is ``'subagent_research'`` (mapped to the
read-only allowlist consumed by ``SubagentResearchService``).

Two CHECK constraints are applied:

1. ``ck_sessions_tool_filter_preset`` — value domain.
   ``tool_filter_preset IS NULL OR tool_filter_preset IN ('subagent_research')``.
   Future presets must add an explicit migration step rather than silently
   flowing through.

2. ``ck_sessions_child_must_have_preset`` — defense-in-depth invariant.
   ``sample_session_id IS NULL OR tool_filter_preset IS NOT NULL``.
   Closes the codex R1 P1 bypass: every child row (``sample_session_id``
   non-null) MUST carry a non-null preset, so a row created via raw SQL or
   a future buggy caller cannot persist as "child without restriction" and
   then be reconstructed on resume with an unbounded tool registry.
   ``SessionService.create_session_with_parent`` mirrors this guard with an
   ``ValueError`` for friendlier error messages; the DB constraint is the
   last-line defense.

Partial index is intentionally omitted — the column is consulted only on
task reconstruction for an already-loaded session row, so no scan path
benefits from an index. A future preset that requires lookup-by-preset
should add its own index in the same migration that introduces the value.

**Downgrade contract (DESTRUCTIVE):** ``downgrade()`` drops the column
unconditionally. Any live child session with ``tool_filter_preset =
'subagent_research'`` loses its persisted first gate on rollback; on
subsequent re-upgrade the column comes back NULL and the child runs
**unrestricted** until the child session naturally completes. Operators
must therefore terminate or COMPLETE active subagent_research child
sessions before downgrading (e.g. ``UPDATE sessions SET status='completed'
WHERE tool_filter_preset='subagent_research' AND status NOT IN
('completed','timed_out');``) and accept that PE-0 alone is the gate during
the downgrade window. This matches the destructive-downgrade semantics
already established for ``sample_session_id`` (p1m).
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "t12_tool_filter_preset"
down_revision = "p1m_sample_session_id"
branch_labels = None
depends_on = None


_VALID_PRESETS = ("subagent_research",)
_CHECK_NAME = "ck_sessions_tool_filter_preset"
_CHILD_CHECK_NAME = "ck_sessions_child_must_have_preset"


def upgrade() -> None:
    op.add_column(
        "sessions",
        sa.Column(
            "tool_filter_preset",
            sa.String(length=64),
            nullable=True,
        ),
    )

    # Raw SQL (not ``op.create_check_constraint``) on purpose:
    # ``op.create_check_constraint("ck_sessions_tool_filter_preset", ...)``
    # would route through SQLAlchemy's naming_convention
    # (api/app/infrastructure/models/base.py:5-11 — ``ck`` template is
    # ``ck_%(table_name)s_%(constraint_name)s``), producing
    # ``ck_sessions_ck_sessions_tool_filter_preset`` (double prefix). Same
    # pattern documented in b4m1_add_cost_records_value_checks.py. Raw
    # SQL with the literal name preserves the canonical name the
    # integration tests assert against.
    preset_list = ", ".join(f"'{p}'" for p in _VALID_PRESETS)
    op.execute(
        f"ALTER TABLE sessions ADD CONSTRAINT {_CHECK_NAME} "
        f"CHECK (tool_filter_preset IS NULL "
        f"OR tool_filter_preset IN ({preset_list}))"
    )

    # Codex R2 P1 fix — backfill historical children BEFORE the cross-column
    # CHECK is added. develop has already shipped subagent_research (PR-4..6
    # commit cc14878 / 4e45f68 etc.), so any existing dev / staging DB row
    # with sample_session_id IS NOT NULL is a subagent_research child that
    # was running under the in-memory-only F8-gap regime. Without this
    # backfill the next statement (cross-column CHECK) refuses to add the
    # constraint when historical children exist, breaking the deploy. Safe
    # blanket UPDATE: the only producer of sample_session_id today is
    # SubagentResearchService and it always tags 'subagent_research'.
    op.execute(
        "UPDATE sessions "
        "SET tool_filter_preset = 'subagent_research' "
        "WHERE sample_session_id IS NOT NULL "
        "  AND tool_filter_preset IS NULL"
    )

    # Defense-in-depth: child rows (sample_session_id non-null) MUST carry a
    # non-null preset so a raw-SQL or buggy-caller insert can't produce an
    # "unrestricted child" that bypasses the F8 fix at reconstruction time.
    # Postgres validates the CHECK against existing data at constraint
    # creation time — the backfill above guarantees historical rows pass.
    # Raw SQL again to avoid the naming_convention double-prefix described
    # above.
    op.execute(
        f"ALTER TABLE sessions ADD CONSTRAINT {_CHILD_CHECK_NAME} "
        "CHECK (sample_session_id IS NULL OR tool_filter_preset IS NOT NULL)"
    )


def downgrade() -> None:
    # Drop in reverse order. Use raw SQL on the literal names so we don't
    # accidentally invoke naming_convention drift on a future schema state.
    op.execute(f"ALTER TABLE sessions DROP CONSTRAINT {_CHILD_CHECK_NAME}")
    op.execute(f"ALTER TABLE sessions DROP CONSTRAINT {_CHECK_NAME}")
    op.drop_column("sessions", "tool_filter_preset")
