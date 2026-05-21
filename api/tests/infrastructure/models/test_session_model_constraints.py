"""Structure tests for SessionModel.__table__.constraints.

C3 PR-1 codex round 12 P2: ORM-level CHECK constraints declared in
``__table_args__`` are subject to ``Base.metadata.naming_convention``
(``ck_%(table_name)s_%(constraint_name)s``). A bare
``name="ck_sessions_subagent_control_plane_valid"`` therefore double-prefixes
to ``ck_sessions_ck_sessions_subagent_control_plane_valid`` in ORM metadata —
mismatching the migration's raw SQL constraint name. The
``sqlalchemy.schema.conv()`` wrapper opts the name out of the convention so
the ORM and the migration agree.

These tests lock that contract in so:

  - ``Base.metadata.create_all()`` (used by integration test conftest) emits
    the same constraint name the migration uses
  - ``alembic revision --autogenerate`` does NOT produce spurious
    drop-and-recreate diffs against an existing DB
"""

from __future__ import annotations

from app.infrastructure.models.session import SessionModel


def test_subagent_control_plane_check_constraint_name_matches_migration() -> None:
    """C3 PR-1 codex round 12 P2: ORM CHECK name must match migration's raw
    SQL name verbatim (no naming_convention double-prefix), else
    ``create_all``-based tests and alembic autogenerate disagree."""
    names = {c.name for c in SessionModel.__table__.constraints if c.name}
    assert "ck_sessions_subagent_control_plane_valid" in names, (
        "Expected migration-aligned CHECK name; got: "
        f"{sorted(n for n in names if n and 'control_plane' in n)}"
    )
    assert "ck_sessions_ck_sessions_subagent_control_plane_valid" not in names, (
        "naming_convention double-prefix detected — wrap the name with "
        "sqlalchemy.schema.conv() in SessionModel.__table_args__."
    )
