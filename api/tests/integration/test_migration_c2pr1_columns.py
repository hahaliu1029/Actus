"""C2 PR-1: integration tests for ``c2pr1_add_coordinator_columns`` migration.

Verifies the post-upgrade DB schema contract added by
``c2pr1_coordinator_columns``:

  * 3 new columns on ``sessions`` — ``coordinator_run_id`` VARCHAR(320) NULLABLE,
    ``work_unit_id`` VARCHAR(64) NULLABLE, ``coordinator_attempts`` JSONB
    NOT NULL DEFAULT ``'{}'::jsonb``.
  * 2 partial indexes — ``ix_sessions_coordinator_run`` (non-unique, WHERE
    ``coordinator_run_id IS NOT NULL``) and ``ux_sessions_coordinator_wu``
    (UNIQUE, WHERE ``coordinator_run_id IS NOT NULL AND work_unit_id IS NOT
    NULL``).
  * Widened ``ck_sessions_tool_filter_preset`` CHECK accepting both
    ``'subagent_research'`` and ``'coordinator_step'``.

Conventions:
  * Fixtures: this file consumes ``db_session`` + ``sample_user`` from
    ``api/tests/integration/conftest.py``. ``async_session`` (as referenced in
    a draft of the spec) does not exist in this repo — the canonical
    auto-rollback async session fixture is ``db_session``.
  * FK satisfaction: ``test_check_allows_coordinator_step`` flushes a parent
    ``SessionModel`` row inline before inserting the child with
    ``tool_filter_preset='coordinator_step'``. Memory
    ``feedback_integration_test_fk`` documents this requirement.

Run: ``cd api && uv run pytest tests/integration/test_migration_c2pr1_columns.py -v``
Requires: pgvector-enabled Postgres reachable via ``SQLALCHEMY_DATABASE_URL``.
"""

from __future__ import annotations

import uuid as _uuid

import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_columns_present(db_session) -> None:
    """The 3 coordinator columns must exist with the right shape."""
    result = await db_session.execute(
        text(
            """
            SELECT column_name,
                   data_type,
                   character_maximum_length,
                   is_nullable
            FROM information_schema.columns
            WHERE table_name = 'sessions'
              AND column_name IN (
                'coordinator_run_id',
                'work_unit_id',
                'coordinator_attempts'
              )
            ORDER BY column_name
            """
        )
    )
    cols = {r[0]: r for r in result.fetchall()}

    assert "coordinator_run_id" in cols, "coordinator_run_id column missing"
    assert cols["coordinator_run_id"][1] == "character varying"
    assert cols["coordinator_run_id"][2] == 320, (
        "coordinator_run_id must be VARCHAR(320), got "
        f"{cols['coordinator_run_id'][2]!r}"
    )
    assert cols["coordinator_run_id"][3] == "YES", (
        "coordinator_run_id must be NULLABLE"
    )

    assert "work_unit_id" in cols, "work_unit_id column missing"
    assert cols["work_unit_id"][1] == "character varying"
    assert cols["work_unit_id"][2] == 64, (
        f"work_unit_id must be VARCHAR(64), got {cols['work_unit_id'][2]!r}"
    )
    assert cols["work_unit_id"][3] == "YES", "work_unit_id must be NULLABLE"

    assert "coordinator_attempts" in cols, "coordinator_attempts column missing"
    assert cols["coordinator_attempts"][1] == "jsonb", (
        "coordinator_attempts must be JSONB, got "
        f"{cols['coordinator_attempts'][1]!r}"
    )
    assert cols["coordinator_attempts"][3] == "NO", (
        "coordinator_attempts must be NOT NULL"
    )


async def test_partial_indexes(db_session) -> None:
    """Both partial indexes exist with the right uniqueness + predicates."""
    result = await db_session.execute(
        text(
            """
            SELECT indexname, indexdef
            FROM pg_indexes
            WHERE tablename = 'sessions'
              AND indexname IN (
                'ix_sessions_coordinator_run',
                'ux_sessions_coordinator_wu'
              )
            """
        )
    )
    by_name = {r[0]: r[1] for r in result.fetchall()}

    assert set(by_name.keys()) == {
        "ix_sessions_coordinator_run",
        "ux_sessions_coordinator_wu",
    }

    # Non-unique lookup index — must NOT be UNIQUE and must carry the
    # partial-WHERE clause that keeps it sparse.
    ix_def = by_name["ix_sessions_coordinator_run"]
    assert not ix_def.upper().startswith("CREATE UNIQUE INDEX"), (
        f"ix_sessions_coordinator_run must NOT be unique: {ix_def!r}"
    )
    assert "coordinator_run_id IS NOT NULL" in ix_def, (
        f"ix_sessions_coordinator_run missing partial WHERE: {ix_def!r}"
    )

    # Unique idempotent-retry guard — MUST be UNIQUE and carry the
    # narrower partial-WHERE (both run + wu non-null).
    ux_def = by_name["ux_sessions_coordinator_wu"]
    assert ux_def.upper().startswith("CREATE UNIQUE INDEX"), (
        f"ux_sessions_coordinator_wu must be UNIQUE: {ux_def!r}"
    )
    assert "coordinator_run_id IS NOT NULL" in ux_def
    assert "work_unit_id IS NOT NULL" in ux_def


async def test_check_allows_coordinator_step(db_session, sample_user) -> None:
    """The widened CHECK must accept ``tool_filter_preset='coordinator_step'``.

    Inserts a parent row first to satisfy ``parent_session_id`` FK, then
    inserts a coordinator-step child. Both rows go through the auto-rollback
    ``db_session`` so no cleanup is needed.

    ``sample_user`` provides a real ``users.id`` row to satisfy the
    ``sessions.user_id`` FK (ON DELETE SET NULL but the row must exist at
    INSERT time).
    """
    parent_id = f"sess-c2pr1-p-{_uuid.uuid4().hex[:12]}"
    child_id = f"sess-c2pr1-c-{_uuid.uuid4().hex[:12]}"
    uid = sample_user.id

    # Parent row — minimal columns; server_defaults cover the rest.
    await db_session.execute(
        text(
            """
            INSERT INTO sessions (id, user_id, worker_type, status, title)
            VALUES (:sid, :uid, 'root', 'pending', 'c2pr1 parent')
            """
        ),
        {"sid": parent_id, "uid": uid},
    )

    # Child row carrying tool_filter_preset='coordinator_step' — the new
    # CHECK clause must accept this value (post-c2pr1) where the narrower
    # t12 CHECK would have rejected it.
    await db_session.execute(
        text(
            """
            INSERT INTO sessions (
                id, user_id, worker_type, parent_session_id,
                tool_filter_preset, status, title
            )
            VALUES (
                :sid, :uid, 'subagent', :psid,
                'coordinator_step', 'pending', 'c2pr1 child'
            )
            """
        ),
        {"sid": child_id, "uid": uid, "psid": parent_id},
    )
    await db_session.flush()

    # Sanity read-back — proves the row landed, not just that the INSERT
    # didn't error.
    res = await db_session.execute(
        text(
            "SELECT tool_filter_preset FROM sessions WHERE id = :sid"
        ),
        {"sid": child_id},
    )
    assert res.scalar_one() == "coordinator_step"
