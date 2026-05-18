"""Verify p1m_sample_session_id adds + drops the column + FK + partial index cleanly.

PR-0 / Phase 1 minimal subagent: ensures the schema migration round-trips
correctly and that the FK delete rule is RESTRICT (per design — parent delete
is blocked while children exist; prevents orphan sessions that would bypass
the frontend useFilteredSessionsForList filter).

Also verifies the partial index on (sample_session_id) WHERE sample_session_id
IS NOT NULL is created (the column is mostly NULL so a partial index is
appropriate).

Includes a downgrade→upgrade roundtrip (test_downgrade_upgrade_roundtrip)
that pins the boundary to (pe0_mode_rev → p1m_sample_session_id) so future
revisions stacking on top of p1m don't silently shift the assertions onto
the wrong pair of revisions. The `finally` clause always restores `head` so
a failed assertion can't leave the DB in a pre-p1m state and pollute
subsequent integration tests.
"""

from __future__ import annotations

import os
import uuid as _uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

pytestmark = pytest.mark.integration


# Mirror conftest.py fallback (see test_r6_migration.py:25-28) so this file is
# runnable the same way as its sibling integration tests.
DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)

# Pin the boundary under test so a future revision stacking on top of
# p1m_sample_session_id doesn't silently shift the "roundtrip" assertions
# onto the wrong pair of revisions.
P1M_REVISION = "p1m_sample_session_id"
PRE_P1M_REVISION = "pe0_mode_rev"


@pytest.fixture
def alembic_cfg() -> Config:
    """Sync alembic Config pointing at the test DB (psycopg2 URL)."""
    sync_url = DB_URL.replace("+asyncpg", "+psycopg2", 1)
    api_root = Path(__file__).resolve().parent.parent.parent.parent  # api/
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


@pytest.mark.asyncio
async def test_upgrade_creates_sample_session_id_column(db_session):
    """alembic upgrade head must add sample_session_id as VARCHAR(255) NULLABLE."""
    # AsyncSession.get_bind() returns the *sync* Engine which has no
    # run_sync. We need an AsyncConnection — obtained via .connection().
    async_conn = await db_session.connection()

    def _check(sync_conn):
        cols = {c["name"]: c for c in inspect(sync_conn).get_columns("sessions")}
        assert "sample_session_id" in cols, "sample_session_id column missing"
        col = cols["sample_session_id"]
        assert col["nullable"] is True, "sample_session_id must be nullable"
        return col

    col = await async_conn.run_sync(_check)
    # Type should be VARCHAR(255) — alembic op.add_column with String(length=255).
    assert "VARCHAR" in str(col["type"]).upper()
    assert getattr(col["type"], "length", None) == 255, (
        f"sample_session_id must be VARCHAR(255), got length="
        f"{getattr(col['type'], 'length', None)!r}"
    )


@pytest.mark.asyncio
async def test_upgrade_creates_fk_with_restrict(db_session):
    """The fk_sessions_sample_session_id_sessions self-FK must use ON DELETE RESTRICT."""
    res = await db_session.execute(
        text(
            """
            SELECT rc.delete_rule
            FROM information_schema.referential_constraints rc
            JOIN information_schema.table_constraints tc
              ON tc.constraint_catalog = rc.constraint_catalog
             AND tc.constraint_schema = rc.constraint_schema
             AND tc.constraint_name = rc.constraint_name
            WHERE tc.table_schema = 'public'
              AND tc.table_name = 'sessions'
              AND tc.constraint_name = 'fk_sessions_sample_session_id_sessions'
            """
        )
    )
    row = res.first()
    assert row is not None, (
        "fk_sessions_sample_session_id_sessions constraint missing"
    )
    assert row[0] == "RESTRICT", (
        "fk_sessions_sample_session_id_sessions delete_rule must be RESTRICT, "
        f"got {row[0]!r}"
    )


@pytest.mark.asyncio
async def test_upgrade_creates_partial_index(db_session):
    """The ix_sessions_sample_session_id index must be a partial index
    (WHERE sample_session_id IS NOT NULL)."""
    res = await db_session.execute(
        text(
            """
            SELECT indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND tablename = 'sessions'
              AND indexname = 'ix_sessions_sample_session_id'
            """
        )
    )
    row = res.first()
    assert row is not None, "ix_sessions_sample_session_id index missing"
    indexdef = row[0]
    # PostgreSQL serialises the WHERE clause back into the indexdef; assert it
    # references sample_session_id IS NOT NULL (case/whitespace flexible).
    assert "sample_session_id" in indexdef.lower()
    assert "is not null" in indexdef.lower(), (
        f"partial WHERE clause missing from indexdef: {indexdef!r}"
    )


def test_downgrade_upgrade_roundtrip(alembic_cfg):
    """p1m_sample_session_id → pe0_mode_rev → p1m_sample_session_id.

    Verifies the migration is reversible: downgrade drops the column, FK,
    and partial index; upgrade re-adds them with the exact original shape.
    The `finally` clause restores `head` even on assertion failure so
    subsequent integration tests aren't poisoned by a downgraded DB.
    """
    sync_url = alembic_cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(sync_url)

    try:
        # Establish baseline at p1m (conftest already upgraded to head; pin
        # explicitly so any future revision on top of p1m can't shift what
        # we're about to assert).
        command.upgrade(alembic_cfg, P1M_REVISION)

        # ---- Downgrade to pe0_mode_rev: column / FK / partial index gone. ----
        command.downgrade(alembic_cfg, PRE_P1M_REVISION)

        with engine.connect() as conn:
            # Column gone.
            col_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'sessions'
                      AND column_name = 'sample_session_id'
                    """
                )
            ).scalar()
            assert col_count == 0, (
                "sample_session_id column still present after downgrade"
            )

            # FK gone.
            fk_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*)
                    FROM information_schema.referential_constraints rc
                    JOIN information_schema.table_constraints tc
                      ON tc.constraint_catalog = rc.constraint_catalog
                     AND tc.constraint_schema = rc.constraint_schema
                     AND tc.constraint_name = rc.constraint_name
                    WHERE tc.table_schema = 'public'
                      AND tc.table_name = 'sessions'
                      AND tc.constraint_name = 'fk_sessions_sample_session_id_sessions'
                    """
                )
            ).scalar()
            assert fk_count == 0, (
                "fk_sessions_sample_session_id_sessions still present after downgrade"
            )

            # Partial index gone.
            idx_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'sessions'
                      AND indexname = 'ix_sessions_sample_session_id'
                    """
                )
            ).scalar()
            assert idx_count == 0, (
                "ix_sessions_sample_session_id still present after downgrade"
            )

        # ---- Upgrade back to p1m: column / FK / partial index restored. ----
        command.upgrade(alembic_cfg, P1M_REVISION)

        with engine.connect() as conn:
            # Column back, nullable VARCHAR(255).
            col_row = conn.execute(
                text(
                    """
                    SELECT is_nullable, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'sessions'
                      AND column_name = 'sample_session_id'
                    """
                )
            ).first()
            assert col_row is not None, (
                "sample_session_id column missing after re-upgrade"
            )
            assert col_row[0] == "YES", "sample_session_id must be nullable"
            assert col_row[1].lower() == "character varying", (
                f"sample_session_id must be character varying, got {col_row[1]!r}"
            )
            assert col_row[2] == 255, (
                f"sample_session_id must be VARCHAR(255), got length={col_row[2]!r}"
            )

            # FK back with ON DELETE RESTRICT.
            fk_row = conn.execute(
                text(
                    """
                    SELECT rc.delete_rule
                    FROM information_schema.referential_constraints rc
                    JOIN information_schema.table_constraints tc
                      ON tc.constraint_catalog = rc.constraint_catalog
                     AND tc.constraint_schema = rc.constraint_schema
                     AND tc.constraint_name = rc.constraint_name
                    WHERE tc.table_schema = 'public'
                      AND tc.table_name = 'sessions'
                      AND tc.constraint_name = 'fk_sessions_sample_session_id_sessions'
                    """
                )
            ).first()
            assert fk_row is not None, (
                "fk_sessions_sample_session_id_sessions missing after re-upgrade"
            )
            assert fk_row[0] == "RESTRICT", (
                "fk delete_rule must be RESTRICT after re-upgrade, "
                f"got {fk_row[0]!r}"
            )

            # Partial index back with WHERE sample_session_id IS NOT NULL.
            idx_row = conn.execute(
                text(
                    """
                    SELECT indexdef FROM pg_indexes
                    WHERE schemaname = 'public'
                      AND tablename = 'sessions'
                      AND indexname = 'ix_sessions_sample_session_id'
                    """
                )
            ).first()
            assert idx_row is not None, (
                "ix_sessions_sample_session_id missing after re-upgrade"
            )
            indexdef = idx_row[0]
            assert "sample_session_id" in indexdef.lower()
            assert "is not null" in indexdef.lower(), (
                "partial WHERE clause missing from indexdef after re-upgrade: "
                f"{indexdef!r}"
            )
    finally:
        # Always restore schema to the real head so a mid-test failure
        # doesn't leave subsequent integration tests running against a
        # downgraded DB.
        command.upgrade(alembic_cfg, "head")


@pytest.mark.asyncio
async def test_fk_restrict_blocks_parent_delete_with_live_children(db_session):
    """FK RESTRICT semantics: deleting a parent session that still has live
    children must raise IntegrityError.

    This complements test_downgrade_upgrade_roundtrip (which verifies the
    schema-level FK shape) by exercising the operationally critical
    RESTRICT *behavior* that prevents orphan child sessions from bypassing
    the frontend useFilteredSessionsForList filter at runtime.
    """
    from app.infrastructure.models.session import SessionModel
    from app.infrastructure.models.user import UserModel

    uid = str(_uuid.uuid4())
    parent_sid = f"sess-p1m-parent-{_uuid.uuid4().hex[:12]}"
    child_sid = f"sess-p1m-child-{_uuid.uuid4().hex[:12]}"

    db_session.add(
        UserModel(id=uid, username=f"p1mtest_{uid[:8]}", password_hash="x")
    )
    db_session.add(
        SessionModel(
            id=parent_sid,
            user_id=uid,
            status="pending",
            title="parent",
        )
    )
    db_session.add(
        SessionModel(
            id=child_sid,
            user_id=uid,
            status="pending",
            title="child",
            sample_session_id=parent_sid,
        )
    )
    await db_session.flush()

    # Verify the link is observable right after insert (post-upgrade state).
    res = await db_session.execute(
        text(
            "SELECT sample_session_id FROM sessions WHERE id = :sid"
        ),
        {"sid": child_sid},
    )
    assert res.scalar_one() == parent_sid

    # Verify RESTRICT semantics: deleting the parent while the child still
    # references it must raise (IntegrityError from asyncpg → DBAPIError).
    # Wrap in begin_nested() savepoint so the failing transaction is isolated
    # and the outer fixture rollback doesn't emit "transaction is aborted"
    # warnings (matches test_r5_grants_schema.py pattern).
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                text("DELETE FROM sessions WHERE id = :sid"),
                {"sid": parent_sid},
            )
            await db_session.flush()
