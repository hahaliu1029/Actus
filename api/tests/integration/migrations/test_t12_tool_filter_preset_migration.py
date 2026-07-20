"""Verify t12_tool_filter_preset adds + drops the column + CHECK constraint cleanly.

T12 / Phase 1 PR-X: ensures the migration round-trips correctly and that the
CHECK constraint actually blocks unknown preset names at the DB layer (the
last-line defense behind ``resolve_preset(...)``'s fail-closed ValueError).

Mirrors the structure of test_session_parent_id_migration.py so future
revisions stacking on top of t12_tool_filter_preset don't silently shift the
"roundtrip" assertions onto the wrong revision pair. The `finally` clause
always restores `head` so a failed assertion can't leave the DB in a
pre-t12 state and pollute subsequent integration tests.
"""

from __future__ import annotations

import os
import uuid as _uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

MIGRATION_TARGET = "t12_tool_filter_preset"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.usefixtures("migration_schema_at"),
]


DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)

T12_REVISION = "t12_tool_filter_preset"
PRE_T12_REVISION = "p1m_sample_session_id"
CHECK_NAME = "ck_sessions_tool_filter_preset"
CHILD_CHECK_NAME = "ck_sessions_child_must_have_preset"


@pytest.fixture
def alembic_cfg() -> Config:
    """Sync alembic Config pointing at the test DB (psycopg2 URL)."""
    sync_url = DB_URL.replace("+asyncpg", "+psycopg2", 1)
    api_root = Path(__file__).resolve().parent.parent.parent.parent  # api/
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


async def _insert_user(db_session, *, user_id: str, username: str) -> None:
    await db_session.execute(
        text(
            "INSERT INTO users (id, username, password_hash) "
            "VALUES (:uid, :username, 'x')"
        ),
        {"uid": user_id, "username": username},
    )


async def _insert_session(
    db_session,
    *,
    session_id: str,
    user_id: str,
    title: str,
    sample_session_id: str | None = None,
    tool_filter_preset: str | None = None,
) -> None:
    """Insert against the historical T12 schema without using head ORM."""
    await db_session.execute(
        text(
            "INSERT INTO sessions "
            "(id, user_id, status, title, sample_session_id, tool_filter_preset) "
            "VALUES (:sid, :uid, 'pending', :title, :parent, :preset)"
        ),
        {
            "sid": session_id,
            "uid": user_id,
            "title": title,
            "parent": sample_session_id,
            "preset": tool_filter_preset,
        },
    )


async def test_upgrade_creates_tool_filter_preset_column(db_session):
    """alembic upgrade head must add tool_filter_preset as VARCHAR(64) NULLABLE."""
    async_conn = await db_session.connection()

    def _check(sync_conn):
        cols = {c["name"]: c for c in inspect(sync_conn).get_columns("sessions")}
        assert "tool_filter_preset" in cols, "tool_filter_preset column missing"
        col = cols["tool_filter_preset"]
        assert col["nullable"] is True, "tool_filter_preset must be nullable"
        return col

    col = await async_conn.run_sync(_check)
    assert "VARCHAR" in str(col["type"]).upper()
    assert getattr(col["type"], "length", None) == 64, (
        "tool_filter_preset must be VARCHAR(64), got length="
        f"{getattr(col['type'], 'length', None)!r}"
    )


async def test_upgrade_creates_check_constraint(db_session):
    """The CHECK constraint must exist and reference tool_filter_preset."""
    res = await db_session.execute(
        text(
            """
            SELECT pg_get_constraintdef(c.oid)
            FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = 'public'
              AND t.relname = 'sessions'
              AND c.conname = :cname
              AND c.contype = 'c'
            """
        ),
        {"cname": CHECK_NAME},
    )
    row = res.first()
    assert row is not None, f"CHECK constraint {CHECK_NAME!r} missing"
    constraintdef = row[0].lower()
    assert "tool_filter_preset" in constraintdef
    assert "subagent_research" in constraintdef
    assert "null" in constraintdef


async def test_check_constraint_allows_null_and_known_preset(db_session):
    """Inserts with NULL or known preset value must succeed."""
    uid = str(_uuid.uuid4())
    sid_null = f"sess-t12-null-{_uuid.uuid4().hex[:12]}"
    sid_known = f"sess-t12-known-{_uuid.uuid4().hex[:12]}"

    await _insert_user(
        db_session, user_id=uid, username=f"t12ok_{uid[:8]}"
    )
    await _insert_session(
        db_session,
        session_id=sid_null,
        user_id=uid,
        title="null preset",
    )
    await _insert_session(
        db_session,
        session_id=sid_known,
        user_id=uid,
        title="known preset",
        tool_filter_preset="subagent_research",
    )
    await db_session.flush()

    res = await db_session.execute(
        text(
            "SELECT id, tool_filter_preset FROM sessions "
            "WHERE id IN (:a, :b) ORDER BY id"
        ),
        {"a": sid_null, "b": sid_known},
    )
    rows = {r[0]: r[1] for r in res.all()}
    assert rows[sid_null] is None
    assert rows[sid_known] == "subagent_research"


async def test_check_constraint_rejects_unknown_preset(db_session):
    """An unknown preset value must be rejected by the DB CHECK constraint."""
    uid = str(_uuid.uuid4())
    sid = f"sess-t12-bad-{_uuid.uuid4().hex[:12]}"

    await _insert_user(
        db_session, user_id=uid, username=f"t12bad_{uid[:8]}"
    )

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await _insert_session(
                db_session,
                session_id=sid,
                user_id=uid,
                title="bad preset",
                tool_filter_preset="not_a_real_preset",
            )


async def test_child_check_constraint_rejects_null_preset_on_child(db_session):
    """Codex R1 P1 defense-in-depth: a child row (sample_session_id non-null)
    with tool_filter_preset = NULL must be rejected at the DB layer.

    Even if a future buggy caller (or raw SQL insert) skips the app-level
    ValueError guard in ``SessionService.create_session_with_parent``, the
    ``ck_sessions_child_must_have_preset`` CHECK constraint must block the
    row so a restored task on resume cannot run unrestricted.
    """
    uid = str(_uuid.uuid4())
    parent_sid = f"sess-t12-p-{_uuid.uuid4().hex[:12]}"
    child_sid = f"sess-t12-c-{_uuid.uuid4().hex[:12]}"

    await _insert_user(
        db_session, user_id=uid, username=f"t12child_{uid[:8]}"
    )
    await _insert_session(
        db_session,
        session_id=parent_sid,
        user_id=uid,
        title="parent",
    )

    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await _insert_session(
                db_session,
                session_id=child_sid,
                user_id=uid,
                title="orphan child",
                sample_session_id=parent_sid,
                tool_filter_preset=None,
            )


async def test_child_check_rejects_update_setting_preset_to_null(db_session):
    """Codex R2 P2: CHECK constraints fire on UPDATE too — pin the
    semantic that an existing valid child row cannot be silently demoted
    to "unrestricted" via UPDATE.

    Postgres validates the CHECK against the new row state on every
    UPDATE, but the previous test only exercised INSERT. This nails down
    the UPDATE path so a future code change that does
    ``UPDATE sessions SET tool_filter_preset = NULL WHERE id = ...`` on
    a child row gets caught at the DB layer.
    """
    uid = str(_uuid.uuid4())
    parent_sid = f"sess-t12-up-p-{_uuid.uuid4().hex[:12]}"
    child_sid = f"sess-t12-up-c-{_uuid.uuid4().hex[:12]}"

    await _insert_user(
        db_session, user_id=uid, username=f"t12upd_{uid[:8]}"
    )
    await _insert_session(
        db_session,
        session_id=parent_sid,
        user_id=uid,
        title="parent",
    )
    await _insert_session(
        db_session,
        session_id=child_sid,
        user_id=uid,
        title="legit child",
        sample_session_id=parent_sid,
        tool_filter_preset="subagent_research",
    )
    await db_session.flush()

    # Attempt to demote the child to unrestricted via UPDATE — CHECK must fire.
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await db_session.execute(
                text(
                    "UPDATE sessions SET tool_filter_preset = NULL "
                    "WHERE id = :sid"
                ),
                {"sid": child_sid},
            )
            await db_session.flush()


def test_upgrade_backfills_existing_children_with_subagent_research(
    alembic_cfg,
):
    """Codex R2 P1 + R4 P1: ``upgrade()`` must backfill pre-existing child
    rows (sample_session_id non-null + NULL preset, the historical state
    from PR-1 / PR-4 before T12 added the column) BEFORE the cross-column
    CHECK is added. Otherwise the migration aborts on any dev / staging
    DB that already shipped subagent_research children.

    Proves the backfill *actually ran* by:
      1. downgrade to p1m (pre-T12) — column doesn't exist
      2. INSERT a child with sample_session_id set (legal at p1m)
      3. upgrade to t12 — the backfill UPDATE runs
      4. assert the child now has tool_filter_preset='subagent_research'

    Without the seed step, a vacuous "no violating rows" assertion would
    pass even if the backfill UPDATE were silently removed.
    """
    sync_url = alembic_cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(sync_url)

    uid = _uuid.uuid4().hex
    parent_sid = f"sess-t12-bf-p-{_uuid.uuid4().hex[:12]}"
    child_sid = f"sess-t12-bf-c-{_uuid.uuid4().hex[:12]}"

    try:
        # Step 1: walk DB down to p1m (T12 column does not exist there).
        command.downgrade(alembic_cfg, PRE_T12_REVISION)

        # Step 2: seed parent + child at the pre-T12 schema. The child has
        # sample_session_id but cannot have tool_filter_preset (column
        # doesn't exist yet). This is exactly the state historical dev /
        # staging DBs will be in when T12 lands.
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO users (id, username, password_hash) "
                "VALUES (:u, :name, 'x')"
            ), {"u": uid, "name": f"t12bf_{uid[:8]}"})
            conn.execute(text(
                "INSERT INTO sessions (id, user_id, status, title) "
                "VALUES (:sid, :u, 'pending', 'parent')"
            ), {"sid": parent_sid, "u": uid})
            conn.execute(text(
                "INSERT INTO sessions (id, user_id, status, title, sample_session_id) "
                "VALUES (:sid, :u, 'pending', 'child', :psid)"
            ), {"sid": child_sid, "u": uid, "psid": parent_sid})

        # Step 3: upgrade to t12 — backfill UPDATE must run before the new
        # cross-column CHECK is added, otherwise this command fails.
        command.upgrade(alembic_cfg, T12_REVISION)

        # Step 4: prove the backfill wrote the canonical preset (not just
        # "anything non-null"). A bug where the UPDATE wrote NULL or an
        # unknown value would fail here.
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT tool_filter_preset FROM sessions WHERE id=:sid"
            ), {"sid": child_sid}).first()
            assert row is not None, "seeded child vanished after upgrade"
            assert row[0] == "subagent_research", (
                f"backfill must have written 'subagent_research', got {row[0]!r}"
            )

            # Symmetric back-compat check: parent row (no sample_session_id)
            # must NOT have been backfilled — only children are touched.
            parent_preset = conn.execute(text(
                "SELECT tool_filter_preset FROM sessions WHERE id=:sid"
            ), {"sid": parent_sid}).scalar()
            assert parent_preset is None, (
                "parent row was incorrectly backfilled — UPDATE WHERE clause "
                "is supposed to scope to sample_session_id IS NOT NULL"
            )
    finally:
        # Restore head schema FIRST so downstream tests run against the
        # right revision even if cleanup hits a snag — schema integrity is
        # more important than seed-row hygiene.
        command.upgrade(alembic_cfg, T12_REVISION)

        # Cleanup order: child → parent → user. Self-FK on sessions is
        # ON DELETE RESTRICT, so parent delete would error while child
        # still references it. UserModel FK is ON DELETE SET NULL on
        # sessions.user_id but explicit ordering keeps the intent clear.
        # No ``except Exception: pass`` — a cleanup failure here means
        # seed rows linger and pollute the test DB; surfacing it loudly
        # is the right behavior (test session is teardown-scoped anyway).
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM sessions WHERE id = :sid"),
                {"sid": child_sid},
            )
            conn.execute(
                text("DELETE FROM sessions WHERE id = :sid"),
                {"sid": parent_sid},
            )
            conn.execute(
                text("DELETE FROM users WHERE id = :u"), {"u": uid},
            )


async def test_child_check_allows_null_preset_on_non_child(db_session):
    """Symmetric to the above: top-level sessions (sample_session_id NULL)
    keep the back-compat right to have NULL preset.

    The CHECK clause is ``sample_session_id IS NULL OR tool_filter_preset
    IS NOT NULL`` — explicit "if non-child, anything goes; if child, must
    have preset". Pin this so a future tightening doesn't accidentally
    break non-child writes.
    """
    uid = str(_uuid.uuid4())
    sid = f"sess-t12-top-{_uuid.uuid4().hex[:12]}"

    await _insert_user(
        db_session, user_id=uid, username=f"t12top_{uid[:8]}"
    )
    await _insert_session(
        db_session,
        session_id=sid,
        user_id=uid,
        title="top-level",
    )
    await db_session.flush()

    res = await db_session.execute(
        text("SELECT tool_filter_preset FROM sessions WHERE id = :sid"),
        {"sid": sid},
    )
    assert res.scalar_one() is None


def test_downgrade_upgrade_roundtrip(alembic_cfg):
    """t12_tool_filter_preset → p1m_sample_session_id → t12_tool_filter_preset.

    Verifies the migration is reversible: downgrade drops the column and
    CHECK; upgrade re-adds them with the exact original shape. The
    `finally` clause restores `head` even on assertion failure so subsequent
    integration tests aren't poisoned by a downgraded DB.
    """
    sync_url = alembic_cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(sync_url)

    try:
        # ---- Downgrade: column + CHECK gone. ----
        command.downgrade(alembic_cfg, PRE_T12_REVISION)
        with engine.connect() as conn:
            col_count = conn.execute(
                text(
                    """
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='sessions'
                      AND column_name='tool_filter_preset'
                    """
                )
            ).scalar()
            assert col_count == 0, (
                "tool_filter_preset column still present after downgrade"
            )

            for _name in (CHECK_NAME, CHILD_CHECK_NAME):
                chk_count = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) FROM pg_constraint c
                        JOIN pg_class t ON t.oid = c.conrelid
                        JOIN pg_namespace n ON n.oid = t.relnamespace
                        WHERE n.nspname='public' AND t.relname='sessions'
                          AND c.conname=:cname AND c.contype='c'
                        """
                    ),
                    {"cname": _name},
                ).scalar()
                assert chk_count == 0, (
                    f"CHECK constraint {_name!r} still present after downgrade"
                )

        # ---- Upgrade back: column + CHECK restored. ----
        command.upgrade(alembic_cfg, T12_REVISION)
        with engine.connect() as conn:
            col_row = conn.execute(
                text(
                    """
                    SELECT is_nullable, data_type, character_maximum_length
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='sessions'
                      AND column_name='tool_filter_preset'
                    """
                )
            ).first()
            assert col_row is not None, (
                "tool_filter_preset column missing after re-upgrade"
            )
            assert col_row[0] == "YES", "tool_filter_preset must be nullable"
            assert col_row[1].lower() == "character varying"
            assert col_row[2] == 64, (
                f"tool_filter_preset must be VARCHAR(64), got length={col_row[2]!r}"
            )

            chk_row = conn.execute(
                text(
                    """
                    SELECT pg_get_constraintdef(c.oid)
                    FROM pg_constraint c
                    JOIN pg_class t ON t.oid = c.conrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    WHERE n.nspname='public' AND t.relname='sessions'
                      AND c.conname=:cname AND c.contype='c'
                    """
                ),
                {"cname": CHECK_NAME},
            ).first()
            assert chk_row is not None, (
                f"CHECK constraint {CHECK_NAME!r} missing after re-upgrade"
            )
            constraintdef = chk_row[0].lower()
            assert "tool_filter_preset" in constraintdef
            assert "subagent_research" in constraintdef

            child_row = conn.execute(
                text(
                    """
                    SELECT pg_get_constraintdef(c.oid)
                    FROM pg_constraint c
                    JOIN pg_class t ON t.oid = c.conrelid
                    JOIN pg_namespace n ON n.oid = t.relnamespace
                    WHERE n.nspname='public' AND t.relname='sessions'
                      AND c.conname=:cname AND c.contype='c'
                    """
                ),
                {"cname": CHILD_CHECK_NAME},
            ).first()
            assert child_row is not None, (
                f"CHECK constraint {CHILD_CHECK_NAME!r} missing after re-upgrade"
            )
            childdef = child_row[0].lower()
            assert "sample_session_id" in childdef
            assert "tool_filter_preset" in childdef
    finally:
        command.upgrade(alembic_cfg, T12_REVISION)
