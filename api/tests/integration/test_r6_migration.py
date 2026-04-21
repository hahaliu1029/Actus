"""Integration test for R6 Alembic migration — round-trip + data survival.

Revisions are pinned to explicit IDs (not `head` / `-1`) so that future
revisions landing on top of R6 don't silently turn this into a test of the
wrong boundary. `finally` restores to `head` after every test so a failed
assertion mid-downgrade can't leave the DB in a pre-R6 state and pollute
subsequent integration tests.
"""

import os
import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text


pytestmark = pytest.mark.integration

# Mirror conftest.py fallback so this file is runnable the same way as its
# sibling integration tests (local devs without SQLALCHEMY_DATABASE_URL set
# still hit the shared manus_test default, not a KeyError).
DB_URL = os.environ.get(
    "SQLALCHEMY_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/manus_test",
)

# Pin the boundary under test so a future R7 migration doesn't silently
# shift the "roundtrip" assertions onto the wrong pair of revisions.
R6_REVISION = "r6_user_tool_split"
PRE_R6_REVISION = "r5_add_tool_approval_grants"


def _alembic_cfg() -> Config:
    sync_url = DB_URL.replace("+asyncpg", "+psycopg2", 1)
    api_root = Path(__file__).resolve().parent.parent.parent  # api/
    cfg = Config(str(api_root / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", sync_url)
    return cfg


def _public_tables(engine) -> set[str]:
    with engine.connect() as conn:
        return {
            r[0]
            for r in conn.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            )
        }


def test_r6_upgrade_and_downgrade_roundtrip():
    """R6 → pre-R6 → R6. Verify table structure at each pinned revision."""
    cfg = _alembic_cfg()
    sync_url = cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(sync_url)

    try:
        # Establish a known baseline at R6 (conftest already upgraded to
        # current head; explicitly pin to R6 so any new revision on top of
        # R6 doesn't change what we're about to assert against).
        command.upgrade(cfg, R6_REVISION)

        tables = _public_tables(engine)
        assert "user_tool_enablements" in tables
        assert "user_tool_approval_policies" in tables
        assert "user_tool_preferences" not in tables

        # Downgrade specifically to R5 (not `-1`, which would drift if
        # newer revisions sit above R6).
        command.downgrade(cfg, PRE_R6_REVISION)

        tables = _public_tables(engine)
        assert "user_tool_preferences" in tables
        assert "user_tool_enablements" not in tables
        assert "user_tool_approval_policies" not in tables

        # Upgrade back to R6.
        command.upgrade(cfg, R6_REVISION)

        tables = _public_tables(engine)
        assert "user_tool_enablements" in tables
        assert "user_tool_approval_policies" in tables
    finally:
        # Always restore schema to the real head so a mid-test failure
        # doesn't leave subsequent integration tests running against a
        # downgraded DB.
        command.upgrade(cfg, "head")


def test_r6_preserves_enablement_rows_across_rename():
    """Insert row before downgrade, verify it survives rename roundtrip."""
    cfg = _alembic_cfg()
    sync_url = cfg.get_main_option("sqlalchemy.url")
    engine = create_engine(sync_url)

    # Pin starting revision to R6 explicitly (see module docstring).
    command.upgrade(cfg, R6_REVISION)
    user_id = str(uuid.uuid4())
    row_id = str(uuid.uuid4())
    try:
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT INTO users (id, username, email, password_hash, status, created_at, updated_at) "
                "VALUES (:id, :u, :e, 'x', 'active', NOW(), NOW())"
            ), {"id": user_id, "u": f"u_{user_id}", "e": f"{user_id}@t"})
            conn.execute(text(
                "INSERT INTO user_tool_enablements (id, user_id, tool_type, tool_id, enabled, created_at, updated_at) "
                "VALUES (:rid, :uid, 'mcp', 'github', true, NOW(), NOW())"
            ), {"rid": row_id, "uid": user_id})

        # Downgrade to the pinned pre-R6 revision — row should still exist
        # in the renamed-back `user_tool_preferences` table.
        command.downgrade(cfg, PRE_R6_REVISION)
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM user_tool_preferences WHERE id=:rid"
            ), {"rid": row_id}).scalar()
            assert count == 1

        # Upgrade back to R6 — row should survive in the renamed-forward
        # `user_tool_enablements` table.
        command.upgrade(cfg, R6_REVISION)
        with engine.connect() as conn:
            count = conn.execute(text(
                "SELECT COUNT(*) FROM user_tool_enablements WHERE id=:rid"
            ), {"rid": row_id}).scalar()
            assert count == 1
    finally:
        # Cleanup — FK CASCADE removes the enablement row regardless of
        # which table it currently lives in. Always restore to real head
        # so subsequent tests aren't affected by a pinned-R6 state.
        command.upgrade(cfg, "head")
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM users WHERE id=:id"), {"id": user_id})
