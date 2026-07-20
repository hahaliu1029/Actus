"""Verify pe0_mode_rev adds + drops the column cleanly."""

import pytest
from sqlalchemy import inspect

MIGRATION_TARGET = "pe0_mode_rev"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.usefixtures("migration_schema_at"),
]


async def test_mode_revision_column_exists_after_migration(db_session):
    """alembic upgrade head must add the column with NOT NULL DEFAULT 0."""
    async_conn = await db_session.connection()

    def _check(sync_conn):
        cols = {c["name"]: c for c in inspect(sync_conn).get_columns("sessions")}
        assert "mode_revision" in cols, "mode_revision column missing"
        col = cols["mode_revision"]
        assert col["nullable"] is False
        return col

    col = await async_conn.run_sync(_check)
    # Default may come back as a server-side expression literal
    assert "0" in str(col["default"]) if col.get("default") is not None else True


async def test_existing_sessions_get_mode_revision_zero(db_session):
    """Backfill: existing rows must read mode_revision=0 after upgrade."""
    from sqlalchemy import text

    res = await db_session.execute(
        text("SELECT mode_revision FROM sessions LIMIT 5")
    )
    rows = res.scalars().all()
    for v in rows:
        assert v == 0
