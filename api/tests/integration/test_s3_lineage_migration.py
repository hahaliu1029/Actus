"""S3 PR-1 (integration, CI-only): backfill invariant on the live test DB.

Runs only where a host Postgres + migrated test DB are available (CI). Asserts
the columns exist after upgrade and the (depth=0) ⇔ (parent IS NULL)
post-backfill invariant holds. Skipped in the local unit run via the marker.
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = pytest.mark.integration


@pytest.mark.anyio
async def test_lineage_columns_present_and_invariant_holds(db_session):
    cols = (
        await db_session.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'sessions' "
                "AND column_name IN ('depth', 'root_session_id')"
            )
        )
    ).scalars().all()
    assert set(cols) == {"depth", "root_session_id"}

    violations = (
        await db_session.execute(
            text(
                "SELECT count(*) FROM sessions "
                "WHERE (depth = 0) <> (parent_session_id IS NULL)"
            )
        )
    ).scalar_one()
    assert violations == 0
