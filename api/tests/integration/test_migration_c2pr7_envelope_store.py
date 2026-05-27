"""C2 PR-7: integration tests for ``c2pr7_add_coordinator_result_envelope_store``.

Verifies the post-upgrade DB schema contract added by
``c2pr7_envelope_store``:

  * 7 columns on ``coordinator_result_envelope_store`` — ``id`` (PK),
    ``coordinator_run_id``, ``work_unit_id``, ``child_session_id``,
    ``envelope_type``, ``payload`` (JSONB), ``received_at`` (TIMESTAMP TZ).
  * Unique index ``ux_result_store_run_wu_terminal`` on
    (coordinator_run_id, work_unit_id) — a second INSERT with the same
    key must raise IntegrityError.

Conventions:
  * Fixture: ``db_session`` from ``api/tests/integration/conftest.py``
    (auto-rollback per test).
  * Cleanup: each test uses a unique row prefix (``r_pr7_t1`` / ``r_pr7_t2``)
    and explicitly DELETEs at start and end so a previously crashed run
    doesn't leak rows into the test. The ``db_session`` rollback would
    normally cover this, but the UNIQUE-violation test commits a savepoint
    rollback that can interact with the outer transaction in subtle ways.

Run: ``cd api && uv run pytest tests/integration/test_migration_c2pr7_envelope_store.py -v``
Requires: pgvector-enabled Postgres reachable via ``SQLALCHEMY_DATABASE_URL``.
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


async def test_table_present(db_session) -> None:
    """[codex R2 P2] All 7 columns of ``coordinator_result_envelope_store``
    must exist with the expected types (BIGINT pk + 4 strings +
    JSONB payload + TIMESTAMPTZ received_at)."""
    result = await db_session.execute(
        text(
            """
            SELECT column_name, data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = 'coordinator_result_envelope_store'
            """
        )
    )
    rows = result.fetchall()
    col_to_type = {r[0]: (r[1], r[2]) for r in rows}

    # All 7 declared columns must be present (id + 4 string cols + payload + received_at).
    expected_columns = {
        "id",
        "coordinator_run_id",
        "work_unit_id",
        "child_session_id",
        "envelope_type",
        "payload",
        "received_at",
    }
    assert expected_columns == set(col_to_type.keys()), (
        f"column set mismatch: expected {expected_columns}, "
        f"got {set(col_to_type.keys())}"
    )

    # ``id`` is BIGINT primary key (BigInteger maps to data_type='bigint').
    assert col_to_type["id"][0] == "bigint"

    # ``payload`` is JSONB (postgres-specific udt_name='jsonb').
    assert col_to_type["payload"][1] == "jsonb", (
        f"payload must be JSONB; got {col_to_type['payload']}"
    )

    # ``received_at`` is TIMESTAMP WITH TIME ZONE.
    assert col_to_type["received_at"][0] == "timestamp with time zone", (
        f"received_at must be timestamptz; got {col_to_type['received_at']}"
    )

    # The 4 string columns are VARCHAR (data_type='character varying').
    for str_col in ("coordinator_run_id", "work_unit_id",
                     "child_session_id", "envelope_type"):
        assert col_to_type[str_col][0] == "character varying", (
            f"{str_col} must be VARCHAR; got {col_to_type[str_col]}"
        )


async def test_unique_per_run_wu(db_session) -> None:
    """Second INSERT with the same (run_id, wu_id) must raise IntegrityError.

    Uses a unique row prefix and DELETE-before / DELETE-after to keep the
    integration DB clean between runs even if a prior test crash leaked
    rows (the ``db_session`` rollback already covers the happy path).
    """
    run_id_1 = "r_pr7_t1"
    run_id_2 = "r_pr7_t2"
    wu_id = "wu_pr7_x"

    # Cleanup any leftover rows from a previously-crashed run.
    await db_session.execute(
        text(
            "DELETE FROM coordinator_result_envelope_store "
            "WHERE coordinator_run_id IN (:r1, :r2)"
        ),
        {"r1": run_id_1, "r2": run_id_2},
    )

    try:
        # First insert — succeeds.
        await db_session.execute(
            text(
                """
                INSERT INTO coordinator_result_envelope_store
                    (coordinator_run_id, work_unit_id, child_session_id,
                     envelope_type, payload)
                VALUES (:rid, :wid, :csid, :etype,
                        CAST(:payload AS JSONB))
                """
            ),
            {
                "rid": run_id_1,
                "wid": wu_id,
                "csid": "child_pr7_a",
                "etype": "RESULT_READY",
                "payload": '{"outcome": "success"}',
            },
        )
        await db_session.flush()

        # Second insert with same (run_id, wu_id) — must raise.
        with pytest.raises(IntegrityError):
            await db_session.execute(
                text(
                    """
                    INSERT INTO coordinator_result_envelope_store
                        (coordinator_run_id, work_unit_id, child_session_id,
                         envelope_type, payload)
                    VALUES (:rid, :wid, :csid, :etype,
                            CAST(:payload AS JSONB))
                    """
                ),
                {
                    "rid": run_id_1,
                    "wid": wu_id,
                    "csid": "child_pr7_b",
                    "etype": "CANCEL_ACK",
                    "payload": '{"outcome": "cancelled"}',
                },
            )
            await db_session.flush()
    finally:
        # Cleanup — best-effort, the outer ``db_session`` rollback will
        # also wipe everything but we want explicit cleanup to keep the
        # invariant tight if a future test author changes the fixture
        # to commit.
        try:
            await db_session.rollback()
        except Exception:
            pass
        await db_session.execute(
            text(
                "DELETE FROM coordinator_result_envelope_store "
                "WHERE coordinator_run_id IN (:r1, :r2)"
            ),
            {"r1": run_id_1, "r2": run_id_2},
        )
