"""C2 PR-5 Task 5.5 — coordinator_apply_audit migration integration tests.

Spec ref: §12.4 P0-8 + r6 (partial unique success index).

These tests verify:
- All spec'd columns are present
- ``ix_apply_audit_run`` exists (non-unique lookup)
- ``ux_apply_audit_run_success`` is a UNIQUE partial index gated on
  ``status='success'`` — duplicates rejected only when status='success'.

These tests require the CI Postgres container (or a local test-only
Postgres pointing at ``manus_test``) — they CANNOT run against the
compose ``manus`` dev database (would pollute live data).
"""
from __future__ import annotations

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


# Use a deterministic-but-unique run_id so concurrent CI shards on the
# same DB don't collide on the partial-unique test.
_RUN_ID_SUCCESS = "r_test_c2pr5_success"
_RUN_ID_FAILED = "r_test_c2pr5_failed"


async def test_table_present(db_session) -> None:
    cols = (await db_session.execute(
        sa.text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='coordinator_apply_audit'
        """)
    )).all()
    column_names = {row[0] for row in cols}
    expected = {
        "id", "coordinator_run_id", "parent_session_id", "status",
        # [codex R6 P2] plan_hash is the PR-7 rehydrate idempotency
        # key (SHA-256 hex of the canonical PatchApplyPlan JSON).
        "plan_hash",
        "file_count", "total_bytes", "failed_at_path", "failed_reason",
        "rollback_status", "rollback_failed_paths",
        "started_at", "finished_at", "duration_ms",
        "applied_files", "plan_files_preview", "diagnostics",
    }
    missing = expected - column_names
    assert not missing, f"missing columns: {missing}"


async def test_nonunique_index_on_run_id(db_session) -> None:
    """``ix_apply_audit_run`` is the lookup-only index (multiple rows
    per run_id are legal — failed/in_progress/rollback_* repeat)."""
    rows = (await db_session.execute(
        sa.text("""
            SELECT indexdef FROM pg_indexes
            WHERE tablename='coordinator_apply_audit'
              AND indexname='ix_apply_audit_run'
        """)
    )).all()
    assert len(rows) == 1, "ix_apply_audit_run missing"
    indexdef = rows[0][0]
    assert "UNIQUE" not in indexdef.upper(), (
        "ix_apply_audit_run must NOT be unique — multiple rows per "
        f"run_id are legal: {indexdef}"
    )


async def test_partial_unique_success_index_exists(db_session) -> None:
    rows = (await db_session.execute(
        sa.text("""
            SELECT indexdef FROM pg_indexes
            WHERE tablename='coordinator_apply_audit'
              AND indexname='ux_apply_audit_run_success'
        """)
    )).all()
    assert len(rows) == 1, "ux_apply_audit_run_success missing"
    indexdef = rows[0][0]
    assert "UNIQUE" in indexdef.upper()
    # Partial WHERE — case-insensitive match on the predicate
    assert "status" in indexdef.lower()
    assert "success" in indexdef.lower()


async def test_partial_unique_blocks_duplicate_success_row(db_session) -> None:
    """[spec §12.4 r6] At most one ``status='success'`` row per run_id.

    A second insert with the same coordinator_run_id and status='success'
    must raise IntegrityError. The transactional db_session rolls back,
    so this leaves no residue."""
    await db_session.execute(sa.text("""
        INSERT INTO coordinator_apply_audit
          (coordinator_run_id, parent_session_id, status, started_at)
        VALUES (:run_id, 'sess_parent_test', 'success', NOW())
    """), {"run_id": _RUN_ID_SUCCESS})
    await db_session.flush()

    with pytest.raises(IntegrityError):
        await db_session.execute(sa.text("""
            INSERT INTO coordinator_apply_audit
              (coordinator_run_id, parent_session_id, status, started_at)
            VALUES (:run_id, 'sess_parent_test', 'success', NOW())
        """), {"run_id": _RUN_ID_SUCCESS})
        await db_session.flush()


async def test_failed_status_allows_duplicates(db_session) -> None:
    """The partial unique index ONLY constrains status='success'. Three
    'write_io_error' rows for the same run_id are legal (each retry attempts)."""
    for _ in range(3):
        await db_session.execute(sa.text("""
            INSERT INTO coordinator_apply_audit
              (coordinator_run_id, parent_session_id, status, started_at)
            VALUES (:run_id, 'sess_parent_test', 'write_io_error', NOW())
        """), {"run_id": _RUN_ID_FAILED})
    await db_session.flush()

    count = (await db_session.execute(sa.text("""
        SELECT COUNT(*) FROM coordinator_apply_audit
        WHERE coordinator_run_id=:run_id AND status='write_io_error'
    """), {"run_id": _RUN_ID_FAILED})).scalar_one()
    assert count == 3
