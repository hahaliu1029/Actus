"""B3-core PR-0: 5 PG schema/CHECK/index anchors (xfail).

Spec v3 §3.1 + §8.1 (group C-PG-*).

PR-2 ``b3p2_add_session_supervisor_columns`` migration ships the schema.
These anchors flip from ``xfail`` → ``xpass`` when the migration is
applied.

Spec basis: docs/superpowers/specs/2026-05-07-b3-core-design.md
Plan basis: docs/superpowers/plans/2026-05-07-b3-core-pr0-plan.md (Task 4).
"""
from __future__ import annotations

import pytest
from sqlalchemy import text

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


# -- C-PG-1: 9 supervisor columns exist on `sessions` table --------------------
@pytest.mark.xfail(strict=False, reason="PR-2 b3p2 migration not yet applied")
async def test_C_PG_1_supervisor_columns_exist(db_session):
    expected = {
        "execution_mode", "background_reason", "expires_at", "last_activity_at",
        "execution_phase", "retry_budget_remaining", "terminal_reason",
        "suspended_reason", "was_background",
    }
    result = await db_session.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='sessions' AND column_name = ANY(:names)"
    ), {"names": list(expected)})
    actual = {row[0] for row in result}
    assert expected == actual, f"Missing supervisor columns: {expected - actual}"


# -- C-PG-2: 11 supervisor CHECK constraints exist -----------------------------
# Round-3 audit P1#7 fix: assert all 11 supervisor CHECK constraints, not 6.
# Spec v3 §3.1 lines 67-82 enumerates 11 CHECKs.  Using subset tolerated only
# 6 — could silently miss the 5 mode/phase coupling CHECKs.
@pytest.mark.xfail(strict=False, reason="PR-2 CHECK constraints not yet applied")
async def test_C_PG_2_eleven_check_constraints(db_session):
    result = await db_session.execute(text(
        "SELECT conname FROM pg_constraint "
        "WHERE conrelid = 'sessions'::regclass AND contype = 'c' "
        # Round-2 audit P1-B fix: PR-2 emits explicit `ck_sessions_*` names
        # via alembic name= kwarg (not the default `sessions_*_check` auto-name).
        "AND conname LIKE 'ck_sessions_%'"
    ))
    names = {row[0] for row in result}
    # All 11 CHECK constraints per PR-2 plan §Task 1 (line 385-440).
    # Names use the `ck_sessions_*` convention (alembic explicit name= kwarg),
    # NOT the default `{table}_{column}_check` PostgreSQL auto-naming.
    # Round-2 audit P1-B fix.
    expected_full = {
        # Enum-value CHECKs (6):
        "ck_sessions_execution_mode",
        "ck_sessions_background_reason",
        "ck_sessions_execution_phase",
        "ck_sessions_retry_budget_range",
        "ck_sessions_terminal_reason",
        "ck_sessions_suspended_reason",
        # Mode/phase coupling CHECKs (5):
        "ck_sessions_bg_requires_expires_at",
        "ck_sessions_bg_requires_reason",
        "ck_sessions_fg_no_bg_fields",
        "ck_sessions_suspended_requires_reason",
        "ck_sessions_terminal_requires_reason",
    }
    assert expected_full.issubset(names), (
        f"Missing CHECKs (expected 11, got {len(names & expected_full)}): "
        f"{expected_full - names}"
    )
    assert len(expected_full) == 11, "spec §3.1 expected_full set drift; should sum to 11"


# -- C-PG-3: 3 partial indexes exist with correct WHERE clauses ----------------
# Round-3 audit P1#7 fix: assert WHERE clauses match spec §3.1 (lines 87-99),
# not just index names.  Prior version verified names but left WHERE clause
# correctness uncovered — a future migration could land an index with the
# right name but the wrong predicate and pass.
@pytest.mark.xfail(strict=False, reason="PR-2 partial indexes not yet applied")
async def test_C_PG_3_three_partial_indexes(db_session):
    result = await db_session.execute(text(
        "SELECT indexname, indexdef FROM pg_indexes "
        "WHERE tablename='sessions' AND indexdef LIKE '%WHERE%'"
    ))
    rows = {row[0]: row[1] for row in result}

    # 3 partial indexes per spec v3 §3.1 (lines 87-99):
    # idx_sessions_user_bg_recent: (user_id, last_activity_at DESC) WHERE execution_mode='background'
    # idx_sessions_expires_at: (expires_at) WHERE expires_at IS NOT NULL
    # idx_sessions_phase_running: (execution_phase, status) WHERE execution_phase IN ('running', 'recovering')

    expected_indexes = {
        "idx_sessions_user_bg_recent": ("user_id", "last_activity_at", "execution_mode", "background"),
        "idx_sessions_expires_at": ("expires_at", "IS NOT NULL"),
        "idx_sessions_phase_running": ("execution_phase", "status", "running", "recovering"),
    }

    missing = set(expected_indexes) - set(rows)
    assert not missing, f"missing partial indexes: {missing}"

    for idx_name, expected_substrs in expected_indexes.items():
        indexdef = rows[idx_name]
        for substr in expected_substrs:
            assert substr in indexdef, (
                f"partial index {idx_name} indexdef missing expected fragment {substr!r}; "
                f"got: {indexdef}"
            )


# -- C-PG-4: T7 transition does NOT clear was_background -----------------------
# Round-6 audit P2 fix: prior body had `# ... (full setup) ...` placeholder
# without ever inserting a real session row.  `update_supervisor_fields` on a
# non-existent UUID would silently affect 0 rows; subsequent `get_by_id` returns
# None, leading to AttributeError on `fresh.execution_mode` — erratic xfail
# signal.  Converted to explicit placeholder.
@pytest.mark.xfail(strict=False, reason="PR-2 T7 reconnect path not yet implemented")
async def test_C_PG_4_t7_preserves_was_background(session_repo, sample_user):
    import pytest as _pytest

    _pytest.fail(
        "placeholder — flip when PR-2 ships T7 reconnect path. "
        "Implementation must:\n"
        "  (1) INSERT a real session row for sample_user (status=running, "
        "execution_mode=background, was_background=True, "
        "expires_at=NOW+2h, background_reason='explicit')\n"
        "  (2) invoke T7 transition: `update_supervisor_fields(sid, "
        "execution_mode='foreground', background_reason=None, expires_at=None)` "
        "(was_background NOT passed → stays True)\n"
        "  (3) re-read row: assert execution_mode='foreground'\n"
        "  (4) assert was_background is True (PERSISTENT flag survives T7 — "
        "spec v3 §3.1 + round-2 P0-2 fix)\n"
        "  (5) optionally assert background_reason is None and expires_at is None\n"
        "Per spec v3 §3.1 + round-2 P0-2."
    )


# -- C-PG-5: Terminal status implies execution_phase='terminated' --------------
@pytest.mark.xfail(strict=False, reason="PR-2 update_to_terminal not yet implemented")
async def test_C_PG_5_terminal_status_phase_invariant(session_repo, sample_session):
    from app.domain.models.session import SessionStatus

    await session_repo.update_to_terminal(
        sample_session.id, SessionStatus.TIMED_OUT,
        terminal_reason="watchdog_timeout",
    )
    fresh = await session_repo.get_by_id(sample_session.id)
    assert fresh.status == SessionStatus.TIMED_OUT
    assert fresh.execution_phase == "terminated"
    assert fresh.terminal_reason == "watchdog_timeout"
