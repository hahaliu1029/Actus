"""[C2 PR-9 §15.2 Gate #7] Partial unique success index lives in DB schema
and blocks duplicate ``status='success'`` rows.

Schema invariant (spec §12.4 r6 + plan §16 r8 P1-6): the
``coordinator_apply_audit`` table carries a *partial* unique index
``ux_apply_audit_run_success`` gated on ``status='success'`` — at most one
success row per ``coordinator_run_id``. Failed / in-progress / rollback_*
rows may repeat freely.

This gate is the PR-9 acceptance test that AST gates cannot cover: it runs
against the migrated DB schema (via the integration fixture stack from
``api/tests/integration/conftest.py``) and:
  1. Confirms ``ux_apply_audit_run_success`` exists with a ``WHERE`` clause
     referencing ``status = 'success'``.
  2. Inserts a row with ``status='success'`` and verifies a second
     duplicate-success insert raises ``IntegrityError``.
  3. Confirms that NON-success statuses (e.g. ``'failed'``) DO permit
     duplicates for the same ``coordinator_run_id``.

Fixture adaptation (vs. plan §15.2 listing):
    The plan's listing used a fixture named ``async_session`` (a callable
    returning a session context manager). That fixture does NOT exist in
    this repo — the canonical async session fixture is ``db_session``,
    pre-bound to a ``begin()...rollback()`` block (see
    ``api/tests/integration/conftest.py:64-71``). We follow the precedent
    set by ``test_c2pr5_apply_audit_migration.py`` (whose tests #3/#4 do
    exactly this work today): use ``db_session``, ``flush()`` instead of
    ``commit()``, and rely on the enclosing rollback for cleanup. The
    ``parent_session_id='sess_parent_test'`` parent-row FK note in the
    plan is moot in this repo because ``coordinator_apply_audit`` does
    **not** carry a FK to ``sessions(id)`` — verified by inspection of
    ``alembic/versions/c2pr5_add_coordinator_apply_audit.py:98``
    (``parent_session_id`` is a plain ``VARCHAR(255)`` column, no
    ``ForeignKey``). No parent-row seeding is needed.

Run: ``cd api && uv run pytest tests/integration/test_coordinator_apply_audit_partial_unique.py -v``
Requires: pgvector-enabled Postgres reachable via ``SQLALCHEMY_DATABASE_URL``
(library ``manus_test``, see ``conftest.py`` docstring).
"""
from __future__ import annotations

import uuid as _uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.anyio, pytest.mark.coordinator_apply]


# Deterministic-but-unique run_id so concurrent CI shards on the same DB
# don't collide on the partial-unique check.
_RUN_ID_DUP = f"r_gate7_dup_{_uuid.uuid4().hex[:8]}"


async def test_partial_unique_success_index_present(db_session) -> None:
    """[Gate #7 part 1] schema check: index exists with WHERE clause referencing 'success'."""
    rows = (await db_session.execute(
        sa.text("""
            SELECT indexdef FROM pg_indexes
            WHERE tablename='coordinator_apply_audit'
              AND indexname='ux_apply_audit_run_success'
        """)
    )).all()
    assert len(rows) == 1, "ux_apply_audit_run_success missing"
    indexdef = rows[0][0]
    assert "WHERE" in indexdef.upper()
    assert "success" in indexdef.lower()


async def test_partial_unique_success_blocks_duplicate_success(db_session) -> None:
    """[r3 P1-6 Gate #7 part 2] Two ``status='success'`` rows with the same
    ``coordinator_run_id`` => second insert raises IntegrityError.
    """
    await db_session.execute(sa.text("""
        INSERT INTO coordinator_apply_audit
          (coordinator_run_id, parent_session_id, status, started_at)
        VALUES (:run_id, 'sess_parent_test', 'success', NOW())
    """), {"run_id": _RUN_ID_DUP})
    await db_session.flush()

    with pytest.raises(IntegrityError):
        await db_session.execute(sa.text("""
            INSERT INTO coordinator_apply_audit
              (coordinator_run_id, parent_session_id, status, started_at)
            VALUES (:run_id, 'sess_parent_test', 'success', NOW())
        """), {"run_id": _RUN_ID_DUP})
        await db_session.flush()


async def test_partial_unique_allows_duplicate_failed(db_session) -> None:
    """[Gate #7 part 3] Three rows with same run_id but ``status='write_io_error'``
    are legal — the partial index ONLY gates on status='success'.

    Uses the real ApplyStatus enum value ``'write_io_error'`` (see
    ``api/app/application/services/patch_applier.py``); the test semantically
    demonstrates "non-success row allows duplicates", and the function name
    ``test_partial_unique_allows_duplicate_failed`` is kept as the conceptual
    label (any non-success status would satisfy the invariant).
    """
    run_id = f"r_gate7_fail_{_uuid.uuid4().hex[:8]}"
    for _ in range(3):
        await db_session.execute(sa.text("""
            INSERT INTO coordinator_apply_audit
              (coordinator_run_id, parent_session_id, status, started_at)
            VALUES (:run_id, 'sess_parent_test', 'write_io_error', NOW())
        """), {"run_id": run_id})
    await db_session.flush()

    count = (await db_session.execute(sa.text("""
        SELECT COUNT(*) FROM coordinator_apply_audit
        WHERE coordinator_run_id=:run_id AND status='write_io_error'
    """), {"run_id": run_id})).scalar_one()
    assert count == 3
