"""Integration tests for c1a_session_tree_expand migration.

Each test inserts real user + parent session rows (FK requirement, per
[[feedback_integration_test_fk]]). Runs against an isolated test PG database
(see api/tests/integration/conftest.py).

Notes on the test transaction model:

- ``db_session`` (conftest.py:64-70) wraps every test in an outer
  ``async with session.begin(): ... await session.rollback()`` block, so
  per-test isolation is guaranteed and tests must NOT call outer
  ``session.commit()`` (it would close the block prematurely and break the
  rollback contract).
- For CHECK / NOT NULL / FK violations and BEFORE-row triggers (here:
  ``trg_sessions_mirror_sample``), ``await db_session.flush()`` is enough
  to surface the error. For IMMEDIATE constraints that must be isolated
  from the outer transaction we use::

      with pytest.raises(IntegrityError):
          async with db_session.begin_nested():
              await db_session.flush()

- For DEFERRABLE INITIALLY DEFERRED constraint triggers (here:
  ``trg_sessions_parent_user_match``), SAVEPOINT release does **not** fire
  deferred triggers — PostgreSQL only evaluates them at outer COMMIT or
  on an explicit ``SET CONSTRAINTS ALL IMMEDIATE`` (PG docs + the
  project's own ``api/tests/integration/test_compaction_recorder_uow.py:3-8``
  comment confirm this). We therefore issue ``SET CONSTRAINTS ALL IMMEDIATE``
  inside the savepoint to force the trigger to evaluate inline; without it
  the trigger queue silently skips evaluation when the savepoint releases
  and ``pytest.raises`` would see ``DID NOT RAISE`` (or the acceptance
  test would trivially pass without actually exercising the trigger).
- The cross-user ``parent_user_match`` blocking case is intentionally split
  out so the cross-user fixture stays test-local instead of leaking through
  the shared ``user_row`` / ``parent_session`` fixtures.
"""
from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa


pytestmark = pytest.mark.integration


@pytest.fixture
async def user_row(db_session):
    """Create a real users row (FK target for sessions.user_id).

    Real ``users`` schema (api/app/infrastructure/models/user.py:13-43): only
    ``id`` is required at the SQL level; ``password_hash`` (NOT
    ``hashed_password``), ``username``, ``email``, ``phone`` are all nullable;
    ``role`` / ``status`` carry server_default; no ``created_at`` /
    ``updated_at`` columns exist.
    """
    user_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO users (id, username, password_hash) "
            "VALUES (:id, :u, 'x')"
        ),
        {"id": user_id, "u": f"u_{user_id[:8]}"},
    )
    await db_session.flush()
    return user_id


@pytest.fixture
async def parent_session(db_session, user_row):
    """Create a real parent sessions row owned by ``user_row``."""
    parent_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, worker_type, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, 'root', '', '', 'PENDING', '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": parent_id, "uid": user_row},
    )
    await db_session.flush()
    return parent_id


@pytest.mark.asyncio
async def test_check_constraint_names_are_clean(db_session):
    """``Base.metadata.naming_convention`` double-prefixed CHECK names in
    T12 / b4m1; C1a uses raw SQL to avoid the same bug. Assert the exact
    canonical names exist on the ``sessions`` table.
    """
    rows = (await db_session.execute(
        sa.text(
            "SELECT conname FROM pg_constraint "
            " WHERE conrelid = 'sessions'::regclass "
            "   AND conname IN ("
            "     'ck_sessions_worker_type', "
            "     'ck_sessions_worker_type_parent_invariant', "
            "     'ck_sessions_no_self_parent'"
            "   )"
        )
    )).all()
    names = {r.conname for r in rows}
    assert names == {
        "ck_sessions_worker_type",
        "ck_sessions_worker_type_parent_invariant",
        "ck_sessions_no_self_parent",
    }


@pytest.mark.asyncio
async def test_mirror_trigger_derives_worker_type_on_insert(
    db_session, user_row, parent_session
):
    """Old pod path: INSERT child with sample_session_id only (no worker_type)
    → mirror trigger auto-derives ``parent_session_id`` AND
    ``worker_type='subagent'``.
    """
    child_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, sample_session_id, tool_filter_preset, "
            "  title, latest_message, status, events, files, memories) "
            "VALUES (:id, :uid, :pid, 'subagent_research', '', '', 'PENDING', "
            "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": child_id, "uid": user_row, "pid": parent_session},
    )
    await db_session.flush()
    row = (await db_session.execute(
        sa.text(
            "SELECT parent_session_id, worker_type FROM sessions WHERE id = :id"
        ),
        {"id": child_id},
    )).one()
    assert row.parent_session_id == parent_session
    assert row.worker_type == "subagent"


@pytest.mark.asyncio
async def test_mirror_trigger_raises_on_two_column_conflict(
    db_session, user_row, parent_session
):
    """If a caller writes both ``sample_session_id`` and ``parent_session_id``
    with conflicting values, the BEFORE-row mirror trigger raises 23514
    with "must be equal during expand-contract" — the dual-write safety net.
    """
    other_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, worker_type, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, 'root', '', '', 'PENDING', '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": other_id, "uid": user_row},
    )
    await db_session.flush()
    with pytest.raises(sa.exc.IntegrityError, match="must be equal during expand-contract"):
        async with db_session.begin_nested():
            await db_session.execute(
                sa.text(
                    "INSERT INTO sessions (id, user_id, sample_session_id, parent_session_id, "
                    "  tool_filter_preset, title, latest_message, status, "
                    "  events, files, memories) "
                    "VALUES (:id, :uid, :sp, :pp, 'subagent_research', '', '', 'PENDING', "
                    "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "uid": user_row,
                    "sp": parent_session,
                    "pp": other_id,
                },
            )
            await db_session.flush()


@pytest.mark.asyncio
async def test_mirror_trigger_propagates_null_on_update(
    db_session, user_row, parent_session
):
    """If ``sample_session_id`` is set NULL on UPDATE, ``parent_session_id``
    must also become NULL and ``worker_type`` must fall back to 'root'.
    The cross-column CHECK ``ck_sessions_worker_type_parent_invariant``
    requires the worker_type rewrite to be atomic with the parent NULL-out.
    """
    child_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
            "  tool_filter_preset, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
            "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": child_id, "uid": user_row, "pid": parent_session},
    )
    await db_session.flush()
    # Must temporarily relax the child-must-have-preset CHECK by also nulling preset
    await db_session.execute(
        sa.text(
            "UPDATE sessions SET sample_session_id = NULL, tool_filter_preset = NULL "
            " WHERE id = :id"
        ),
        {"id": child_id},
    )
    await db_session.flush()
    row = (await db_session.execute(
        sa.text(
            "SELECT parent_session_id, worker_type FROM sessions WHERE id = :id"
        ),
        {"id": child_id},
    )).one()
    assert row.parent_session_id is None
    assert row.worker_type == "root"


@pytest.mark.asyncio
async def test_parent_user_match_trigger_blocks_cross_user_child(db_session):
    """INSERT a child belonging to user B referencing a parent owned by user A
    → DEFERRED constraint trigger ``trg_sessions_parent_user_match`` raises
    23514 "parent must belong to same user".

    The trigger is ``DEFERRABLE INITIALLY DEFERRED``, so SAVEPOINT release
    does NOT fire it — we force evaluation with ``SET CONSTRAINTS ALL
    IMMEDIATE`` inside the savepoint. Without that the trigger would queue
    until outer COMMIT (which the conftest rollback never reaches), and
    ``pytest.raises`` would silently see ``DID NOT RAISE``.
    """
    user_a = uuid.uuid4().hex
    user_b = uuid.uuid4().hex
    parent_id = uuid.uuid4().hex
    for uid in (user_a, user_b):
        await db_session.execute(
            sa.text(
                "INSERT INTO users (id, username, password_hash) "
                "VALUES (:id, :u, 'x')"
            ),
            {"id": uid, "u": f"u_{uid[:8]}"},
        )
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, worker_type, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, 'root', '', '', 'PENDING', '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": parent_id, "uid": user_a},
    )
    await db_session.flush()
    with pytest.raises(sa.exc.IntegrityError, match="parent must belong to same user"):
        async with db_session.begin_nested():
            await db_session.execute(
                sa.text(
                    "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
                    "  tool_filter_preset, title, latest_message, status, "
                    "  events, files, memories) "
                    "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
                    "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
                ),
                {
                    "id": uuid.uuid4().hex,
                    "uid": user_b,
                    "pid": parent_id,
                },
            )
            await db_session.flush()
            # Force DEFERRED trg_sessions_parent_user_match to fire NOW
            # within the savepoint — SAVEPOINT release does not fire
            # deferred constraint triggers; only outer COMMIT or
            # SET CONSTRAINTS ALL IMMEDIATE does.
            await db_session.execute(sa.text("SET CONSTRAINTS ALL IMMEDIATE"))


@pytest.mark.asyncio
async def test_parent_user_match_allows_cascade_null(
    db_session, user_row, parent_session
):
    """DELETE user → cascade SET NULL on both parent and child ``user_id``;
    the DEFERRABLE trigger must NOT raise because both rows end up with
    ``user_id`` NULL (``parent_user IS NOT DISTINCT FROM child_user`` is
    true on the NULL/NULL terminus).

    We explicitly issue ``SET CONSTRAINTS ALL IMMEDIATE`` after the delete
    so the deferred trigger actually evaluates the post-cascade NULL/NULL
    state inline (the savepoint release path does NOT fire it). Without
    that, the trigger would queue until the outer COMMIT — which never
    arrives because the conftest rolls back — and the test would
    trivially pass without proving the trigger accepts the terminus.
    """
    child_id = uuid.uuid4().hex
    await db_session.execute(
        sa.text(
            "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
            "  tool_filter_preset, title, latest_message, status, "
            "  events, files, memories) "
            "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
            "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
        ),
        {"id": child_id, "uid": user_row, "pid": parent_session},
    )
    await db_session.flush()
    # Wrap the cascade + forced trigger fire in a savepoint so any unexpected
    # trigger rejection (real bug) rolls back to the savepoint instead of
    # tainting the outer test transaction. Mirrors test #5's begin_nested
    # discipline so both DEFERRED-trigger tests look structurally identical.
    async with db_session.begin_nested():
        await db_session.execute(
            sa.text("DELETE FROM users WHERE id = :id"), {"id": user_row}
        )
        await db_session.flush()
        # Force the DEFERRED trigger to evaluate the post-cascade NULL/NULL
        # terminus inline. If the trigger were going to reject (real bug),
        # this would raise; the test then fails loudly here instead of
        # silently passing on a trigger that never fired.
        await db_session.execute(sa.text("SET CONSTRAINTS ALL IMMEDIATE"))
    rows = (await db_session.execute(
        sa.text("SELECT user_id FROM sessions WHERE id IN (:p, :c)"),
        {"p": parent_session, "c": child_id},
    )).all()
    assert all(r.user_id is None for r in rows)
