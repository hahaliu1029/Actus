"""C1a regression: DEFERRABLE trigger violations at COMMIT must surface to caller.

Pre-fix: DBUnitOfWork.__aexit__ swallowed IntegrityError, caller saw "ok" but
the row never landed. C1a `trg_sessions_parent_user_match` is DEFERRABLE INITIALLY
DEFERRED, so the violation fires inside `commit()` - must propagate.
"""
from __future__ import annotations

import uuid

import pytest
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
async def two_users(async_session_factory):
    user_a, user_b = uuid.uuid4().hex, uuid.uuid4().hex
    async with async_session_factory() as session:
        for uid in (user_a, user_b):
            await session.execute(
                sa.text(
                    "INSERT INTO users (id, username, password_hash) "
                    "VALUES (:id, :u, 'x')"
                ),
                {"id": uid, "u": f"u_{uid[:8]}"},
            )
        await session.commit()
    try:
        yield user_a, user_b
    finally:
        async with async_session_factory() as session:
            await session.execute(
                sa.text("DELETE FROM users WHERE id IN (:a, :b)"),
                {"a": user_a, "b": user_b},
            )
            await session.commit()


async def test_deferred_trigger_violation_propagates(uow_factory, two_users):
    user_a, user_b = two_users
    parent_id = uuid.uuid4().hex
    child_id = uuid.uuid4().hex

    with pytest.raises(IntegrityError, match="parent must belong to same user"):
        async with uow_factory() as uow:
            await uow.db_session.execute(
                sa.text(
                    "INSERT INTO sessions (id, user_id, worker_type, title, latest_message, status, "
                    "  events, files, memories) "
                    "VALUES (:id, :uid, 'root', '', '', 'PENDING', '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
                ),
                {"id": parent_id, "uid": user_a},
            )
            await uow.db_session.execute(
                sa.text(
                    "INSERT INTO sessions (id, user_id, parent_session_id, worker_type, "
                    "  tool_filter_preset, title, latest_message, status, "
                    "  events, files, memories) "
                    "VALUES (:id, :uid, :pid, 'subagent', 'subagent_research', '', '', 'PENDING', "
                    "  '[]'::jsonb, '[]'::jsonb, '{}'::jsonb)"
                ),
                {"id": child_id, "uid": user_b, "pid": parent_id},
            )
            # Violation must surface from __aexit__'s commit, not from raise above.
