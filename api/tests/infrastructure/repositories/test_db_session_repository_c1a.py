"""Unit-level tests for C1a additions on DBSessionRepository.

These use mocked SQLAlchemy AsyncSession; real-PG behavior is covered by
api/tests/integration/test_session_service_c1a_concurrency.py.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.infrastructure.repositories.db_session_repository import DBSessionRepository


@pytest.mark.anyio
async def test_find_by_id_for_user_returns_none_when_missing():
    mock_session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result
    repo = DBSessionRepository(db_session=mock_session)

    out = await repo.find_by_id_for_user("missing", user_id="u1")
    assert out is None


@pytest.mark.anyio
async def test_find_descendants_empty_returns_empty_list():
    mock_session = AsyncMock()
    rows_result = MagicMock()
    rows_result.all.return_value = []
    mock_session.execute.return_value = rows_result
    repo = DBSessionRepository(db_session=mock_session)

    out = await repo.find_descendants(
        "ancestor", user_id="u1", max_depth=1, limit=11
    )
    assert out == []
    # exactly one CTE query when no rows
    assert mock_session.execute.await_count == 1


@pytest.mark.anyio
async def test_count_descendants_uses_limit_subquery():
    """Implementation must wrap CTE in a LIMIT subquery and count *that*."""
    mock_session = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 11
    mock_session.execute.return_value = result
    repo = DBSessionRepository(db_session=mock_session)

    out = await repo.count_descendants("ancestor", user_id="u1", cap=10)
    assert out == 11
    sql_arg = mock_session.execute.await_args[0][0]
    # sa.text() instances expose `.text`
    sql_text = getattr(sql_arg, "text", str(sql_arg))
    assert "LIMIT" in sql_text


@pytest.mark.anyio
async def test_lock_session_for_spawn_pushes_user_id_into_sql_where():
    """Defense-in-depth: SELECT ... FOR UPDATE must filter by both id AND user_id
    so a cross-tenant parent_id never acquires a row lock (closes a brief
    DoS window where a foreign user_id could pin a victim's row before the
    application-layer owner check rejects)."""
    mock_session = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = result
    repo = DBSessionRepository(db_session=mock_session)

    out = await repo.lock_session_for_spawn("p", user_id="u1")
    assert out is None
    stmt = mock_session.execute.await_args[0][0]

    # Compile WHERE only — the full SELECT column list happens to include
    # ``sessions.user_id`` as a column, so asserting "user_id" against the
    # whole statement would pass even if the owner predicate were removed.
    where_clause = stmt.whereclause
    assert where_clause is not None, "lock_session_for_spawn must have a WHERE clause"
    where_sql = str(
        where_clause.compile(compile_kwargs={"literal_binds": True})
    )
    assert "sessions.id" in where_sql, where_sql
    assert "sessions.user_id" in where_sql, where_sql
    assert "'p'" in where_sql, where_sql
    assert "'u1'" in where_sql, where_sql

    # FOR UPDATE belongs on the full statement, not the WHERE expression.
    full_sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
    assert "FOR UPDATE" in full_sql.upper(), full_sql
