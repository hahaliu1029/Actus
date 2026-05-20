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
