"""Unit tests for DBMemoryChunkRepository — conversion helpers + empty batch."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.models.memory_chunk_orm import MemoryChunkModel
from app.infrastructure.repositories.db_memory_chunk_repository import DBMemoryChunkRepository

from tests.conftest import TEST_OTHER_USER_ID_FIXED, TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


def _make_orm_row(
    *,
    embedding: list[float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a MagicMock that mimics MemoryChunkModel attributes."""
    row = MagicMock(spec=MemoryChunkModel)
    row.id = "chunk-1"
    row.user_id = TEST_USER_ID_FIXED
    row.session_id = "sess-1"
    row.content = "test content"
    row.content_hash = "hash123"
    row.embedding = embedding
    row.source = "session_flush"
    row.metadata_ = metadata if metadata is not None else {}
    row.created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    row.updated_at = datetime(2026, 1, 2, tzinfo=timezone.utc)
    return row


def _make_domain_chunk(**overrides: Any) -> MemoryChunk:
    defaults = dict(
        id="chunk-1",
        user_id=TEST_USER_ID_FIXED,
        session_id="sess-1",
        content="test content",
        content_hash="hash123",
        source="session_flush",
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MemoryChunk(**defaults)


class TestToDomain:
    """_to_domain: ORM row → domain MemoryChunk."""

    def test_basic_conversion(self) -> None:
        row = _make_orm_row()
        result = DBMemoryChunkRepository._to_domain(row)

        assert isinstance(result, MemoryChunk)
        assert result.id == "chunk-1"
        assert result.user_id == TEST_USER_ID_FIXED
        assert result.session_id == "sess-1"
        assert result.content == "test content"
        assert result.content_hash == "hash123"
        assert result.source == "session_flush"
        assert result.metadata == {}
        assert result.embedding is None

    def test_embedding_converted_to_tuple(self) -> None:
        """pgvector returns list/ndarray; _to_domain converts to tuple."""
        row = _make_orm_row(embedding=[0.1, 0.2, 0.3])
        result = DBMemoryChunkRepository._to_domain(row)

        assert result.embedding == (0.1, 0.2, 0.3)
        assert isinstance(result.embedding, tuple)

    def test_none_embedding_preserved(self) -> None:
        row = _make_orm_row(embedding=None)
        result = DBMemoryChunkRepository._to_domain(row)
        assert result.embedding is None

    def test_metadata_deep_copied(self) -> None:
        """metadata must be deep-copied to cut ORM managed dict reference."""
        nested = {"tools": ["search", "browse"], "config": {"verbose": True}}
        row = _make_orm_row(metadata=nested)
        result = DBMemoryChunkRepository._to_domain(row)

        # Top-level identity must differ
        assert result.metadata is not row.metadata_
        # Nested list identity must also differ (deep copy)
        assert result.metadata["tools"] is not nested["tools"]
        # Values must be equal
        assert result.metadata == nested


class TestToOrmDict:
    """_to_orm_dict: domain MemoryChunk → dict for pg_insert."""

    def test_basic_conversion(self) -> None:
        chunk = _make_domain_chunk()
        result = DBMemoryChunkRepository._to_orm_dict(chunk)

        assert isinstance(result, dict)
        assert result["id"] == "chunk-1"
        assert result["user_id"] == TEST_USER_ID_FIXED
        assert result["content"] == "test content"
        assert result["source"] == "session_flush"
        # Key must be Python attribute name, NOT DB column name
        assert "metadata_" in result
        assert "metadata" not in result

    def test_embedding_converted_to_list(self) -> None:
        chunk = _make_domain_chunk(embedding=(0.1, 0.2, 0.3))
        result = DBMemoryChunkRepository._to_orm_dict(chunk)

        assert result["embedding"] == [0.1, 0.2, 0.3]
        assert isinstance(result["embedding"], list)

    def test_none_embedding_preserved(self) -> None:
        chunk = _make_domain_chunk(embedding=None)
        result = DBMemoryChunkRepository._to_orm_dict(chunk)
        assert result["embedding"] is None


class TestGetById:
    """get_by_id: user_id isolation + found/not-found branches."""

    async def test_found_returns_domain_chunk(self) -> None:
        row = _make_orm_row()
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = row

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        result = await repo.get_by_id("chunk-1", user_id=TEST_USER_ID_FIXED)

        assert isinstance(result, MemoryChunk)
        assert result.id == "chunk-1"
        mock_session.execute.assert_awaited_once()

    async def test_not_found_returns_none(self) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        result = await repo.get_by_id("nonexistent", user_id=TEST_USER_ID_FIXED)

        assert result is None

    async def test_wrong_user_returns_none(self) -> None:
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        result = await repo.get_by_id("chunk-1", user_id=TEST_OTHER_USER_ID_FIXED)

        assert result is None


class TestBatchInsertIgnoreEmpty:
    """batch_insert_ignore with empty list should short-circuit."""

    async def test_empty_list_returns_zero(self) -> None:
        mock_session = AsyncMock()
        repo = DBMemoryChunkRepository(db_session=mock_session)

        result = await repo.batch_insert_ignore([])

        assert result == 0
        mock_session.execute.assert_not_called()
