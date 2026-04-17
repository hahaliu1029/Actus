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
    category: str | None = None,
    auto_promoted_at: datetime | None = None,
    fs_synced: bool = False,
    pinned: bool = False,
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
    # M1 PR-1 fields
    row.category = category
    row.auto_promoted_at = auto_promoted_at
    row.fs_synced = fs_synced
    row.pinned = pinned
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


# ─── M1 PR-1: category/fs_synced/auto_promoted_at/pinned ─────────────────────


class TestToDomainM1Fields:
    """_to_domain 把 ORM 新字段原样带到 domain MemoryChunk。"""

    def test_m1_fields_passthrough(self) -> None:
        promoted = datetime(2026, 4, 17, tzinfo=timezone.utc)
        row = _make_orm_row(
            category="user",
            auto_promoted_at=promoted,
            fs_synced=True,
            pinned=True,
        )
        result = DBMemoryChunkRepository._to_domain(row)
        assert result.category == "user"
        assert result.auto_promoted_at == promoted
        assert result.fs_synced is True
        assert result.pinned is True

    def test_m1_fields_default_none_false_for_legacy_rows(self) -> None:
        """旧 migration 之前的行 category IS NULL，fs_synced 由 migration 置 true。"""
        row = _make_orm_row()  # 默认 category=None, fs_synced=False, pinned=False
        result = DBMemoryChunkRepository._to_domain(row)
        assert result.category is None
        assert result.auto_promoted_at is None
        assert result.fs_synced is False
        assert result.pinned is False


class TestToOrmDictM1Fields:
    """_to_orm_dict 把 domain 新字段映射到 ORM insert payload。"""

    def test_m1_fields_included(self) -> None:
        promoted = datetime(2026, 4, 17, tzinfo=timezone.utc)
        chunk = _make_domain_chunk(
            category="rule",
            auto_promoted_at=promoted,
            fs_synced=True,
            pinned=False,
        )
        payload = DBMemoryChunkRepository._to_orm_dict(chunk)
        assert payload["category"] == "rule"
        assert payload["auto_promoted_at"] == promoted
        assert payload["fs_synced"] is True
        assert payload["pinned"] is False


class TestApplyFiltersCategory:
    """_apply_filters 新增 category 过滤项。"""

    def test_category_filter_produces_where_clause(self) -> None:
        """category='user' → WHERE 中带 category 等值 predicate；None → 不加过滤。"""
        from sqlalchemy import select

        base = select(MemoryChunkModel)
        with_filter = DBMemoryChunkRepository._apply_filters(
            base,
            query=None,
            source=None,
            category="user",
            created_from=None,
            created_to=None,
            updated_from=None,
            updated_to=None,
        )
        compiled = str(with_filter.compile(compile_kwargs={"literal_binds": True}))
        # SELECT clause 总会列出所有列——只检查 WHERE 段
        assert "WHERE" in compiled
        where_segment = compiled.split("WHERE", 1)[1]
        assert "category = 'user'" in where_segment

        without_filter = DBMemoryChunkRepository._apply_filters(
            base,
            query=None,
            source=None,
            category=None,
            created_from=None,
            created_to=None,
            updated_from=None,
            updated_to=None,
        )
        compiled_no = str(without_filter.compile(compile_kwargs={"literal_binds": True}))
        # 无任何 filter → 不应生成 WHERE 段，或 WHERE 段不含 category
        if "WHERE" in compiled_no:
            where_no = compiled_no.split("WHERE", 1)[1]
            assert "category" not in where_no


class TestFindPendingFsSync:
    """find_pending_fs_sync：FsReconciler 用的待同步行查询。"""

    async def test_returns_domain_chunks(self) -> None:
        row = _make_orm_row(fs_synced=False)
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = [row]
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        result = await repo.find_pending_fs_sync(limit=10)

        assert len(result) == 1
        assert result[0].fs_synced is False
        # SQL 应带上 fs_synced = false 过滤
        executed_stmt = mock_session.execute.call_args.args[0]
        compiled = str(executed_stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "fs_synced" in compiled and "false" in compiled.lower()

    async def test_user_id_filter_applied(self) -> None:
        mock_scalars = MagicMock()
        mock_scalars.all.return_value = []
        mock_result = MagicMock()
        mock_result.scalars.return_value = mock_scalars
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        await repo.find_pending_fs_sync(user_id=TEST_USER_ID_FIXED, limit=5)

        executed_stmt = mock_session.execute.call_args.args[0]
        compiled = str(executed_stmt.compile(compile_kwargs={"literal_binds": True}))
        assert TEST_USER_ID_FIXED in compiled


class TestMarkFsSynced:
    """mark_fs_synced：写盘成功后把 false → true。"""

    async def test_returns_true_when_row_hit(self) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        hit = await repo.mark_fs_synced(
            chunk_id="chunk-1", user_id=TEST_USER_ID_FIXED
        )
        assert hit is True

    async def test_returns_false_when_no_row_hit(self) -> None:
        mock_result = MagicMock()
        mock_result.rowcount = 0
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        hit = await repo.mark_fs_synced(
            chunk_id="missing", user_id=TEST_USER_ID_FIXED
        )
        assert hit is False

    async def test_can_set_false_for_rollback(self) -> None:
        """synced=False 用于 update/move_category 路径开启同步窗口。"""
        mock_result = MagicMock()
        mock_result.rowcount = 1
        mock_session = AsyncMock()
        mock_session.execute.return_value = mock_result

        repo = DBMemoryChunkRepository(db_session=mock_session)
        hit = await repo.mark_fs_synced(
            chunk_id="chunk-1", user_id=TEST_USER_ID_FIXED, synced=False
        )
        assert hit is True
