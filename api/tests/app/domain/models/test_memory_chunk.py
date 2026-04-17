"""Tests for RawChunk, FlushBatch, and MemoryChunk dataclasses."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.domain.models.memory_chunk import FlushBatch, MemoryChunk, RawChunk

from tests.conftest import TEST_USER_ID_FIXED


class TestRawChunk:
    def test_fields_accessible(self) -> None:
        """RawChunk 字段可以正常访问。"""
        chunk = RawChunk(
            content="hello world",
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={"key": "value"},
            content_hash="abc123",
        )
        assert chunk.content == "hello world"
        assert chunk.session_id == "sess-1"
        assert chunk.user_id == TEST_USER_ID_FIXED
        assert chunk.source == "react_graph"
        assert chunk.metadata == {"key": "value"}
        assert chunk.content_hash == "abc123"

    def test_frozen_immutability(self) -> None:
        """RawChunk 是 frozen dataclass，修改字段应抛出 FrozenInstanceError。"""
        chunk = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={},
            content_hash="abc123",
        )
        with pytest.raises(Exception):  # FrozenInstanceError (subclass of AttributeError)
            chunk.content = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """相同字段的 RawChunk 应相等。"""
        chunk_a = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={"k": "v"},
            content_hash="abc123",
        )
        chunk_b = RawChunk(
            content="hello",
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={"k": "v"},
            content_hash="abc123",
        )
        assert chunk_a == chunk_b


class TestFlushBatch:
    def _make_chunk(self, content: str = "test content") -> RawChunk:
        return RawChunk(
            content=content,
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={},
            content_hash="hash_" + content[:8],
        )

    def test_fields_accessible(self) -> None:
        """FlushBatch 字段可以正常访问。"""
        chunk = self._make_chunk()
        batch = FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=5,
            chunks=(chunk,),
        )
        assert batch.session_id == "sess-1"
        assert batch.user_id == TEST_USER_ID_FIXED
        assert batch.from_cursor == 0
        assert batch.target_cursor == 5
        assert len(batch.chunks) == 1
        assert batch.chunks[0] is chunk

    def test_frozen_immutability(self) -> None:
        """FlushBatch 是 frozen dataclass，修改字段应抛出 FrozenInstanceError。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=5,
            chunks=(),
        )
        with pytest.raises(Exception):
            batch.session_id = "other"  # type: ignore[misc]

    def test_from_cursor_and_target_cursor(self) -> None:
        """from_cursor 和 target_cursor 正确存储游标值。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=10,
            target_cursor=20,
            chunks=(),
        )
        assert batch.from_cursor == 10
        assert batch.target_cursor == 20

    def test_empty_chunks_tuple(self) -> None:
        """FlushBatch 支持空 chunks 元组。"""
        batch = FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=0,
            chunks=(),
        )
        assert batch.chunks == ()
        assert len(batch.chunks) == 0

    def test_multiple_chunks(self) -> None:
        """FlushBatch 支持多个 RawChunk。"""
        chunks = tuple(self._make_chunk(f"content {i}") for i in range(3))
        batch = FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=3,
            chunks=chunks,
        )
        assert len(batch.chunks) == 3


class TestMemoryChunk:
    """MemoryChunk frozen dataclass tests — mirrors TestRawChunk/TestFlushBatch above."""

    def _make_chunk(self, **overrides) -> MemoryChunk:
        defaults = dict(
            id="chunk-1",
            user_id=TEST_USER_ID_FIXED,
            content="hello world",
            content_hash="abc123def456",
            source="session_flush",
            metadata={"key": "value"},
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            updated_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )
        defaults.update(overrides)
        return MemoryChunk(**defaults)

    def test_fields_accessible(self) -> None:
        """MemoryChunk 字段可以正常访问。"""
        chunk = self._make_chunk()
        assert chunk.id == "chunk-1"
        assert chunk.user_id == TEST_USER_ID_FIXED
        assert chunk.content == "hello world"
        assert chunk.content_hash == "abc123def456"
        assert chunk.source == "session_flush"
        assert chunk.metadata == {"key": "value"}
        assert chunk.created_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert chunk.updated_at == datetime(2026, 1, 2, tzinfo=timezone.utc)

    def test_frozen_immutability(self) -> None:
        """MemoryChunk 是 frozen dataclass，修改字段应抛出 FrozenInstanceError。"""
        from dataclasses import FrozenInstanceError

        chunk = self._make_chunk()
        with pytest.raises(FrozenInstanceError):
            chunk.content = "modified"  # type: ignore[misc]

    def test_equality(self) -> None:
        """相同字段的 MemoryChunk 应相等。"""
        chunk_a = self._make_chunk()
        chunk_b = self._make_chunk()
        assert chunk_a == chunk_b

    def test_session_id_defaults_to_none(self) -> None:
        """session_id 默认为 None。"""
        chunk = self._make_chunk()
        assert chunk.session_id is None

    def test_session_id_can_be_set(self) -> None:
        """session_id 可以显式赋值。"""
        chunk = self._make_chunk(session_id="sess-1")
        assert chunk.session_id == "sess-1"

    def test_embedding_defaults_to_none(self) -> None:
        """embedding 默认为 None（provider 故障降级场景）。"""
        chunk = self._make_chunk()
        assert chunk.embedding is None

    def test_embedding_as_tuple(self) -> None:
        """embedding 可以是 float tuple。"""
        emb = (0.1, 0.2, 0.3)
        chunk = self._make_chunk(embedding=emb)
        assert chunk.embedding == (0.1, 0.2, 0.3)
        assert isinstance(chunk.embedding, tuple)

    def test_different_source_types(self) -> None:
        """source 支持不同的字符串值。"""
        for source in ("session_flush", "manual", "memory_save"):
            chunk = self._make_chunk(source=source)
            assert chunk.source == source

    # ── M1 PR-1 new fields ────────────────────────────────────────────────

    def test_m1_fields_default_to_none_or_false(self) -> None:
        """M1 PR-1 新增字段：category/auto_promoted_at/fs_synced/pinned 默认值。

        未显式传入时按"legacy / 未同步 / 非 pinned"解读：
        - category=None 表示历史行（M1 前写入的 session_flush）
        - fs_synced=False 让 FsReconciler 首次启动时把真实落盘状态拾回
        - pinned=False 永远安全——pinned=True 只对 user category 合法
        """
        chunk = self._make_chunk()
        assert chunk.category is None
        assert chunk.auto_promoted_at is None
        assert chunk.fs_synced is False
        assert chunk.pinned is False

    def test_m1_fields_can_be_set(self) -> None:
        """显式传入 M1 新字段。"""
        promoted = datetime(2026, 4, 17, tzinfo=timezone.utc)
        chunk = self._make_chunk(
            category="user",
            auto_promoted_at=promoted,
            fs_synced=True,
            pinned=True,
        )
        assert chunk.category == "user"
        assert chunk.auto_promoted_at == promoted
        assert chunk.fs_synced is True
        assert chunk.pinned is True
