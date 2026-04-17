"""Tests for MemoryChunkRepository Protocol — method call contract."""
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

import pytest

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


class StubMemoryChunkRepository:
    """满足 MemoryChunkRepository 协议的最小桩实现。"""

    def __init__(self) -> None:
        self.inserted: list[MemoryChunk] = []
        self.deleted_sessions: list[str] = []

    async def batch_insert_ignore(self, chunks: Sequence[MemoryChunk]) -> int:
        self.inserted.extend(chunks)
        return len(chunks)

    async def search_by_vector(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> list[MemoryChunk]:
        return []

    async def delete_by_session(self, session_id: str) -> int:
        self.deleted_sessions.append(session_id)
        return 0

    async def get_by_id(self, chunk_id: str, user_id: str) -> MemoryChunk | None:
        return None


def _make_chunk(**overrides) -> MemoryChunk:
    defaults = dict(
        id="chunk-1",
        user_id=TEST_USER_ID_FIXED,
        content="test content",
        content_hash="hash123",
        source="session_flush",
        metadata={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(overrides)
    return MemoryChunk(**defaults)


class TestMemoryChunkRepositoryProtocol:
    """Verify stub's 3 async methods are callable with correct arg/return types.

    这组测试验证方法调用契约（参数可传入、返回值类型正确），不验证
    Stub 与 Protocol 的签名一致性——签名漂移检测留给 mypy/pyright
    （项目尚未配置静态类型检查器，不在 C2 范围）。
    """

    async def test_batch_insert_ignore(self) -> None:
        """batch_insert_ignore 接受 Sequence[MemoryChunk] 并返回 int。"""
        repo = StubMemoryChunkRepository()
        chunk = _make_chunk()
        result = await repo.batch_insert_ignore([chunk])
        assert isinstance(result, int)
        assert result == 1

    async def test_batch_insert_ignore_with_none_embedding(self) -> None:
        """batch_insert_ignore 接受 embedding=None 的 chunk。"""
        repo = StubMemoryChunkRepository()
        chunk = _make_chunk(embedding=None)
        result = await repo.batch_insert_ignore([chunk])
        assert result == 1

    async def test_batch_insert_ignore_with_embedding(self) -> None:
        """batch_insert_ignore 接受带 embedding 的 chunk。"""
        repo = StubMemoryChunkRepository()
        chunk = _make_chunk(embedding=(0.1, 0.2, 0.3))
        result = await repo.batch_insert_ignore([chunk])
        assert result == 1

    async def test_search_by_vector(self) -> None:
        """search_by_vector 接受参数并返回 list[MemoryChunk]。"""
        repo = StubMemoryChunkRepository()
        result = await repo.search_by_vector(
            user_id=TEST_USER_ID_FIXED,
            embedding=[0.1, 0.2, 0.3],
            top_k=5,
            threshold=0.35,
        )
        assert isinstance(result, list)

    async def test_search_by_vector_defaults(self) -> None:
        """search_by_vector 的 top_k 和 threshold 有默认值。"""
        repo = StubMemoryChunkRepository()
        result = await repo.search_by_vector(
            user_id=TEST_USER_ID_FIXED,
            embedding=[0.1, 0.2, 0.3],
        )
        assert isinstance(result, list)

    async def test_delete_by_session(self) -> None:
        """delete_by_session 接受 session_id 并返回 int。"""
        repo = StubMemoryChunkRepository()
        result = await repo.delete_by_session("sess-1")
        assert isinstance(result, int)
        assert repo.deleted_sessions == ["sess-1"]

    async def test_get_by_id_returns_none(self) -> None:
        """get_by_id stub returns None."""
        repo = StubMemoryChunkRepository()
        result = await repo.get_by_id("chunk-1", user_id=TEST_USER_ID_FIXED)
        assert result is None

    def test_protocol_has_four_methods(self) -> None:
        """MemoryChunkRepository 协议声明了 4 个方法。"""
        methods = {"batch_insert_ignore", "search_by_vector", "delete_by_session", "get_by_id"}
        for method in methods:
            assert hasattr(MemoryChunkRepository, method)
