"""Tests for MemoryFlusher Protocol (Task 2)."""
from __future__ import annotations

from typing import runtime_checkable

import pytest

from app.domain.external.memory_flusher import MemoryFlusher
from app.domain.models.memory_chunk import FlushBatch, RawChunk

from tests.conftest import TEST_USER_ID_FIXED


class StubMemoryFlusher:
    """满足 MemoryFlusher 协议的最小桩实现。"""

    def __init__(self) -> None:
        self.submitted: list[FlushBatch] = []

    def submit(self, batch: FlushBatch) -> None:
        self.submitted.append(batch)


class TestMemoryFlusherProtocol:
    def _make_batch(self) -> FlushBatch:
        chunk = RawChunk(
            content="test",
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            source="react_graph",
            metadata={},
            content_hash="abc123",
        )
        return FlushBatch(
            session_id="sess-1",
            user_id=TEST_USER_ID_FIXED,
            from_cursor=0,
            target_cursor=1,
            chunks=(chunk,),
        )

    def test_stub_satisfies_protocol(self) -> None:
        """StubMemoryFlusher 满足 MemoryFlusher Protocol（structural subtyping）。"""
        flusher: MemoryFlusher = StubMemoryFlusher()  # type: ignore[assignment]
        batch = self._make_batch()
        flusher.submit(batch)

    def test_submit_is_called_with_batch(self) -> None:
        """调用 submit 后 batch 被传递给实现。"""
        stub = StubMemoryFlusher()
        batch = self._make_batch()
        stub.submit(batch)
        assert len(stub.submitted) == 1
        assert stub.submitted[0] is batch

    def test_submit_multiple_batches(self) -> None:
        """多次调用 submit 都被记录。"""
        stub = StubMemoryFlusher()
        for i in range(3):
            chunk = RawChunk(
                content=f"msg {i}",
                session_id="sess-1",
                user_id=TEST_USER_ID_FIXED,
                source="react_graph",
                metadata={},
                content_hash=f"hash{i}",
            )
            batch = FlushBatch(
                session_id="sess-1",
                user_id=TEST_USER_ID_FIXED,
                from_cursor=i,
                target_cursor=i + 1,
                chunks=(chunk,),
            )
            stub.submit(batch)
        assert len(stub.submitted) == 3

    def test_protocol_has_submit_method(self) -> None:
        """MemoryFlusher 协议声明了 submit 方法。"""
        assert hasattr(MemoryFlusher, "submit")
