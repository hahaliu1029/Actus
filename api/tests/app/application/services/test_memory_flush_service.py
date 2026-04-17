"""Tests for MemoryFlushService: scheduling skeleton (C5.0) + embed/write pipeline (C5.1).

Verifies:
1. MemoryFlushService satisfies MemoryFlusher protocol
2. submit creates a background task
3. Background task completes and self-discards from _pending_tasks
4. shutdown empty (no error)
5. shutdown waits for pending tasks
6. Circuit breaker initial state
7. Circuit breaker blocks submit when threshold exceeded
8. _embed_batch success / degradation / count mismatch
9. _do_flush success / cold data / retry failure
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.domain.models.memory_chunk import FlushBatch, RawChunk

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


def _make_batch(**overrides) -> FlushBatch:
    """Create a minimal FlushBatch for testing."""
    defaults = {
        "session_id": "test-session",
        "user_id": TEST_USER_ID_FIXED,
        "from_cursor": 0,
        "target_cursor": 5,
        "chunks": (
            RawChunk(
                content="test content",
                session_id="test-session",
                user_id=TEST_USER_ID_FIXED,
                source="test",
                metadata={},
                content_hash="abc123",
            ),
        ),
    }
    defaults.update(overrides)
    return FlushBatch(**defaults)


def _make_service(**overrides):
    """Construct MemoryFlushService with mock dependencies."""
    from app.application.services.memory_flush_service import MemoryFlushService

    defaults = dict(
        embedding_provider=AsyncMock(),
        session_factory=MagicMock(),
        repo_factory=MagicMock(),
    )
    defaults.update(overrides)
    return MemoryFlushService(**defaults)


# ── Existing C5.0 tests (adapted for new __init__ signature) ───────────────


class TestMemoryFlushServiceProtocol:
    """MemoryFlushService satisfies MemoryFlusher protocol."""

    def test_satisfies_memory_flusher_protocol(self) -> None:
        from app.domain.external.memory_flusher import MemoryFlusher

        service = _make_service()
        assert hasattr(service, "submit")
        assert callable(service.submit)
        assert isinstance(service, MemoryFlusher)


class TestMemoryFlushServiceSubmit:
    """submit creates background task."""

    async def test_submit_creates_task(self) -> None:
        service = _make_service()
        with patch.object(service, "_do_flush", new_callable=AsyncMock):
            batch = _make_batch()
            service.submit(batch)
            assert len(service._pending_tasks) == 1

    async def test_task_completes_and_self_discards(self) -> None:
        service = _make_service()
        with patch.object(service, "_do_flush", new_callable=AsyncMock):
            batch = _make_batch()
            service.submit(batch)
            assert len(service._pending_tasks) == 1
            await asyncio.sleep(0.1)
            assert len(service._pending_tasks) == 0


class TestMemoryFlushServiceShutdown:
    """shutdown behavior."""

    async def test_shutdown_empty_no_error(self) -> None:
        service = _make_service()
        await service.shutdown()  # Should not raise

    async def test_shutdown_waits_for_pending(self) -> None:
        service = _make_service()
        with patch.object(service, "_do_flush", new_callable=AsyncMock):
            batch = _make_batch()
            service.submit(batch)
            assert len(service._pending_tasks) == 1
            await service.shutdown()
            assert len(service._pending_tasks) == 0


class TestMemoryFlushServiceCircuitBreaker:
    """Circuit breaker behavior."""

    def test_initial_state(self) -> None:
        service = _make_service()
        assert service._consecutive_failures == 0
        assert service._last_failure_time is None

    async def test_circuit_breaker_blocks_submit(self) -> None:
        service = _make_service(circuit_breaker_threshold=3)
        service._consecutive_failures = 3
        service._last_failure_time = time.monotonic()

        batch = _make_batch()
        service.submit(batch)
        assert len(service._pending_tasks) == 0


# ── C5.1 new tests: _embed_batch ───────────────────────────────────────────


class TestEmbedBatch:
    """_embed_batch: embed texts, degrade on EmbeddingUnavailableError."""

    async def test_success_returns_tuples(self) -> None:
        provider = AsyncMock()
        provider.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        service = _make_service(embedding_provider=provider)

        chunks = (
            _make_batch().chunks[0],
            RawChunk(content="second", session_id="s", user_id="u",
                     source="test", metadata={}, content_hash="h2"),
        )
        result = await service._embed_batch(chunks)

        assert len(result) == 2
        assert result[0] == (0.1, 0.2)
        assert isinstance(result[0], tuple)

    async def test_unavailable_returns_none_list(self) -> None:
        from app.domain.external.embedding_provider import EmbeddingUnavailableError

        provider = AsyncMock()
        provider.embed.side_effect = EmbeddingUnavailableError("disabled")
        service = _make_service(embedding_provider=provider)

        chunks = (_make_batch().chunks[0],)
        result = await service._embed_batch(chunks)

        assert result == [None]

    async def test_empty_chunks_returns_empty(self) -> None:
        provider = AsyncMock()
        provider.embed.return_value = []
        service = _make_service(embedding_provider=provider)

        result = await service._embed_batch(())
        assert result == []

    async def test_count_mismatch_raises_value_error(self) -> None:
        provider = AsyncMock()
        provider.embed.return_value = [[0.1]]  # 1 vector for 2 chunks
        service = _make_service(embedding_provider=provider)

        chunks = (
            _make_batch().chunks[0],
            RawChunk(content="second", session_id="s", user_id="u",
                     source="test", metadata={}, content_hash="h2"),
        )
        with pytest.raises(ValueError, match="mismatch"):
            await service._embed_batch(chunks)


# ── C5.1 new tests: _do_flush ──────────────────────────────────────────────


class TestDoFlush:
    """_do_flush: full pipeline embed → build → insert."""

    async def test_success_path(self) -> None:
        """Embed + insert + commit + reset breaker."""
        provider = AsyncMock()
        provider.embed.return_value = [[0.1, 0.2]]

        mock_repo = AsyncMock()
        mock_repo.batch_insert_ignore.return_value = 1

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        session_factory = MagicMock(return_value=mock_session)
        repo_factory = MagicMock(return_value=mock_repo)

        service = _make_service(
            embedding_provider=provider,
            session_factory=session_factory,
            repo_factory=repo_factory,
        )
        batch = _make_batch()

        await service._do_flush(batch)

        provider.embed.assert_awaited_once()
        repo_factory.assert_called_once_with(mock_session)
        mock_repo.batch_insert_ignore.assert_awaited_once()
        mock_session.commit.assert_awaited_once()
        assert service._consecutive_failures == 0

    async def test_cold_data_on_embedding_unavailable(self) -> None:
        """When embedding is unavailable, still write with None embeddings."""
        from app.domain.external.embedding_provider import EmbeddingUnavailableError

        provider = AsyncMock()
        provider.embed.side_effect = EmbeddingUnavailableError("disabled")

        mock_repo = AsyncMock()
        mock_repo.batch_insert_ignore.return_value = 1

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        session_factory = MagicMock(return_value=mock_session)
        repo_factory = MagicMock(return_value=mock_repo)

        service = _make_service(
            embedding_provider=provider,
            session_factory=session_factory,
            repo_factory=repo_factory,
        )
        batch = _make_batch()

        await service._do_flush(batch)

        mock_repo.batch_insert_ignore.assert_awaited_once()
        chunks = mock_repo.batch_insert_ignore.call_args[0][0]
        assert all(c.embedding is None for c in chunks)

    async def test_db_failure_increments_breaker(self) -> None:
        """All retries fail → circuit breaker incremented."""
        provider = AsyncMock()
        provider.embed.return_value = [[0.1]]

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit.side_effect = RuntimeError("DB down")
        mock_session.rollback = AsyncMock()

        session_factory = MagicMock(return_value=mock_session)
        repo_factory = MagicMock(return_value=AsyncMock())

        service = _make_service(
            embedding_provider=provider,
            session_factory=session_factory,
            repo_factory=repo_factory,
            max_retries=0,
        )
        batch = _make_batch()

        await service._do_flush(batch)

        assert service._consecutive_failures == 1
        mock_session.rollback.assert_awaited()  # 事务失败时必须显式 rollback
