from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.memory_chunk import MemoryChunk

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.models.memory_chunk import FlushBatch, RawChunk
    from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository

logger = logging.getLogger(__name__)
_CIRCUIT_BREAKER_RECOVERY_SECONDS = 300.0


class MemoryFlushService:
    """记忆刷写服务（app.state 单例）。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        session_factory: async_sessionmaker[AsyncSession],
        repo_factory: Callable[[AsyncSession], MemoryChunkRepository],
        max_retries: int = 3,
        circuit_breaker_threshold: int = 3,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._session_factory = session_factory
        self._repo_factory = repo_factory
        self._pending_tasks: set[asyncio.Task] = set()
        self._consecutive_failures: int = 0
        self._last_failure_time: float | None = None
        self._max_retries = max_retries
        self._circuit_breaker_threshold = circuit_breaker_threshold

    def submit(self, batch: FlushBatch) -> None:
        """Submit FlushBatch to background flush queue. Sync, non-blocking."""
        # Circuit breaker: check + time-based auto-recovery
        if self._consecutive_failures >= self._circuit_breaker_threshold:
            if (
                self._last_failure_time
                and (time.monotonic() - self._last_failure_time)
                > _CIRCUIT_BREAKER_RECOVERY_SECONDS
            ):
                self._consecutive_failures = 0
                self._last_failure_time = None
            else:
                logger.warning("MemoryFlushService: circuit breaker open, skipping")
                return

        task = asyncio.create_task(self._do_flush(batch))
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _do_flush(self, batch: FlushBatch) -> None:
        """Execute flush with retry."""
        last_exc = None
        for attempt in range(1 + self._max_retries):
            try:
                logger.info(
                    "MemoryFlushService._do_flush: session=%s from=%d to=%d chunks=%d attempt=%d",
                    batch.session_id,
                    batch.from_cursor,
                    batch.target_cursor,
                    len(batch.chunks),
                    attempt,
                )

                # 1. 批量嵌入（降级为冷数据）
                embeddings = await self._embed_batch(batch.chunks)

                # 2. 构建 MemoryChunk
                # M1 PR-4+8：plumb gate-assigned category + auto_promoted_at
                # （只有 LLM 质量闸通过的 chunk 才带值；size-only 路径两个都
                # 是 None，入库后 MemoryChunk.category 留 NULL = legacy 语义）。
                now = datetime.now(tz=timezone.utc)
                memory_chunks = [
                    MemoryChunk(
                        id=str(uuid.uuid4()),
                        user_id=c.user_id,
                        session_id=c.session_id,
                        content=c.content,
                        content_hash=c.content_hash,
                        source=c.source,
                        metadata=c.metadata,
                        created_at=now,
                        updated_at=now,
                        embedding=emb,
                        category=c.category,
                        auto_promoted_at=c.auto_promoted_at,
                    )
                    for c, emb in zip(batch.chunks, embeddings)
                ]

                # 3. 独立 session 写入
                async with self._session_factory() as session:
                    try:
                        repo = self._repo_factory(session)
                        inserted = await repo.batch_insert_ignore(memory_chunks)
                        await session.commit()
                    except Exception:
                        await session.rollback()
                        raise

                logger.info(
                    "Flush success: inserted=%d/%d",
                    inserted,
                    len(memory_chunks),
                )
                self._consecutive_failures = 0
                self._last_failure_time = None
                return
            except Exception as exc:
                last_exc = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(2**attempt)

        self._consecutive_failures += 1
        self._last_failure_time = time.monotonic()
        logger.warning("MemoryFlushService flush failed: %s", last_exc)

    async def _embed_batch(
        self, chunks: Sequence[RawChunk],
    ) -> list[tuple[float, ...] | None]:
        """批量嵌入。EmbeddingUnavailableError 时降级为全 None（冷数据）。"""
        texts = [c.content for c in chunks]
        try:
            vectors = await self._embedding_provider.embed(texts)
            if len(vectors) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: got {len(vectors)}, expected {len(texts)}"
                )
            return [tuple(v) for v in vectors]
        except EmbeddingUnavailableError:
            logger.warning(
                "Embedding unavailable, flushing as cold data (no vectors)"
            )
            return [None] * len(chunks)

    async def shutdown(self) -> None:
        if not self._pending_tasks:
            return
        _, pending = await asyncio.wait(self._pending_tasks, timeout=5)
        for task in pending:
            task.cancel()
        self._pending_tasks.clear()
