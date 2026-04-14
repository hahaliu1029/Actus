from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository
from app.infrastructure.models.memory_chunk_orm import MemoryChunkModel

if TYPE_CHECKING:
    from typing import Any


class DBMemoryChunkRepository(MemoryChunkRepository):
    """MemoryChunkRepository Protocol 的 PostgreSQL + pgvector 实现。

    显式继承 Protocol 与 DBSessionRepository / DBFileRepository 等一致，
    便于依赖注入时类型检查和代码搜索。
    """

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    # ---- Protocol methods ----

    async def batch_insert_ignore(self, chunks: Sequence[MemoryChunk]) -> int:
        if not chunks:
            return 0
        stmt = (
            pg_insert(MemoryChunkModel)
            .values([self._to_orm_dict(c) for c in chunks])
            .on_conflict_do_nothing(constraint="uq_memory_user_hash")
        )
        result = await self.db_session.execute(stmt)
        return result.rowcount

    async def search_by_vector(
        self,
        user_id: str,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.35,
    ) -> list[MemoryChunk]:
        max_distance = 1.0 - threshold
        distance_expr = MemoryChunkModel.embedding.cosine_distance(embedding)
        stmt = (
            select(MemoryChunkModel)
            .where(
                MemoryChunkModel.user_id == user_id,
                MemoryChunkModel.embedding.is_not(None),
                distance_expr <= max_distance,
            )
            .order_by(distance_expr.asc())
            .limit(top_k)
        )
        result = await self.db_session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    async def delete_by_session(self, session_id: str) -> int:
        stmt = delete(MemoryChunkModel).where(
            MemoryChunkModel.session_id == session_id
        )
        result = await self.db_session.execute(stmt)
        return result.rowcount

    async def get_by_id(self, chunk_id: str, user_id: str) -> MemoryChunk | None:
        stmt = select(MemoryChunkModel).where(
            MemoryChunkModel.id == chunk_id,
            MemoryChunkModel.user_id == user_id,
        )
        result = await self.db_session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    # ---- Conversion helpers ----

    @staticmethod
    def _to_domain(row: MemoryChunkModel) -> MemoryChunk:
        return MemoryChunk(
            id=row.id,
            user_id=row.user_id,
            session_id=row.session_id,
            content=row.content,
            content_hash=row.content_hash,
            embedding=tuple(row.embedding) if row.embedding is not None else None,
            source=row.source,
            metadata=copy.deepcopy(row.metadata_),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def _to_orm_dict(chunk: MemoryChunk) -> dict[str, Any]:
        return {
            "id": chunk.id,
            "user_id": chunk.user_id,
            "session_id": chunk.session_id,
            "content": chunk.content,
            "content_hash": chunk.content_hash,
            "embedding": list(chunk.embedding) if chunk.embedding is not None else None,
            "source": chunk.source,
            "metadata_": chunk.metadata,
            "created_at": chunk.created_at,
            "updated_at": chunk.updated_at,
        }
