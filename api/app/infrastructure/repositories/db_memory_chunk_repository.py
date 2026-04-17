from __future__ import annotations

import copy
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import Select, delete, func, select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.memory_chunk import MemoryChunk
from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository
from app.infrastructure.models.memory_chunk_orm import MemoryChunkModel

def _escape_ilike(query: str) -> str:
    """Escape SQL LIKE wildcards (``%``/``_``) in user input.

    必须与 ``.ilike(pattern, escape="\\")`` 配对使用；否则 ``%``/``_`` 仍会被
    Postgres 解析为通配符，导致过滤条件失效（例如 ``file_name`` 会匹配
    ``filename``）。
    """
    return query.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")


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

    async def list_by_user(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        category: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[MemoryChunk]:
        stmt = select(MemoryChunkModel).where(MemoryChunkModel.user_id == user_id)
        stmt = self._apply_filters(
            stmt,
            query=query,
            source=source,
            category=category,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
        )
        # id DESC 作为 updated_at 并列时的确定性 tie-breaker：
        # MemoryFlushService 会给同一 batch 复用同一 now()，并列行非常常见；
        # 没有 tie-breaker 时 offset 分页在并列行上不保证稳定顺序，跨页可能重复/漏项。
        stmt = (
            stmt.order_by(
                MemoryChunkModel.updated_at.desc(),
                MemoryChunkModel.id.desc(),
            )
            .offset(offset)
            .limit(limit)
        )
        result = await self.db_session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    async def count_by_user(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        category: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(MemoryChunkModel).where(
            MemoryChunkModel.user_id == user_id
        )
        stmt = self._apply_filters(
            stmt,
            query=query,
            source=source,
            category=category,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
        )
        result = await self.db_session.execute(stmt)
        return int(result.scalar_one())

    async def find_pending_fs_sync(
        self,
        *,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[MemoryChunk]:
        """返回 fs_synced=false 的行，按 updated_at ASC 排序（先补旧的）。"""
        stmt = select(MemoryChunkModel).where(MemoryChunkModel.fs_synced.is_(False))
        if user_id is not None:
            stmt = stmt.where(MemoryChunkModel.user_id == user_id)
        stmt = (
            stmt.order_by(
                MemoryChunkModel.updated_at.asc(),
                MemoryChunkModel.id.asc(),
            )
            .limit(limit)
        )
        result = await self.db_session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    async def mark_fs_synced(
        self, *, chunk_id: str, user_id: str, synced: bool = True
    ) -> bool:
        """标记 chunk 的 fs_synced 字段。命中返回 True。"""
        stmt = (
            update(MemoryChunkModel)
            .where(
                MemoryChunkModel.id == chunk_id,
                MemoryChunkModel.user_id == user_id,
            )
            .values(fs_synced=synced)
        )
        result = await self.db_session.execute(stmt)
        return bool(result.rowcount)

    async def update_content(
        self,
        *,
        chunk_id: str,
        user_id: str,
        content: str,
        content_hash: str,
        embedding: tuple[float, ...] | None,
    ) -> MemoryChunk | None:
        # updated_at 走数据库 now() —— `update().values()` 绕过 ORM 脏标记，
        # onupdate=datetime.now 不会触发；使用服务端时间消除多 pod 时钟漂移。
        # fs_synced=False 原子归并到同一条 UPDATE：内容一变，文件就 out-of-sync，
        # 直到 FsMemoryWriter 回写完成。放一条语句里可免一次 round trip + 避免
        # "content 已改但 flag 还没翻" 的窗口期（如果两步分开，窗口内 reconciler
        # 看到 fs_synced=true 会跳过已失效文件）。
        stmt = (
            update(MemoryChunkModel)
            .where(
                MemoryChunkModel.id == chunk_id,
                MemoryChunkModel.user_id == user_id,
            )
            .values(
                content=content,
                content_hash=content_hash,
                embedding=list(embedding) if embedding is not None else None,
                updated_at=text("now()"),
                fs_synced=False,
            )
            .returning(MemoryChunkModel)
        )
        result = await self.db_session.execute(stmt)
        row = result.scalar_one_or_none()
        return self._to_domain(row) if row else None

    async def delete_by_ids(
        self, *, user_id: str, ids: list[str]
    ) -> list[MemoryChunk]:
        """按 id 批量删除本用户的记忆，返回实际被删除的行（DELETE ... RETURNING）。

        审计路径依赖"实际删除集"：把删除与快照合并到同一条 RETURNING 语句
        避免 READ COMMITTED 下 SELECT → DELETE 两步之间的并发 TOCTOU。
        """
        if not ids:
            return []
        stmt = (
            delete(MemoryChunkModel)
            .where(
                MemoryChunkModel.user_id == user_id,
                MemoryChunkModel.id.in_(ids),
            )
            .returning(MemoryChunkModel)
        )
        result = await self.db_session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    async def delete_all_by_user(self, *, user_id: str) -> list[MemoryChunk]:
        """删除本用户所有记忆，返回实际被删除的行（``DELETE ... RETURNING *``）。

        PR-5A 起返回完整 row：调用方同时需要 source 分布（审计）和 (id, category)
        对（FsMemoryWriter.delete）。单条 RETURNING 把"实际删除集"、"source
        分布"、"待清盘文件列表"锁在同一语句下，避免 READ COMMITTED 并发竞态。

        N.B. 相对于 PR-0 版本（``RETURNING source`` 标量），本版本把每行所有列
        拉回来。nuclear delete 是低频操作（用户 "danger zone" 按钮），多出的
        列开销可接受；memory_chunks 行无超大字段（content 典型 <1KB），10 万
        行 ≈ 100MB 也只在一次请求内存里活几秒。
        """
        stmt = (
            delete(MemoryChunkModel)
            .where(MemoryChunkModel.user_id == user_id)
            .returning(MemoryChunkModel)
        )
        result = await self.db_session.execute(stmt)
        return [self._to_domain(row) for row in result.scalars().all()]

    # ---- Filter helper ----

    @staticmethod
    def _apply_filters(
        stmt: Select[Any],
        *,
        query: str | None,
        source: str | None,
        category: str | None = None,
        created_from: datetime | None,
        created_to: datetime | None,
        updated_from: datetime | None,
        updated_to: datetime | None,
    ) -> Select[Any]:
        """共享的过滤条件装配，list_by_user / count_by_user 复用。"""
        if query is not None and query != "":
            escaped = _escape_ilike(query)
            stmt = stmt.where(
                MemoryChunkModel.content.ilike(f"%{escaped}%", escape="\\")
            )
        if source is not None:
            stmt = stmt.where(MemoryChunkModel.source == source)
        if category is not None:
            # 显式传入类别 → 只返回该类；category IS NULL 的 legacy 行不命中
            stmt = stmt.where(MemoryChunkModel.category == category)
        if created_from is not None:
            stmt = stmt.where(MemoryChunkModel.created_at >= created_from)
        if created_to is not None:
            stmt = stmt.where(MemoryChunkModel.created_at <= created_to)
        if updated_from is not None:
            stmt = stmt.where(MemoryChunkModel.updated_at >= updated_from)
        if updated_to is not None:
            stmt = stmt.where(MemoryChunkModel.updated_at <= updated_to)
        return stmt

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
            category=row.category,
            auto_promoted_at=row.auto_promoted_at,
            fs_synced=row.fs_synced,
            pinned=row.pinned,
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
            "category": chunk.category,
            "auto_promoted_at": chunk.auto_promoted_at,
            "fs_synced": chunk.fs_synced,
            "pinned": chunk.pinned,
        }
