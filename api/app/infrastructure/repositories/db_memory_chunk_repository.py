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
        pinned: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        auto_promoted_after: datetime | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> list[MemoryChunk]:
        stmt = select(MemoryChunkModel).where(MemoryChunkModel.user_id == user_id)
        stmt = self._apply_filters(
            stmt,
            query=query,
            source=source,
            category=category,
            pinned=pinned,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
            auto_promoted_after=auto_promoted_after,
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
        pinned: bool | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        auto_promoted_after: datetime | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(MemoryChunkModel).where(
            MemoryChunkModel.user_id == user_id
        )
        stmt = self._apply_filters(
            stmt,
            query=query,
            source=source,
            category=category,
            pinned=pinned,
            created_from=created_from,
            created_to=created_to,
            updated_from=updated_from,
            updated_to=updated_to,
            auto_promoted_after=auto_promoted_after,
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

    async def distinct_user_ids(self) -> list[str]:
        stmt = select(MemoryChunkModel.user_id).distinct()
        result = await self.db_session.execute(stmt)
        return list(result.scalars().all())

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

    async def delete_legacy_by_user(
        self,
        *,
        user_id: str,
        rollout_at: datetime | None = None,
    ) -> list[MemoryChunk]:
        """删除 legacy session_flush 行（``DELETE ... RETURNING *``）。

        核心谓词 ``source='session_flush' AND category IS NULL AND
        auto_promoted_at IS NULL`` —— 未分类、也未经 LLM gate 收录的旧
        flush 块。categorized / auto-promoted / manual / memory_save 行
        永远不命中，防止误删已被用户或系统背书的数据。

        ``rollout_at`` 非空 → 再加 ``AND created_at < rollout_at``（codex fix
        P1）：gate 关闭 deployment 里 post-launch 新写入 session_flush 也是
        (NULL, NULL)，时间边界防误删。为空时沿用旧谓词，由上层 UI 做警告。

        单条 RETURNING * 让调用方同时拿到"被删条数 + chunk_ids（审计）"+
        "(id, category) 对（fs 清盘）"——legacy 行 category IS NULL 从未
        落盘，service 层跳过 ``file_store.delete``。
        """
        conditions = [
            MemoryChunkModel.user_id == user_id,
            MemoryChunkModel.source == "session_flush",
            MemoryChunkModel.category.is_(None),
            MemoryChunkModel.auto_promoted_at.is_(None),
        ]
        if rollout_at is not None:
            conditions.append(MemoryChunkModel.created_at < rollout_at)

        stmt = (
            delete(MemoryChunkModel)
            .where(*conditions)
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
        pinned: bool | None = None,
        created_from: datetime | None,
        created_to: datetime | None,
        updated_from: datetime | None,
        updated_to: datetime | None,
        auto_promoted_after: datetime | None = None,
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
        if pinned is not None:
            # DB CHECK 保证 pinned=true 仅在 category='user' 时合法；partial index
            # ``ix_memory_chunks_user_pinned`` (pinned=true WHERE) 支撑 M2 snapshot
            # 的 "pinned 优先" 两阶段拉取。
            stmt = stmt.where(MemoryChunkModel.pinned.is_(pinned))
        if created_from is not None:
            stmt = stmt.where(MemoryChunkModel.created_at >= created_from)
        if created_to is not None:
            stmt = stmt.where(MemoryChunkModel.created_at <= created_to)
        if updated_from is not None:
            stmt = stmt.where(MemoryChunkModel.updated_at >= updated_from)
        if updated_to is not None:
            stmt = stmt.where(MemoryChunkModel.updated_at <= updated_to)
        if auto_promoted_after is not None:
            # 仅 auto-flush 路径写 auto_promoted_at；manual / memory_save 路径
            # 留 NULL。NULL 不进入比较结果集合，下游 audit 视图自动只看 LLM gate
            # 实际收录的行。索引：用户级走 ix_memory_chunks_user_updated_at 命中
            # user，再在用户范围内 in-memory 比较 timestamp（典型 100-1000 行
            # 量级，不需要 dedicated index）。
            stmt = stmt.where(MemoryChunkModel.auto_promoted_at >= auto_promoted_after)
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
