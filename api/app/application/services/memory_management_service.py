"""用户级长期记忆管理 Service。

职责：规范化输入 → 计算 hash → 重算 embedding → 写审计 → 提交事务。

与 MemoryFlushService 的分工：
- Flush：后台异步写路径，RawChunk → MemoryChunk 批量入库
- Management：前台用户 CRUD + 审计（edit / delete / bulk_delete / delete_all）

审计写入与业务写入共享同一 AsyncSession/事务，保证原子性：
业务写成功但审计写失败会整体回滚。
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Callable

from sqlalchemy.exc import IntegrityError

from app.application.errors.exceptions import ConflictError
from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.memory_chunk import MemoryChunk, memory_content_hash

_AUDIT_CONTENT_PREVIEW_LIMIT = 200

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.external.file_memory_store import FileMemoryStore
    from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository

logger = logging.getLogger(__name__)


class MemoryManagementService:
    """用户级长期记忆管理服务。"""

    def __init__(
        self,
        repo_factory: Callable[["AsyncSession"], "MemoryChunkRepository"],
        embedding_provider: "EmbeddingProvider",
        session_factory: "async_sessionmaker[AsyncSession]",
        *,
        file_store: "FileMemoryStore | None" = None,
    ) -> None:
        # ``file_store`` 在 PR-0 期间恒为 None（DB-only 模式），PR-5A 起由
        # lifespan 注入真实的 ``FsMemoryWriter``。None 时所有 CRUD 只落 DB，
        # 不碰文件系统，保持现有行为。
        self._repo_factory = repo_factory
        self._embedding_provider = embedding_provider
        self._session_factory = session_factory
        self._file_store = file_store

    async def list_memories(
        self,
        user_id: str,
        *,
        query: str | None = None,
        source: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[MemoryChunk], int]:
        """返回 (items, total)。page_size 钳位到 50。"""
        page_size = min(page_size, 50)
        page = max(page, 1)
        offset = (page - 1) * page_size

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            items = await repo.list_by_user(
                user_id,
                query=query,
                source=source,
                created_from=created_from,
                created_to=created_to,
                updated_from=updated_from,
                updated_to=updated_to,
                offset=offset,
                limit=page_size,
            )
            total = await repo.count_by_user(
                user_id,
                query=query,
                source=source,
                created_from=created_from,
                created_to=created_to,
                updated_from=updated_from,
                updated_to=updated_to,
            )
        return items, total

    async def get_memory(self, user_id: str, chunk_id: str) -> MemoryChunk | None:
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            return await repo.get_by_id(chunk_id, user_id)

    async def update_memory_content(
        self, user_id: str, chunk_id: str, new_content: str
    ) -> MemoryChunk | None:
        """编辑记忆内容。

        返回更新后的记忆；chunk 不存在或越权返回 None。
        hash 冲突（同用户下相同内容已存在）抛 ConflictError。
        """
        content = new_content.strip()
        if not content:
            raise ValueError("content must not be empty")

        new_hash = memory_content_hash(content)

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            # 先确认 chunk 存在且归属本用户，避免对越权/不存在的请求白白触发
            # embedding 计算（可能消耗 provider token）
            old = await repo.get_by_id(chunk_id, user_id)
            if old is None:
                return None

            # 重算 embedding：已知降级信号（断路器/provider 故障）静默降级为 None，
            # 与 flush 路径的冷数据语义一致；其他异常上抛，避免掩盖 bug。
            embedding: tuple[float, ...] | None = None
            try:
                vectors = await self._embedding_provider.embed([content])
                if vectors:
                    embedding = tuple(vectors[0])
            except EmbeddingUnavailableError:
                logger.warning(
                    "embedding 重算失败（provider 不可用），降级为冷数据 chunk_id=%s",
                    chunk_id,
                    exc_info=True,
                )

            try:
                updated = await repo.update_content(
                    chunk_id=chunk_id,
                    user_id=user_id,
                    content=content,
                    content_hash=new_hash,
                    embedding=embedding,
                )
            except IntegrityError as exc:
                # SQLAlchemy 将 asyncpg.UniqueViolationError 包装为 IntegrityError。
                # 只把 pgcode=23505（unique_violation）映射成 ConflictError；
                # 其他完整性错误（如 FK 违反）保留原始异常，避免误报。
                orig = getattr(exc, "orig", None)
                pgcode = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
                if pgcode == "23505":
                    raise ConflictError("相同内容的长期记忆已存在") from exc
                raise

            if updated is None:
                return None

            # 写审计（同一 session/事务）：快照 content 截断以避免日志聚合系统存留
            # 未脱敏的敏感全文（若需要全量内容可结合 content_hash 复查）。
            await self._write_audit(
                session,
                user_id=user_id,
                chunk_id=chunk_id,
                action="edit",
                old_snapshot={
                    "content": old.content[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                    "content_hash": old.content_hash,
                },
                new_snapshot={
                    "content": content[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                    "content_hash": new_hash,
                },
            )
            await session.commit()
            return updated

    async def delete_memory(self, user_id: str, chunk_id: str) -> bool:
        """单条删除。审计快照用 DELETE ... RETURNING 返回的真实被删行，
        避免先 SELECT 再 DELETE 两步之间的并发更新造成审计失真。"""
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            deleted_rows = await repo.delete_by_ids(
                user_id=user_id, ids=[chunk_id]
            )
            if not deleted_rows:
                return False

            actually_deleted = deleted_rows[0]
            await self._write_audit(
                session,
                user_id=user_id,
                chunk_id=chunk_id,
                action="delete",
                old_snapshot={
                    "content": actually_deleted.content[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                    "content_hash": actually_deleted.content_hash,
                    "source": actually_deleted.source,
                },
            )
            await session.commit()
            return True

    async def bulk_delete_memories(self, user_id: str, ids: list[str]) -> int:
        """批量删除。仅删除本用户拥有的 id；审计只记录被 DELETE ... RETURNING
        返回的实际删除集合，避免并发删除造成审计/真实不一致。"""
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            # DELETE ... RETURNING：删除与读取原子化，审计记录的是真正被删的行
            deleted_rows = await repo.delete_by_ids(user_id=user_id, ids=ids)
            deleted = len(deleted_rows)
            if deleted > 0:
                await self._write_audit(
                    session,
                    user_id=user_id,
                    chunk_id=None,
                    chunk_ids=[c.id for c in deleted_rows],
                    action="bulk_delete",
                    affected_count=deleted,
                    old_snapshot={
                        "deleted_summaries": [
                            {
                                "id": c.id,
                                "content": c.content[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                                "source": c.source,
                            }
                            for c in deleted_rows
                        ],
                    },
                )
                await session.commit()
            return deleted

    async def delete_all_memories(self, user_id: str) -> int:
        """一键清空本用户全部记忆。

        审计的 source_distribution 与 affected_count 均来自同一条
        ``DELETE ... RETURNING source`` 语句（repo 层聚合），保证三者一致。
        """
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            source_dist = await repo.delete_all_by_user(user_id=user_id)
            deleted = sum(source_dist.values())
            if deleted > 0:
                await self._write_audit(
                    session,
                    user_id=user_id,
                    chunk_id=None,
                    action="delete_all",
                    affected_count=deleted,
                    # source_distribution 即为实际删除分布；total 与 affected_count 恒等，
                    # 不再单独写入 total_before_delete 冗余字段。
                    old_snapshot={"source_distribution": source_dist},
                )
                await session.commit()
            return deleted

    async def _write_audit(
        self,
        session: "AsyncSession",
        *,
        user_id: str,
        chunk_id: str | None,
        chunk_ids: list[str] | None = None,
        action: str,
        old_snapshot: dict | None = None,
        new_snapshot: dict | None = None,
        affected_count: int | None = None,
    ) -> None:
        """写入审计行。与业务写入共享 session，确保事务原子性。"""
        from app.infrastructure.models.memory_audit_log import MemoryAuditLogModel

        log = MemoryAuditLogModel(
            id=str(uuid.uuid4()),
            user_id=user_id,
            chunk_id=chunk_id,
            chunk_ids=chunk_ids,
            action=action,
            old_snapshot=old_snapshot,
            new_snapshot=new_snapshot,
            affected_count=affected_count,
        )
        session.add(log)
