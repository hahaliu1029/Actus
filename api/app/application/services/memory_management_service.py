"""用户级长期记忆管理 Service。

职责：规范化输入 → 计算 hash → 重算 embedding → 写审计 → 提交事务。

与 MemoryFlushService 的分工：
- Flush：后台异步写路径，RawChunk → MemoryChunk 批量入库
- Management：前台用户 CRUD + 审计（edit / delete / bulk_delete / delete_all）

审计写入与业务写入共享同一 AsyncSession/事务，保证原子性：
业务写成功但审计写失败会整体回滚。
"""
from __future__ import annotations

import dataclasses
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from sqlalchemy.exc import IntegrityError

from app.application.errors.exceptions import ConflictError
from app.application.services.memory_quota import (
    check_and_increment_user_daily,
    refund_user_daily,
)
from app.domain.external.embedding_provider import EmbeddingUnavailableError
from app.domain.models.memory_chunk import MemoryChunk, memory_content_hash

_AUDIT_CONTENT_PREVIEW_LIMIT = 200
# 允许的 manual / memory_save 写入路径枚举；DB CHECK 再兜一次
_ALLOWED_CATEGORIES = frozenset({"user", "rule", "fact"})
_ALLOWED_SOURCES = frozenset({"session_flush", "manual", "memory_save"})

if TYPE_CHECKING:
    from redis.asyncio import Redis
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
        redis: "Redis | None" = None,
        user_daily_quota: int | None = None,
    ) -> None:
        # ``file_store`` 在 PR-0 期间恒为 None（DB-only 模式），PR-5A 起由
        # lifespan 注入真实的 ``FsMemoryWriter``。None 时所有 CRUD 只落 DB，
        # 不碰文件系统，保持现有行为。
        # ``redis`` + ``user_daily_quota`` 必须同传或同不传——防止运维只改
        # config 不改 DI wiring，把配额默默吞掉。DI factory 默认二者齐发，测试
        # 不传 quota 时必须两个都不传。
        if (redis is None) != (user_daily_quota is None):
            raise ValueError(
                "redis 和 user_daily_quota 必须同时提供或同时省略——"
                "二者缺一等于配额关闭，为防止 misconfig 显式报错"
            )
        self._repo_factory = repo_factory
        self._embedding_provider = embedding_provider
        self._session_factory = session_factory
        self._file_store = file_store
        self._redis = redis
        self._user_daily_quota = user_daily_quota

    async def list_memories(
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
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[MemoryChunk], int]:
        """返回 (items, total)。page_size 钳位到 50。

        ``category`` 传入 ``user/rule/fact`` 精确过滤；传 None 返回全部（含 legacy
        的 ``category IS NULL`` 行）。
        """
        page_size = min(page_size, 50)
        page = max(page, 1)
        offset = (page - 1) * page_size

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            items = await repo.list_by_user(
                user_id,
                query=query,
                source=source,
                category=category,
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
                category=category,
                created_from=created_from,
                created_to=created_to,
                updated_from=updated_from,
                updated_to=updated_to,
            )
        return items, total

    async def create_memory(
        self,
        user_id: str,
        content: str,
        category: str,
        *,
        source: str = "manual",
        pinned: bool = False,
        session_id: str | None = None,
    ) -> MemoryChunk:
        """DB-first 写入（设计文档 "Create Flow"）。

        Steps:
        1. 入参校验 + 每日 quota 校验
        2. 计算 content_hash + embedding（embedding 失败降级为 None）
        3. INSERT memory_chunks (fs_synced=false)——若 hash 与本用户已有行撞车
           抛 ConflictError 409
        4. 若注入了 file_store：file_store.write()；成功则 UPDATE fs_synced=true
           （PR-5A 会接上真实的 FsMemoryWriter；在此之前 None 表示 DB-only 模式）

        返回值：创建的 MemoryChunk。

        invariants：
        - ``category`` 必须在 ``{user, rule, fact}`` 内
        - ``source`` 必须在 ``{session_flush, manual, memory_save}`` 内
        - ``pinned=True`` 仅允许在 ``category='user'`` 时——DB CHECK 兜底，这里
          提前校验给调用方一个 ValueError（API 层会 map 成 400）
        """
        content = content.strip()
        if not content:
            raise ValueError("content must not be empty")
        if category not in _ALLOWED_CATEGORIES:
            raise ValueError(
                f"category 必须是 {sorted(_ALLOWED_CATEGORIES)} 之一"
            )
        if source not in _ALLOWED_SOURCES:
            raise ValueError(
                f"source 必须是 {sorted(_ALLOWED_SOURCES)} 之一"
            )
        if pinned and category != "user":
            raise ValueError("pinned=True 仅允许在 category='user' 时使用")

        # Per-user 每日 quota（跨 memory_save / POST / 未来文件导入共享）。
        # ``quota_was_incremented`` 追踪本次是否真的 INCR 过——只有当 Redis
        # 成功 INCR（返回 ≥ 1）时才允许后续 refund；fail-open（返回 0）路径
        # 必须**不**触发 refund，否则会对不存在的 key 做 DECR，把当天计数永久
        # 打到 -1 且没有 TTL，Redis 恢复后后续请求都少算额度。
        quota_was_incremented = False
        if self._redis is not None and self._user_daily_quota is not None:
            current = await check_and_increment_user_daily(
                self._redis,
                user_id,
                daily_cap=self._user_daily_quota,
            )
            quota_was_incremented = current > 0

        # Step 2: embedding 计算——与 update_memory_content 相同的降级策略，
        # provider 故障时写入 None，保留文本但暂不可召回，后续可补算
        content_hash = memory_content_hash(content)
        embedding: tuple[float, ...] | None = None
        try:
            vectors = await self._embedding_provider.embed([content])
            if vectors:
                embedding = tuple(vectors[0])
        except EmbeddingUnavailableError:
            logger.warning(
                "embedding 计算失败（provider 不可用），chunk 以冷数据写入 user_id=%s",
                user_id,
                exc_info=True,
            )

        chunk_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        # fs_synced 一定从 False 起步——FsMemoryWriter 落盘后才翻成 True，
        # 若注入 NoopFileMemoryStore（测试）或保持 None（DB-only），下面的
        # fs-write step 会处理 true 的翻转
        chunk = MemoryChunk(
            id=chunk_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            content_hash=content_hash,
            source=source,
            metadata={},
            created_at=now,
            updated_at=now,
            embedding=embedding,
            category=category,
            auto_promoted_at=None,
            fs_synced=False,
            pinned=pinned,
        )

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            try:
                inserted_count = await repo.batch_insert_ignore([chunk])
                await session.commit()
            except IntegrityError as exc:
                orig = getattr(exc, "orig", None)
                pgcode = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
                if pgcode == "23505":
                    # 冲突的 INSERT **不算**配额消耗——refund 之前的 INCR 以
                    # 防止"客户端自动重试打到配额上限而实际一条都没写进去"的
                    # retry-bomb 攻击模式。仅当本次真的 INCR 过才 refund——
                    # Redis fail-open 路径（INCR 抛异常）不应 DECR 一个本来
                    # 不存在的 key。
                    if quota_was_incremented and self._redis is not None:
                        await refund_user_daily(self._redis, user_id)
                    raise ConflictError("相同内容的长期记忆已存在") from exc
                raise
        if inserted_count == 0:
            # batch_insert_ignore 的 ON CONFLICT DO NOTHING 路径——同 hash 已存在
            if quota_was_incremented and self._redis is not None:
                await refund_user_daily(self._redis, user_id)
            raise ConflictError("相同内容的长期记忆已存在")

        # Step 4: 异步写文件；file_store=None 时保持 fs_synced=False 返回，
        # FsReconciler 或未来 lifespan 注入真实 writer 后再收尾
        if self._file_store is not None:
            try:
                await self._file_store.write(
                    user_id=user_id,
                    memory_id=chunk_id,
                    category=category,
                    content=content,
                    frontmatter=self._build_frontmatter(chunk),
                    overwrite=False,
                )
            except Exception:
                # 写文件失败不 rollback DB——fs_synced=false 让 FsReconciler 重试
                logger.warning(
                    "file_store.write 失败，保留 fs_synced=false 等 reconciler 重试 chunk_id=%s",
                    chunk_id,
                    exc_info=True,
                )
                return chunk

            # 成功写盘 → 翻 fs_synced。这里独立一个 session/事务，如果第二
            # 事务失败（连接抖动、pool exhausted 等）**不能** 把整个请求打成
            # 500——DB 行已经落了（fs_synced=false + 文件也写了），FsReconciler
            # 扫到 false 会重试把 flag 补齐。吞异常 + warning，保持请求成功。
            try:
                async with self._session_factory() as session:
                    repo = self._repo_factory(session)
                    await repo.mark_fs_synced(
                        chunk_id=chunk_id, user_id=user_id, synced=True
                    )
                    await session.commit()
                # 返回反映最新状态的 chunk（frozen dataclass → replace）
                chunk = dataclasses.replace(chunk, fs_synced=True)
            except Exception:
                logger.warning(
                    "mark_fs_synced 失败——文件已落盘但 DB flag 未翻，"
                    "等待 FsReconciler 补写 chunk_id=%s",
                    chunk_id,
                    exc_info=True,
                )

        return chunk

    @staticmethod
    def _build_frontmatter(chunk: MemoryChunk) -> dict:
        """SKILL.md / memory frontmatter canonical set（设计文档 L424）。

        保持与 FsMemoryWriter（PR-5A）期望的字段一致，避免字段漂移。
        依赖 MemoryChunk 的类型保证（created_at/updated_at/metadata 非空）。
        """
        return {
            "id": chunk.id,
            "category": chunk.category,
            "source": chunk.source,
            "created_at": chunk.created_at.isoformat(),
            "updated_at": chunk.updated_at.isoformat(),
            "pinned": chunk.pinned,
            "tags": chunk.metadata.get("tags", []),
        }

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
