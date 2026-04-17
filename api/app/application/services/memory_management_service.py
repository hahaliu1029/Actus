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
from collections import Counter
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
from app.infrastructure.external.memory.frontmatter import (
    build_memory_frontmatter,
)

_AUDIT_CONTENT_PREVIEW_LIMIT = 200
_TAG_MAX_LENGTH = 64
_TAGS_MAX_COUNT = 20
# 允许的 manual / memory_save 写入路径枚举；DB CHECK 再兜一次
_ALLOWED_CATEGORIES = frozenset({"user", "rule", "fact"})
_ALLOWED_SOURCES = frozenset({"session_flush", "manual", "memory_save"})


def _clean_tags(tags: list[str] | None) -> list[str]:
    """Service 侧的 tags 容错清洗：strip + 丢空 + 去重（保序、大小写敏感）+ 长度截断。

    interfaces 层 ``CreateMemoryRequest._normalize_tags`` 已做主清洗；这里再跑一遍
    是为了覆盖 interfaces 之外的调用者（未来 flush gate / 批量导入路径）——service
    层不能假设 tags 一定来自 API 入口。返回空 list 表示"无 tags"。
    """
    if not tags:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for raw in tags:
        if not isinstance(raw, str):
            continue
        t = raw.strip()[:_TAG_MAX_LENGTH]
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= _TAGS_MAX_COUNT:
            break
    return out


if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from app.domain.external.embedding_provider import EmbeddingProvider
    from app.domain.external.file_memory_store import FileMemoryStore
    from app.domain.external.memory_notification_emitter import (
        MemoryNotificationEmitter,
    )
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
        notification_emitter: "MemoryNotificationEmitter | None" = None,
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
        # ``notification_emitter`` 可选：注入时 fs 写失败路径会发 ``fs_permanent_failure``
        # 通知给用户（design §183 三种 M1 event_type 之一）；未注入时降级为
        # 只写 audit_log，保持 legacy 路径可用，但对外契约里 ``fs_permanent_failure``
        # 就等同于空承诺——生产部署必须通过 DI 注入真实 emitter。
        self._notification_emitter = notification_emitter

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
        tags: list[str] | None = None,
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
        - ``tags`` 可选（设计 L71）：落到 ``metadata.tags``，被 ``_build_frontmatter``
          序列化到磁盘 YAML。已由 ``CreateMemoryRequest`` pydantic 侧做 strip/dedupe/
          长度校验；service 只做容错清洗（非 str / 超长）+ 持久化。
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
        cleaned_tags = _clean_tags(tags)

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
        # metadata 里只在有 tags 时放 "tags" key——避免一堆 {"tags": []} 的噪声
        # 落到 DB。``_build_frontmatter`` 读取时 ``metadata.get("tags", [])`` 兜底。
        metadata: dict = {}
        if cleaned_tags:
            metadata["tags"] = cleaned_tags
        chunk = MemoryChunk(
            id=chunk_id,
            user_id=user_id,
            session_id=session_id,
            content=content,
            content_hash=content_hash,
            source=source,
            metadata=metadata,
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
            fs_written = await self._try_fs_write(
                user_id=user_id,
                chunk=chunk,
                content=content,
                category=category,
                overwrite=False,
                action="create",
            )
            if fs_written:
                chunk = await self._try_mark_fs_synced(
                    user_id=user_id,
                    chunk=chunk,
                    synced=True,
                )

        return chunk

    async def _try_fs_write(
        self,
        *,
        user_id: str,
        chunk: MemoryChunk,
        content: str,
        category: str,
        overwrite: bool,
        action: str,
    ) -> bool:
        """Wrap ``file_store.write`` with audit logging on final failure.

        Returns True on success, False on failure. Failure path writes
        ``memory_audit_log`` with ``action='fs_write_failed'`` (design L426)
        so ops can grep for it + FsReconciler can pick up the
        ``fs_synced=false`` row on next scan.

        ``action`` is forwarded to the audit log field so callers distinguish
        create / update / move failures. The FsMemoryWriter's internal retry
        exhaustion triggers this path only once per logical operation.
        """
        if self._file_store is None:
            return False
        try:
            await self._file_store.write(
                user_id=user_id,
                memory_id=chunk.id,
                category=category,
                content=content,
                frontmatter=self._build_frontmatter(chunk),
                overwrite=overwrite,
            )
            return True
        except Exception as exc:
            logger.warning(
                "file_store.write 失败（action=%s），保留 fs_synced=false 等 reconciler 重试 chunk_id=%s: %s",
                action,
                chunk.id,
                exc,
                exc_info=True,
            )
            error_type = type(exc).__name__
            error_msg = str(exc)[:500]
            await self._write_fs_failure_audit(
                user_id=user_id,
                chunk_id=chunk.id,
                action=action,
                error_type=error_type,
                error_msg=error_msg,
            )
            # fs_permanent_failure notification（design §183）——FsMemoryWriter
            # 内部指数退避重试已经耗尽（max_retries 默认 5），这里把"尝试写失败"
            # 事件暴露给用户通知托盘；FsReconciler 后续扫到 fs_synced=false
            # 可能重试成功，此时通知相当于"已解决的告警"，用户托盘 UI 可按
            # notification 创建时间折旧。schema 声明 fs_permanent_failure 是
            # M1 已知 event_type，这条发射是对那个契约的兑现。
            await self._try_emit_fs_failure_notification(
                user_id=user_id,
                chunk_id=chunk.id,
                category=category,
                action=action,
                error_type=error_type,
                error_msg=error_msg,
            )
            return False

    async def _try_mark_fs_synced(
        self,
        *,
        user_id: str,
        chunk: MemoryChunk,
        synced: bool,
    ) -> MemoryChunk:
        """Flip ``fs_synced`` in its own txn, swallowing failures.

        DB row 已经落了（fs_synced=false + 文件也写了），mark 失败只是 flag
        暂时没翻，FsReconciler 扫到 false 会重试补齐。吞异常 + warning，
        保持请求成功——否则 500 让 UI 看起来像写入失败，但文件实际已落盘，
        用户状态不一致，下次重试撞 ConflictError。
        """
        try:
            async with self._session_factory() as session:
                repo = self._repo_factory(session)
                await repo.mark_fs_synced(
                    chunk_id=chunk.id, user_id=user_id, synced=synced
                )
                await session.commit()
            return dataclasses.replace(chunk, fs_synced=synced)
        except Exception:
            logger.warning(
                "mark_fs_synced 失败——文件状态 %s 但 DB flag 未翻，"
                "等待 FsReconciler 补写 chunk_id=%s",
                "已落盘" if synced else "待补写",
                chunk.id,
                exc_info=True,
            )
            return chunk

    async def _write_fs_failure_audit(
        self,
        *,
        user_id: str,
        chunk_id: str,
        action: str,
        error_type: str,
        error_msg: str,
    ) -> None:
        """Append a ``fs_write_failed`` audit row.

        独立事务——审计写失败不应掩盖原错误 / 不应让调用方感知额外异常。
        设计 L426：retries 耗尽后写 audit，让 ops 有可 grep 的入口。
        """
        try:
            async with self._session_factory() as session:
                await self._write_audit(
                    session,
                    user_id=user_id,
                    chunk_id=chunk_id,
                    action="fs_write_failed",
                    new_snapshot={
                        "failed_op": action,
                        "error_type": error_type,
                        "error_msg": error_msg,
                    },
                )
                await session.commit()
        except Exception:
            logger.warning(
                "fs_write_failed audit 记录本身失败 chunk_id=%s",
                chunk_id,
                exc_info=True,
            )

    async def _try_emit_fs_failure_notification(
        self,
        *,
        user_id: str,
        chunk_id: str,
        category: str,
        action: str,
        error_type: str,
        error_msg: str,
    ) -> None:
        """Emit ``fs_permanent_failure`` notification (design §183 third M1 event_type).

        Notification 是 advisory 的对用户通道，与 audit_log（运维可 grep）
        正交。未注入 emitter 时 no-op——这种部署等同于没兑现 schema 里
        列出的 ``fs_permanent_failure`` 契约，但 legacy 测试/CI 路径能跑。
        Emitter 自身 swallow 内部异常，调用方不需要再加 try/except。
        """
        if self._notification_emitter is None:
            return
        # Defense-in-depth：即使 emitter 实现违反"内部 swallow"契约，service
        # 也不能因为通知失败把 create/update 打成 500——用户实际内容已入库，
        # 告诉不了用户只是丢一条推送，和 audit 一样最多记 warning。
        try:
            await self._notification_emitter.emit(
                user_id=user_id,
                event_type="fs_permanent_failure",
                payload={
                    "chunk_id": chunk_id,
                    "category": category,
                    "action": action,
                    "error_type": error_type,
                    "error_msg": error_msg,
                },
            )
        except Exception:
            logger.warning(
                "fs_permanent_failure 通知发射失败（emitter 违约或底层故障）chunk_id=%s",
                chunk_id,
                exc_info=True,
            )

    # _build_frontmatter 已迁到 infrastructure/external/memory/frontmatter.py
    # 共享（PR-5B）。FsReconciler 重建 orphan DB 行也用同一个 builder，避免
    # "应用层写 vs reconciler 重建"格式漂移。这里保留别名仅为 stable import 路径。
    _build_frontmatter = staticmethod(build_memory_frontmatter)

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

        # Fs sync 路径：``update_content`` 已把 fs_synced 原子翻成 False（单条
        # SQL 合并）。这里独立写盘 + 成功后再把 flag 翻回 True。category 在 M1
        # 不变（PATCH 不允许改 category），复用 ``updated.category``。
        if self._file_store is not None and updated.category is not None:
            fs_written = await self._try_fs_write(
                user_id=user_id,
                chunk=updated,
                content=updated.content,
                category=updated.category,
                overwrite=True,
                action="update",
            )
            if fs_written:
                updated = await self._try_mark_fs_synced(
                    user_id=user_id,
                    chunk=updated,
                    synced=True,
                )
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

        # Fs delete 是 best-effort：DB 是检索主键来源，DB 删掉即对 agent 立刻
        # 不可见；文件残留只是孤儿，由 FsReconciler 下一次扫描清。Legacy 行
        # (category IS NULL) 从未写盘过，直接跳过。
        if self._file_store is not None and actually_deleted.category is not None:
            await self._best_effort_fs_delete(
                user_id=user_id,
                chunk_id=actually_deleted.id,
                category=actually_deleted.category,
            )
        return True

    async def _best_effort_fs_delete(
        self, *, user_id: str, chunk_id: str, category: str
    ) -> None:
        """Idempotent fs-level unlink. Logs but does not raise on failure."""
        if self._file_store is None:
            return
        try:
            await self._file_store.delete(
                user_id=user_id, memory_id=chunk_id, category=category
            )
        except Exception:
            logger.warning(
                "file_store.delete 失败，留作孤儿等 FsReconciler 清 chunk_id=%s",
                chunk_id,
                exc_info=True,
            )

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

        # Fs cleanup 放 DB 事务之外，best-effort。循环一行一行调 delete 而非
        # 批量 API 是因为 FileMemoryStore 协议只暴露单条 delete——M1 范围内不
        # 扩协议。Legacy 行（category IS NULL）从未写盘，直接跳过。
        if self._file_store is not None and deleted_rows:
            for row in deleted_rows:
                if row.category is None:
                    continue
                await self._best_effort_fs_delete(
                    user_id=user_id,
                    chunk_id=row.id,
                    category=row.category,
                )
        return deleted

    async def delete_all_memories(self, user_id: str) -> int:
        """一键清空本用户全部记忆。

        PR-5A 起 repo 返回完整 row 列表——调用方同时用于：
        - 审计 ``source_distribution``（``Counter(c.source for c in rows)``）
        - fs 清盘（遍历 rows 拿 (id, category)）

        repo 侧单条 ``DELETE ... RETURNING *`` 保证两视图出自同一语句，
        避免 READ COMMITTED 并发竞态。
        """
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            deleted_rows = await repo.delete_all_by_user(user_id=user_id)
            deleted = len(deleted_rows)
            if deleted > 0:
                source_dist = dict(Counter(row.source for row in deleted_rows))
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

        # Fs cleanup best-effort。同 bulk_delete，legacy (category IS NULL)
        # 跳过。数量大时也不 Parallelize——避免 thread pool 饱和影响其他请求；
        # delete_all 是用户主动点 "danger zone" 按钮的低频操作，顺序 I/O 可接受。
        if self._file_store is not None and deleted_rows:
            for row in deleted_rows:
                if row.category is None:
                    continue
                await self._best_effort_fs_delete(
                    user_id=user_id,
                    chunk_id=row.id,
                    category=row.category,
                )
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
