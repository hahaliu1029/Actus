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

from app.application.errors.exceptions import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    SecurityError,
    ServiceUnavailableError,
)
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


@dataclasses.dataclass(frozen=True)
class ReindexResult:
    """``MemoryManagementService.reindex_memory`` 返回值。

    - ``reindexed_fields``: 实际被 apply 到 DB 的字段；Option A 只有 ``content``
      或空 list（no-op）。
    - ``warnings``: frontmatter 中 Option A 不支持 / 只读字段被忽略的说明
      列表；前端 dialog 展示给 power user。
    - ``fs_synced``: reindex 后 DB 的 ``fs_synced`` 值（成功翻为 True；
      ``_try_mark_fs_synced`` 失败时可能留 False，reconciler 会补）。
    """

    reindexed_fields: list[str]
    warnings: list[str]
    fs_synced: bool
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
        memory_gate_rollout_at: datetime | None = None,
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
        # ``memory_gate_rollout_at``（codex fix P1）：legacy 清理的时间边界。
        # 非空时 ``delete_legacy_memories`` 额外 AND ``created_at < rollout_at``；
        # 为空时沿用旧谓词（未经 gate 全清），由前端 dialog 显式警告当前
        # deployment 未设 rollout 时间。
        self._memory_gate_rollout_at = memory_gate_rollout_at

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
        auto_promoted_after: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[MemoryChunk], int]:
        """返回 (items, total)。page_size 钳位到 50。

        ``category`` 传入 ``user/rule/fact`` 精确过滤；传 None 返回全部（含 legacy
        的 ``category IS NULL`` 行）。

        ``auto_promoted_after`` 传入 ``datetime`` 仅返回
        ``auto_promoted_at >= auto_promoted_after`` 的行。配合
        ``source='session_flush'`` 即 design doc §777 的"最近自动收录的 memory 审阅"
        路径——manual / memory_save 入口的行 ``auto_promoted_at IS NULL``，自动
        排除在 audit 视图外。
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
                auto_promoted_after=auto_promoted_after,
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
                auto_promoted_after=auto_promoted_after,
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

    async def update_memory_pinned(
        self, user_id: str, chunk_id: str, pinned: bool
    ) -> MemoryChunk | None:
        """切换 chunk 的 pinned 字段。

        **约束**：``pinned=True`` 仅允许在 ``category='user'`` 上（与
        ``create_memory`` 一致 + DB CHECK 兜底）。非 user 类 + pinned=True
        → ``BadRequestError`` (400)；IntegrityError(23514) 也 map 到同一错误
        （防御 DB 约束和 app 校验漂移）。

        **幂等 PATCH 语义**（codex round-11 P1）：PATCH 是"设为目标状态"不是
        toggle——对已 ``pinned=target`` 的行重复请求不写 audit、不 UPDATE、
        直接返现有 chunk。避免 no-op 污染 audit log 噪音 + 无意义刷
        ``updated_at`` 扰乱列表排序。

        **fs 语义**（codex round-11 P1）：不改 content / embedding，走
        ``repo.update_pinned`` 单 SQL UPDATE **保留现有 fs_synced**（不强制
        True——否则会吞 pre-existing ``fs_synced=false`` backlog，让
        reconciler scan_pending 扫不到先前写盘失败的行）。file 层 frontmatter
        的 pinned 字段可能与 DB 漂移——当前 PATCH 不回写文件，漂移由未来
        显式 rewrite/rebuild 路径处理（reindex Option A 也不同步 pinned
        字段回 DB）。

        返回更新后的 chunk；chunk 不存在或越权返 None（route 层 map 404）。
        """
        # 先拉一次 chunk 做 category 校验 + no-op 短路。比让 DB CHECK 抛错
        # 再 map 400 更友好，错误消息能直接说清原因。
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            existing = await repo.get_by_id(chunk_id, user_id)
        if existing is None:
            return None

        # codex round-11 P1 no-op 短路：幂等 PATCH 对已经是目标状态的行
        # 直接返回现有，不碰 DB 也不写 audit。
        if existing.pinned == pinned:
            return existing

        if pinned and existing.category != "user":
            raise BadRequestError(
                f"pinned=True 仅允许 category='user'，当前 category="
                f"{existing.category!r}（DB CHECK 约束：pinned=false "
                f"OR category='user'）"
            )

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            try:
                updated = await repo.update_pinned(
                    chunk_id=chunk_id, user_id=user_id, pinned=pinned,
                )
            except IntegrityError as exc:
                orig = getattr(exc, "orig", None)
                pgcode = (
                    getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
                )
                if pgcode == "23514":
                    # CHECK constraint violation（理论上前置校验已拦，
                    # 这里兜 race / 过期 existing 状态）
                    raise BadRequestError(
                        "pinned=True 仅允许 category='user'（DB CHECK 约束）"
                    ) from exc
                raise

            if updated is None:
                return None

            # 审计：action='pin' 或 'unpin'，old/new snapshot 记 pinned 值
            action = "pin" if pinned else "unpin"
            await self._write_audit(
                session,
                user_id=user_id,
                chunk_id=chunk_id,
                action=action,
                old_snapshot={"pinned": existing.pinned},
                new_snapshot={"pinned": pinned},
            )
            await session.commit()

        return updated

    async def reindex_memory(
        self, user_id: str, chunk_id: str
    ) -> "ReindexResult":
        """从磁盘读 hand-edit 的 memory 文件并同步 DB（Option A：body-only）。

        闭环 hand-edit power-user 工作流：用户编辑 ``${MEMORY_ROOT}/{uid}/
        {cat}/{id}.md`` → POST /reindex → DB content + embedding 立即重算，
        ``memory_search`` / ``memory_recall`` 可查到新内容，不必重启 session
        或跑 CLI reconciler。

        **Option A 权威契约表（codex round-4 收口，四层对齐）：**

        | Frontmatter field                 | 行为       | 原因                              |
        |-----------------------------------|------------|-----------------------------------|
        | body (正文)                       | **apply**  | reindex 本职：同步正文到 DB       |
        | id                                | **409 Conflict** | id 不允许 hand-edit；mismatch 直接报错而非 warn |
        | source / created_at / auto_promoted_at | warning    | 系统字段，被忽略（不 apply）       |
        | title / category / pinned / tags  | warning (file-only) | 留在文件层；**不进 DB / search / prompt** |

        **警告文案必须诚实**：file-only 字段改动后没有受支持的 DB 同步路径
        （PATCH 只收 content；FsReconciler 不回写 frontmatter 到 DB）——
        文案直说"留在文件层"而不是承诺虚假恢复路径。codex round-4 P1
        把这条钉死。

        **错误映射**：
        - DB 无此 chunk → ``NotFoundError`` (404)
        - file_store 未注入 → ``ServiceUnavailableError`` (503) —— deployment
          配置问题不是客户端错（codex round-4 P2）
        - 文件不存在 → ``ConflictError`` (409)
        - frontmatter parse 失败 / 空 body → ``BadRequestError`` (400)
        - frontmatter id 与 DB id 不匹配 → ``ConflictError`` (409)
        - 路径穿越 / symlink → ``SecurityError`` (403)，writer 原样抛出

        **fs_synced 契约（codex round-4 P0 race fix）**：走
        ``repo.reindex_content`` 直接 ``fs_synced=True`` 的 UPDATE，绕开
        ``update_content`` 的 "原子置 False + 后置翻 True" 两阶段。并发
        ``FsReconciler.scan_pending_fs_sync`` 无法捕到 False 窗口 → 不会用
        canonical frontmatter 覆盖 hand-edit。no-op 分支若 existing 行
        ``fs_synced=False``，也补调 ``mark_fs_synced(True)`` 修复 stale 状态。

        ``embedding_provider`` 不可用时降级为 None（与 ``update_memory_content``
        一致，保留文本但暂不可召回）。
        """
        if self._file_store is None:
            # deployment 配置问题（DB-only 模式部署但 UI 启用 reindex）；
            # 503 比 400 更准确——不是客户端请求错（codex round-4 P2）
            raise ServiceUnavailableError(
                "reindex 需要 file_store 后端；当前 deployment 运行在 DB-only 模式"
            )

        # Step 1: 从 DB 取现有 chunk
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            existing = await repo.get_by_id(chunk_id, user_id)
        if existing is None:
            raise NotFoundError("记忆不存在")

        # Legacy 行 category IS NULL → 不能 reindex（没有 fs 路径可读）
        if existing.category is None:
            raise ConflictError(
                "legacy 记忆（category 为空）不支持 reindex；"
                "请先通过 UI / API 分类后再操作"
            )

        # Step 2: 从 fs 读文件 + parse
        try:
            frontmatter, body = await self._file_store.read(
                user_id=user_id,
                memory_id=chunk_id,
                category=existing.category,
            )
        except FileNotFoundError as exc:
            raise ConflictError(
                f"磁盘上不存在此记忆文件（{existing.category}/{chunk_id}.md）；"
                f"可能已被 hand-delete，调用 DELETE /v2/memories/{chunk_id} 清 DB 行"
            ) from exc
        except ValueError as exc:
            raise BadRequestError(f"memory 文件 frontmatter 解析失败：{exc}") from exc
        except NotImplementedError as exc:
            # NoopFileMemoryStore.read() 会抛 NotImplementedError；DI 注入
            # 空实现时语义等同于"不支持 reindex"，映射 503 而非裸 500。
            raise ServiceUnavailableError(
                "当前 file_store 实现不支持 read 操作（DB-only 变体）"
            ) from exc
        # SecurityError 原样抛出到 interfaces 层

        # Step 3: invariant check — frontmatter id 必须与 DB id 一致
        fm_id = frontmatter.get("id")
        if fm_id != chunk_id:
            raise ConflictError(
                f"frontmatter id 不匹配：fs={fm_id!r} DB={chunk_id!r}；"
                f"hand-edit 不允许改 id，建议重启 session 让 reconciler 搬至 .orphans/"
            )

        # Step 4: 对比 body content；空 body → 与 create/update 契约一致拒绝
        body_normalized = body.rstrip("\n")
        if not body_normalized.strip():
            # 与 create_memory / update_memory_content 的 "content must not be
            # empty" 契约对齐——否则 hand-edit 成为唯一绕过不变式的入口
            # （codex round-4 P2）
            raise BadRequestError(
                "reindex 结果 body 为空；与 create/update 契约对齐，拒绝写入 DB"
            )

        # Step 5: 收集 warnings（需要 body_normalized 才能对比 derived title）
        warnings: list[str] = self._collect_reindex_warnings(
            existing, frontmatter, body_normalized
        )

        # Step 6: 对比 body；相同 → no-op 幂等返回（但 existing.fs_synced=False
        # 要修回 True，避免把旧的 stale flag 留给 reconciler 覆盖）
        db_normalized = existing.content.rstrip("\n")
        if body_normalized == db_normalized:
            logger.info(
                "reindex no-op（盘上 body 与 DB 一致）user_id=%s chunk_id=%s fs_synced=%s",
                user_id, chunk_id, existing.fs_synced,
            )
            fs_synced = existing.fs_synced
            if not fs_synced:
                # codex round-4 P0：no-op 也要修 stale fs_synced=False，否则
                # reconciler 下次扫到会用 canonical builder 覆盖 hand-edit。
                fixed = await self._try_mark_fs_synced(
                    user_id=user_id, chunk=existing, synced=True,
                )
                fs_synced = fixed.fs_synced
            return ReindexResult(
                reindexed_fields=[],
                warnings=warnings,
                fs_synced=fs_synced,
            )

        # Step 7: 重算 embedding + UPDATE DB（走 reindex_content 保持 fs_synced=True）
        new_hash = memory_content_hash(body_normalized)
        embedding: tuple[float, ...] | None = None
        try:
            vectors = await self._embedding_provider.embed([body_normalized])
            if vectors:
                embedding = tuple(vectors[0])
        except EmbeddingUnavailableError:
            logger.warning(
                "reindex embedding 降级（provider 不可用）chunk_id=%s",
                chunk_id,
                exc_info=True,
            )

        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            try:
                # codex round-4 P0 race fix：走 reindex_content 单语句把
                # content/hash/embedding/fs_synced=True 一起 UPDATE，**不**
                # 经过 update_content 的 fs_synced=False 窗口。
                updated = await repo.reindex_content(
                    chunk_id=chunk_id,
                    user_id=user_id,
                    content=body_normalized,
                    content_hash=new_hash,
                    embedding=embedding,
                )
            except IntegrityError as exc:
                orig = getattr(exc, "orig", None)
                pgcode = (
                    getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
                )
                if pgcode == "23505":
                    # hand-edit 出了与其它 chunk 内容完全相同的文本 → unique
                    # violation on (user_id, content_hash)。
                    raise ConflictError(
                        "reindex 结果与本用户其它记忆内容重复"
                    ) from exc
                raise

            if updated is None:
                # repo.reindex_content 返 None = chunk_id + user_id 不匹配；
                # Step 1 get_by_id 已确认过——可能刚被并发删了。
                raise NotFoundError("记忆已被并发删除")

            await self._write_audit(
                session,
                user_id=user_id,
                chunk_id=chunk_id,
                action="reindex",
                old_snapshot={
                    "content": existing.content[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                    "content_hash": existing.content_hash,
                },
                new_snapshot={
                    "content": body_normalized[:_AUDIT_CONTENT_PREVIEW_LIMIT],
                    "content_hash": new_hash,
                    "warnings": warnings,
                },
            )
            await session.commit()

        # fs_synced 已经是 True（reindex_content 直接 UPDATE set），不需要
        # 后置 _try_mark_fs_synced(True) —— 这是 P0 race fix 的核心。
        return ReindexResult(
            reindexed_fields=["content"],
            warnings=warnings,
            fs_synced=updated.fs_synced,
        )

    @staticmethod
    def _collect_reindex_warnings(
        existing: MemoryChunk,
        frontmatter: dict,
        body_normalized: str,
    ) -> list[str]:
        """收集 Option A 不 apply 的 frontmatter 改动（codex round-4 收口）。

        权威契约（service docstring / route description / schema / UI 四处对齐）:

        - ``body`` → reindex 的唯一 apply 对象（在本函数外对比）
        - ``id`` → 不在 warnings 里；mismatch 直接 409（reindex_memory Step 3）
        - ``source`` / ``created_at`` / ``auto_promoted_at`` → warnings（系统字段）
        - ``title`` / ``category`` / ``pinned`` / ``tags`` → warnings + file-only

        **诚实文案**（codex round-4 P1）：file-only 字段改动后**没有**受支持的
        DB 同步路径——``PATCH /v2/memories/{id}`` 只收 ``content``，
        ``FsReconciler.walk_user_directory`` 不把这些 frontmatter 字段写回 DB。
        所以 warning 不能承诺虚假恢复路径；直接说"留在文件层，不进 DB /
        search / prompt"。

        ``updated_at`` 不检查——用户 hand-edit 可能改也可能不改，reindex
        完成后服务端会统一刷新（走 DB ``now()``）。
        """
        warnings: list[str] = []

        # ---- 只读系统字段（source / created_at / auto_promoted_at）----
        # id 不在这里：mismatch 已经在调用方直接 409（避免降级为 warning 把
        # 契约冲突掩盖掉）。
        if frontmatter.get("source") not in (None, existing.source):
            warnings.append(
                f"frontmatter.source='{frontmatter['source']}' 与 DB "
                f"'{existing.source}' 不一致；系统字段，已忽略"
            )
        fm_created = frontmatter.get("created_at")
        if fm_created is not None:
            existing_iso = existing.created_at.isoformat()
            if str(fm_created) != existing_iso:
                warnings.append(
                    f"frontmatter.created_at='{fm_created}' 与 DB "
                    f"'{existing_iso}' 不一致；系统字段，已忽略"
                )
        if "auto_promoted_at" in frontmatter:
            fm_ap_raw = frontmatter["auto_promoted_at"]
            fm_ap = str(fm_ap_raw) if fm_ap_raw is not None else None
            db_ap = (
                existing.auto_promoted_at.isoformat()
                if existing.auto_promoted_at is not None
                else None
            )
            if fm_ap != db_ap:
                warnings.append(
                    "frontmatter.auto_promoted_at 改动已忽略（系统字段，仅 LLM gate 写入）"
                )

        # ---- file-only 字段（title / category / pinned / tags）----
        # 核心契约：这些字段仍保留在盘上（File LIVE view 可见），但**不进入**
        # DB / memory_search / prompt snapshot。目前**没有**同步路径——
        # PATCH 只收 content，reconciler walk 也不写回 frontmatter。

        # title：系统真实语义是 derive_title(body_normalized) —— 用户手改的
        # title 即便 body 没变也无法生效（系统重新 derive 会覆盖）
        from app.infrastructure.external.memory.frontmatter import derive_title

        expected_title = derive_title(body_normalized)
        fm_title = frontmatter.get("title")
        if fm_title is not None and str(fm_title) != expected_title:
            warnings.append(
                f"frontmatter.title='{fm_title}' 已忽略；当前系统 title 仍由"
                f" 正文首行派生为 '{expected_title}'，不进入 DB"
            )

        fm_category = frontmatter.get("category")
        if fm_category is not None and fm_category != existing.category:
            warnings.append(
                f"frontmatter.category='{fm_category}' 已忽略；Option A 仅"
                f" 同步正文，category 仍只存在于文件侧，不会进入 DB 索引 /"
                f" search / prompt"
            )
        fm_pinned = frontmatter.get("pinned")
        if fm_pinned is not None and bool(fm_pinned) != bool(existing.pinned):
            warnings.append(
                f"frontmatter.pinned={fm_pinned} 已忽略；Option A 仅同步正文，"
                f"pinned 仍只存在于文件侧，不会进入 DB 索引 / prompt 注入排序"
            )
        fm_tags = frontmatter.get("tags")
        if fm_tags is not None:
            existing_tags = (
                existing.metadata.get("tags", []) if existing.metadata else []
            )
            if list(fm_tags) != list(existing_tags):
                warnings.append(
                    "frontmatter.tags 改动已忽略；Option A 仅同步正文，tags"
                    " 仍只存在于文件侧，不会进入 DB 索引"
                )

        return warnings

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

    async def delete_legacy_memories(self, user_id: str) -> int:
        """一键清理旧 session_flush 遗留行（M3-A）。

        删除条件（仓库侧 AND 合取）：``source='session_flush' AND category
        IS NULL AND auto_promoted_at IS NULL`` —— M1 之前未经 LLM gate 分类、
        也未被系统背书的旧 flush 块。categorized / auto-promoted / manual /
        memory_save 行永远不会进入清理范围。

        Audit 快照存每行 ``content_hash`` 列表（无 PII，可事后查 orphan bug
        或用户误点按钮后追溯）。不存 content preview——legacy 行按设计是
        "可安全丢"的数据，但 hash 保留一层可追溯性。

        返回值：实际删除条数。空时不写审计（与 delete_all_memories 一致）。

        fs 清盘：legacy 行 category IS NULL，本来从未落盘，跳过
        ``file_store.delete``。如果意外遇到非 None category（SQL 条件应已
        排除），写 warn log 但不触发 fs delete——legacy 清理应该只针对
        "确定从未落盘"的行，兜底调 fs 路径反而可能把正常文件当 orphan 删。
        """
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            deleted_rows = await repo.delete_legacy_by_user(
                user_id=user_id,
                rollout_at=self._memory_gate_rollout_at,
            )
            deleted = len(deleted_rows)
            if deleted > 0:
                await self._write_audit(
                    session,
                    user_id=user_id,
                    chunk_id=None,
                    chunk_ids=[c.id for c in deleted_rows],
                    action="delete_legacy",
                    affected_count=deleted,
                    old_snapshot={
                        "content_hashes": [c.content_hash for c in deleted_rows],
                        # 审计保留本次清理的时间边界（None = 未设，沿用旧谓词）。
                        "rollout_at": (
                            self._memory_gate_rollout_at.isoformat()
                            if self._memory_gate_rollout_at is not None
                            else None
                        ),
                    },
                )
                await session.commit()

        # 防御式：legacy 条件 (category IS NULL) 理论保证不会有 non-None
        # category 行进来；若真发生说明 repo 过滤逻辑已被破坏——记 warn，
        # 但不调 fs_store.delete（可能误删正常文件）。
        for row in deleted_rows:
            if row.category is not None:
                logger.warning(
                    "delete_legacy 遇到 category=%s 的行（预期 None），"
                    "可能 repo 过滤逻辑有 bug chunk_id=%s",
                    row.category,
                    row.id,
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
