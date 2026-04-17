"""FsReconciler — 修复 DB ↔ filesystem 三视图不一致（M1 PR-5B）。

两条路径各管一头：

1. **启动快路径** ``scan_pending_fs_sync`` — 扫 DB 里 ``fs_synced=false`` 的行，
   调 ``FsMemoryWriter.write`` 把它们同步到盘上。走 partial index
   ``ix_memory_chunks_fs_synced_pending``，O(pending) 而非 O(total)。
   lifespan 里以 best-effort 后台任务起。

2. **每用户懒路径** ``walk_user_directory(user_id)`` — 用户首次创建 session 时
   触发（由 SessionService hook），对 ``${memory_root}/{user_id}`` 做两轮：

   - fs → DB：遍历 ``{user_id}/{category}/*.md``（跳过 ``.orphans/``），读
     frontmatter 取 ``id``，DB 查不到即孤儿文件，移到 ``.orphans/{ts}/``。
     user 目录内任何 symlink 一并隔离（design L655 Case C 防御）。
   - DB → fs：分页列 ``list_by_user``，对 ``fs_synced=true`` 且磁盘不存在的
     行调用 writer.write 重建文件。

**CLI 全量** ``reconcile_all_users`` 取 ``distinct_user_ids`` ∪ ``listdir(memory_root)``
的并集，逐个 walk——覆盖"DB 有 fs 无"和"fs 有 DB 无"两侧。CLI 入口在
``app/cli/memory_reconcile.py``。

**进程内 idempotence：** ``_walked_users`` 缓存当前进程已 walk 过的 user_id，
防止每次用户开新 session 都重扫（单用户单进程只扫一次）。``force=True``
bypass 缓存（CLI 手动触发路径用）。
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncContextManager, Callable

from app.application.errors.exceptions import SecurityError
from app.domain.models.memory_chunk import MemoryChunk
from app.infrastructure.external.memory.frontmatter import (
    build_memory_frontmatter,
    parse_memory_file,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from app.domain.external.file_memory_store import FileMemoryStore
    from app.domain.repositories.memory_chunk_repository import MemoryChunkRepository

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncContextManager["AsyncSession"]]
RepoFactory = Callable[["AsyncSession"], "MemoryChunkRepository"]

# ``.orphans`` 子目录保留名——orphan 文件/symlink 移到这里后 reconciler 自己
# 也要跳过，不然下次扫描会无限嵌套。
_ORPHANS_DIRNAME = ".orphans"


def _orphan_timestamp() -> str:
    """Filesystem-safe ISO timestamp, colons → dashes so路径解析器不把 ``:`` 当
    Windows drive separator / MinIO key分隔。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


class FsReconciler:
    """修复 DB ↔ filesystem 不一致。

    ``session_factory`` 负责开独立的 AsyncSession（reconciler 不能共享调用方
    session —— 它会 commit 自己的 ``mark_fs_synced`` 写，不该带上调用方的其他
    挂起状态）。``repo_factory`` 把 session 包成仓储。
    """

    def __init__(
        self,
        *,
        session_factory: SessionFactory,
        repo_factory: RepoFactory,
        file_store: "FileMemoryStore",
        memory_root: str | os.PathLike[str],
    ) -> None:
        self._session_factory = session_factory
        self._repo_factory = repo_factory
        self._file_store = file_store
        self._memory_root = Path(os.fspath(memory_root)).expanduser()
        self._walked_users: set[str] = set()
        self._walk_lock = asyncio.Lock()
        self._db_page_size = 200

    # ── 启动快路径 ──────────────────────────────────────────────────────

    async def scan_pending_fs_sync(
        self,
        *,
        user_id: str | None = None,
        limit: int = 500,
    ) -> dict:
        """扫 ``fs_synced=false`` 的行，逐条 writer.write → mark_fs_synced(True)。

        writer 失败的行保持 ``fs_synced=false``，下一轮继续捡——不在这里重试，
        避免阻塞 lifespan。返回 ``{attempted, succeeded, failed}`` 供 ops 观测。
        """
        async with self._open_repo() as (repo, session):
            pending = await repo.find_pending_fs_sync(user_id=user_id, limit=limit)
            attempted = len(pending)
            succeeded = 0
            failed = 0
            for chunk in pending:
                ok = await self._write_and_mark(chunk, repo)
                if ok:
                    succeeded += 1
                else:
                    failed += 1
            if attempted:
                # 即使全部失败也 commit —— ``_write_and_mark`` 成功路径已写
                # DB（mark_fs_synced=True）；若 succeeded>0 提交这些持久化。
                # succeeded=0 时 commit 是 no-op，免得漏 commit 导致 mark
                # 回滚（下轮 scan 重复写盘）。
                await session.commit()
        logger.info(
            "FsReconciler scan_pending_fs_sync user=%s attempted=%d succeeded=%d failed=%d",
            user_id,
            attempted,
            succeeded,
            failed,
        )
        return {"attempted": attempted, "succeeded": succeeded, "failed": failed}

    async def _write_and_mark(
        self, chunk: MemoryChunk, repo: "MemoryChunkRepository"
    ) -> bool:
        if chunk.category is None:
            # category NULL 是 M1 前的 legacy 行；没有 category 无法拼路径。
            # 被 m2 migration 已经标为 fs_synced=true，但防御性处理 false 情况。
            logger.info(
                "FsReconciler 跳过无 category 的 pending chunk_id=%s",
                chunk.id,
            )
            return False
        try:
            await self._file_store.write(
                user_id=chunk.user_id,
                memory_id=chunk.id,
                category=chunk.category,
                content=chunk.content,
                frontmatter=build_memory_frontmatter(chunk),
                overwrite=True,
            )
        except (OSError, SecurityError) as exc:
            logger.warning(
                "FsReconciler 写盘失败 chunk_id=%s err=%s (将保持 fs_synced=false)",
                chunk.id,
                exc,
            )
            return False
        await repo.mark_fs_synced(
            chunk_id=chunk.id, user_id=chunk.user_id, synced=True
        )
        return True

    # ── 每用户懒路径 ────────────────────────────────────────────────────

    async def walk_user_directory(
        self,
        user_id: str,
        *,
        force: bool = False,
    ) -> dict:
        """对 ``{root}/{user_id}`` 做一次孤儿扫描 + 重建。

        ``force=False`` 时，进程内已 walk 过一次就直接返回 ``{"walked": False}``；
        ``force=True`` 绕过缓存（CLI 路径、测试用）。
        """
        async with self._walk_lock:
            if not force and user_id in self._walked_users:
                return {
                    "walked": False,
                    "orphan_files": 0,
                    "orphan_symlinks": 0,
                    "rebuilt_files": 0,
                }

        user_dir = self._memory_root / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        orphan_files = 0
        orphan_symlinks = 0
        rebuilt_files = 0

        async with self._open_repo() as (repo, session):
            orphan_files, orphan_symlinks = await self._scan_fs_orphans(
                user_dir=user_dir, user_id=user_id, repo=repo
            )
            rebuilt_files, db_dirty = await self._rebuild_missing_files(
                user_dir=user_dir, user_id=user_id, repo=repo
            )
            if db_dirty:
                # rebuild 路径无论成功（无 DB 写）还是失败（mark_fs_synced(False)
                # 翻 flag）都可能触发 DB 变更；只要 _rebuild_missing_files
                # 动过仓储就 commit，不然 mark(False) 静默回滚，下轮 scan 不
                # 会重试，数据视图一直 out-of-sync。
                await session.commit()

        async with self._walk_lock:
            self._walked_users.add(user_id)

        logger.info(
            "FsReconciler walk_user_directory user=%s orphan_files=%d "
            "orphan_symlinks=%d rebuilt_files=%d",
            user_id,
            orphan_files,
            orphan_symlinks,
            rebuilt_files,
        )
        return {
            "walked": True,
            "orphan_files": orphan_files,
            "orphan_symlinks": orphan_symlinks,
            "rebuilt_files": rebuilt_files,
        }

    async def _scan_fs_orphans(
        self,
        *,
        user_dir: Path,
        user_id: str,
        repo: "MemoryChunkRepository",
    ) -> tuple[int, int]:
        orphan_files = 0
        orphan_symlinks = 0
        # 一次 walk round 内所有隔离物共用同一个 .orphans/{ts}/，便于 operator
        # 按"这轮清理出的"对账；不用每次都拼新时间戳。
        orphan_bucket = _orphan_timestamp()

        def _iter_entries() -> list[Path]:
            """线程池跑 os.walk，避免阻塞 event loop（fs 可能很大）。"""
            out: list[Path] = []
            for entry in user_dir.iterdir():
                if entry.name == _ORPHANS_DIRNAME:
                    continue
                if entry.is_symlink():
                    out.append(entry)
                    continue
                if entry.is_dir():
                    for sub in entry.rglob("*"):
                        if sub.is_symlink() or (sub.is_file() and sub.suffix == ".md"):
                            out.append(sub)
            return out

        entries = await asyncio.to_thread(_iter_entries)
        for entry in entries:
            if entry.is_symlink():
                await self._move_to_orphans(entry, user_dir, bucket=orphan_bucket)
                orphan_symlinks += 1
                continue
            try:
                content = await asyncio.to_thread(entry.read_text, encoding="utf-8")
                frontmatter, _ = parse_memory_file(content)
            except (OSError, ValueError) as exc:
                # 解析失败的文件也当孤儿——写坏了的文件不能留着给 sandbox 吃。
                logger.warning(
                    "FsReconciler 解析 memory 文件失败 path=%s err=%s; 移入 .orphans",
                    entry,
                    exc,
                )
                await self._move_to_orphans(entry, user_dir, bucket=orphan_bucket)
                orphan_files += 1
                continue

            memory_id = frontmatter.get("id")
            if not isinstance(memory_id, str) or not memory_id:
                await self._move_to_orphans(entry, user_dir, bucket=orphan_bucket)
                orphan_files += 1
                continue

            chunk = await repo.get_by_id(memory_id, user_id)
            if chunk is None:
                await self._move_to_orphans(entry, user_dir, bucket=orphan_bucket)
                orphan_files += 1

        return orphan_files, orphan_symlinks

    async def _rebuild_missing_files(
        self,
        *,
        user_dir: Path,
        user_id: str,
        repo: "MemoryChunkRepository",
    ) -> tuple[int, bool]:
        """返回 ``(rebuilt, db_dirty)``：``db_dirty=True`` 说明调用过
        ``mark_fs_synced(False)``（写失败翻 flag），上层应 commit。"""
        rebuilt = 0
        db_dirty = False
        offset = 0
        while True:
            chunks = await repo.list_by_user(
                user_id, offset=offset, limit=self._db_page_size
            )
            if not chunks:
                break
            for chunk in chunks:
                if chunk.category is None:
                    continue  # legacy 行 —— 没有 category 无法拼路径
                if not chunk.fs_synced:
                    # 交给 scan_pending_fs_sync 处理，避免两条路径抢写
                    continue
                target = user_dir / chunk.category / f"{chunk.id}.md"
                if target.exists():
                    continue
                try:
                    await self._file_store.write(
                        user_id=chunk.user_id,
                        memory_id=chunk.id,
                        category=chunk.category,
                        content=chunk.content,
                        frontmatter=build_memory_frontmatter(chunk),
                        overwrite=False,
                    )
                except (OSError, SecurityError) as exc:
                    logger.warning(
                        "FsReconciler 重建 orphan DB 行失败 chunk_id=%s err=%s",
                        chunk.id,
                        exc,
                    )
                    # 文件写不出 → 把 fs_synced 翻回 false，让 scan_pending 下次捡
                    await repo.mark_fs_synced(
                        chunk_id=chunk.id, user_id=chunk.user_id, synced=False
                    )
                    db_dirty = True
                    continue
                rebuilt += 1
            if len(chunks) < self._db_page_size:
                break
            offset += self._db_page_size
        return rebuilt, db_dirty

    # ── CLI 全量 ────────────────────────────────────────────────────────

    async def reconcile_all_users(self) -> dict:
        """CLI 入口：DB distinct user_ids ∪ fs listdir 并集，逐个 force-walk。"""
        async with self._open_repo() as (repo, _):
            db_users = await repo.distinct_user_ids()

        fs_users: list[str] = []
        if self._memory_root.exists():
            for entry in await asyncio.to_thread(
                lambda: list(self._memory_root.iterdir())
            ):
                if entry.is_dir() and not entry.is_symlink():
                    fs_users.append(entry.name)

        all_users = sorted(set(db_users) | set(fs_users))
        per_user: list[dict] = []
        for user_id in all_users:
            summary = await self.walk_user_directory(user_id, force=True)
            summary["user_id"] = user_id
            per_user.append(summary)

        return {
            "users_walked": len(all_users),
            "user_ids": all_users,
            "per_user": per_user,
        }

    # ── Helpers ─────────────────────────────────────────────────────────

    @asynccontextmanager
    async def _open_repo(self):
        """yield (repo, session) —— session 由 reconciler 自己管生命周期。"""
        async with self._session_factory() as session:
            repo = self._repo_factory(session)
            yield repo, session

    async def _move_to_orphans(
        self, source: Path, user_dir: Path, *, bucket: str
    ) -> None:
        """把 ``source`` 原子 rename 到 ``{user_dir}/.orphans/{bucket}/<name>``。

        ``bucket`` 由上层 walk 统一注入（一次 walk round 共用一个时间戳目录），
        便于 operator 按时间点对账。rename 跨 inode 会降级为 copy+unlink，但
        user_dir 内正常 case 是同一 mount，rename 是 atomic。
        """
        orphans_root = user_dir / _ORPHANS_DIRNAME / bucket

        def _do_move() -> None:
            orphans_root.mkdir(parents=True, exist_ok=True)
            dest = orphans_root / source.name
            counter = 0
            while dest.exists():
                counter += 1
                dest = orphans_root / f"{source.stem}.{counter}{source.suffix}"
            os.replace(source, dest)

        try:
            await asyncio.to_thread(_do_move)
        except OSError as exc:
            logger.warning(
                "FsReconciler 移动 orphan 失败 source=%s err=%s", source, exc
            )
