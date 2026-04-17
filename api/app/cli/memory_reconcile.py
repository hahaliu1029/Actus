"""Memory reconcile CLI — 手动触发 FsReconciler 全库/单用户扫描。

用法：

    python -m app.cli.memory_reconcile                  # 全库 reconcile_all_users：
                                                          先跑全局 pending scan，
                                                          再对每个 user 做 fs-walk
    python -m app.cli.memory_reconcile --user-id <UID>  # 仅单用户 walk（force）
    python -m app.cli.memory_reconcile --pending-only   # 只跑 scan_pending_fs_sync

设计文档 L475-478：API lifespan 启动时只做 ``fs_synced=false`` 的快路径扫描，
per-user fs-walk 走懒式（session 创建触发）。当 operator 怀疑数据视图不一致
时用本 CLI 一次扫干净，不跟 API lifespan 耦合——CLI 独立连 DB，扫完就退。

默认分支"真·全量"：两阶段依次跑——pending backlog → fs-walk。``--pending-only``
只跑第一阶段（快），``--user-id`` 仅修单个用户（不触发全局 pending scan）。

**连接生命周期：** 一次性任务，连接池开成 min=1/max=2 即可；跑完手动 close。
不走 FastAPI lifespan，直接用 ``asyncpg`` 的 async engine。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.infrastructure.external.memory import FsMemoryWriter, FsReconciler
from app.infrastructure.repositories.db_memory_chunk_repository import (
    DBMemoryChunkRepository,
)
from core.config import get_settings

logger = logging.getLogger("app.cli.memory_reconcile")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.memory_reconcile",
        description="Run FsReconciler across all users or a single user.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="若指定，仅对该 user 做 walk；否则全库 reconcile_all_users",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="只跑 scan_pending_fs_sync（快路径），不做 fs-walk",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="scan_pending_fs_sync 每轮取行上限（默认 500）",
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> dict:
    settings = get_settings()
    engine = create_async_engine(
        settings.sqlalchemy_database_url,
        echo=False,
        pool_size=1,
        max_overflow=1,
    )
    session_factory = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
    )

    file_store = FsMemoryWriter(memory_root=settings.memory_root_container)
    reconciler = FsReconciler(
        session_factory=session_factory,
        repo_factory=DBMemoryChunkRepository,
        file_store=file_store,
        memory_root=settings.memory_root_container,
    )

    try:
        if args.pending_only:
            return await reconciler.scan_pending_fs_sync(
                user_id=args.user_id, limit=args.limit
            )
        if args.user_id:
            return await reconciler.walk_user_directory(args.user_id, force=True)
        return await reconciler.reconcile_all_users()
    finally:
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
