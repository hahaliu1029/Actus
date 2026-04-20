"""R5 CS4 backfill CLI —— 将旧 ``tool_approval_rules`` 迁成新 ``tool_approval_grants``。

用法：

    uv run python -m app.cli.backfill_approval_grants
    uv run python -m app.cli.backfill_approval_grants --batch 500
    uv run python -m app.cli.backfill_approval_grants --dry-run

设计文档 §Distribution Plan step 3：**脱离 alembic chain**，手工触发。
- 线上 ``main.py`` 启动只跑 DDL (``alembic upgrade head``)，不跑数据迁移
- 本脚本可任意时间执行，可重跑（``WHERE NOT EXISTS`` 去重）
- 使用独立 async engine，跑完关闭

字段映射（design doc §Backfill）：

========================  ==============================
tool_approval_rules.id   → 生成新 decision_id (uuid4)
.user_id                 → user_id（pass-through）
.tool_name               → tool_name
.command_pattern         → primary_arg
.dir_pattern             → dir_arg
.rule='always_deny'      → effect='deny'（其余为 'approve'）
scope                    ='always'（旧规则都是全局）
source_type              ='user_click'
tool_source              = resolve_tool_source(tool_name) 或 'native'
session_id               = NULL
expires_at               = NULL（always scope 永不过期）
arg_digest               = ''（legacy 无 digest 概念）
confirmation_id          = NULL
.created_at              → created_at（保留原始时间戳）
========================  ==============================

``tool_source`` 未知名的兜底（``ToolSourceUnknownError``）降级为 ``'native'``
并 log warning，避免 backfill 卡在个别旧工具名上。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from typing import Any, Mapping

import sqlalchemy as sa
from sqlalchemy import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.domain.services.approval_grant_policy import to_naive_utc
from app.domain.services.tools.tool_source_resolver import (
    ToolSourceUnknownError,
    resolve_tool_source,
)
from app.infrastructure.models.tool_approval_grant import ToolApprovalGrantModel
from core.config import get_settings

logger = logging.getLogger("app.cli.backfill_approval_grants")

DEFAULT_BATCH = 1000


def _resolve_or_default(tool_name: str) -> str:
    """``tool_source`` 未知时兜底 ``'native'``，log 警告。"""
    try:
        return resolve_tool_source(tool_name).source
    except ToolSourceUnknownError:
        logger.warning(
            "tool_name=%r 不在 tool registry，backfill 降级 tool_source='native'",
            tool_name,
        )
        return "native"


def _build_payload_row(row: Mapping[str, Any]) -> dict[str, Any]:
    effect = "deny" if row["rule"] == "always_deny" else "approve"
    return {
        "decision_id": str(uuid.uuid4()),
        "user_id": row["user_id"],
        "session_id": None,
        "tool_name": row["tool_name"],
        "tool_source": _resolve_or_default(row["tool_name"]),
        "arg_digest": "",
        "primary_arg": row["command_pattern"],
        "dir_arg": row["dir_pattern"] or "",
        "scope": "always",
        "effect": effect,
        "source_type": "user_click",
        "confirmation_id": None,
        "expires_at": None,
        # 显式归一 naive UTC：防止 raw-text SELECT 的 asyncpg 返回 aware datetime
        # 后再下推到 TIMESTAMP WITHOUT TIME ZONE 列时走"local→UTC"漂移（Codex
        # round-3 HIGH）。读取侧已加 .columns(created_at=sa.DateTime())，这里是
        # 纵深防御：即便将来读取侧变动，payload 边界仍然保证 naive UTC 语义。
        "created_at": to_naive_utc(row["created_at"]),
    }


_SELECT_LEGACY_SQL = sa.text(
    """
    SELECT
        r.id,
        r.user_id,
        r.tool_name,
        r.rule,
        r.command_pattern,
        r.dir_pattern,
        r.created_at
    FROM tool_approval_rules r
    WHERE NOT EXISTS (
        SELECT 1
        FROM tool_approval_grants g
        WHERE g.user_id = r.user_id
          AND g.tool_name = r.tool_name
          AND g.primary_arg = r.command_pattern
          AND g.dir_arg = COALESCE(NULLIF(r.dir_pattern, ''), '')
          AND g.scope = 'always'
    )
    LIMIT :batch
    """
).columns(
    # 显式 column types（Codex round-3 HIGH）：raw sa.text() SELECT 下 asyncpg 按
    # 默认推断返回类型，TIMESTAMP WITHOUT TIME ZONE 可能被识别成 TIMESTAMPTZ 并返 aware
    # datetime。明示 DateTime（无 timezone）强制 asyncpg 返 naive datetime，
    # 避免写回时的"local→UTC"漂移 bug。
    id=sa.String(255),
    user_id=sa.String(255),
    tool_name=sa.String(255),
    rule=sa.String(32),
    command_pattern=sa.String(512),
    dir_pattern=sa.String(512),
    created_at=sa.DateTime(),
)

def _grants_insert_stmt():
    """SQLAlchemy Core typed insert。

    关键：使用 ORM 映射的 Core ``insert()`` 而非 raw ``sa.text()``，让
    SQLAlchemy + asyncpg 在绑定时知道 ``created_at`` 是 ``TIMESTAMP WITHOUT
    TIME ZONE``（Codex HIGH-1：raw text 下 asyncpg 会把 naive datetime 按
    TIMESTAMPTZ 绑定，结果被服务端做 local→UTC 漂移，8h 偏移 bug）。
    """
    return insert(ToolApprovalGrantModel.__table__)


async def run_backfill(
    session: AsyncSession,
    batch: int,
    dry_run: bool,
    *,
    commit_each_batch: bool = True,
) -> int:
    """执行 backfill 循环。返回处理的行数（不去重前）。

    循环直到 ``WHERE NOT EXISTS`` 查询返回 0 行；每轮最多 ``batch`` 条。

    ``commit_each_batch``:
      - True（CLI 默认）：每批 INSERT 后立即 ``session.commit()``，失败时已写入的
        批次不回滚（符合独立脚本"渐进式推进"的运维语义）
      - False：不 commit；由调用方控制事务边界（集成测试把 backfill 放在
        fixture 的 outer transaction 里，跑完统一 rollback 清盘）
    """
    total = 0
    while True:
        rows = (
            await session.execute(_SELECT_LEGACY_SQL, {"batch": batch})
        ).mappings().all()
        if not rows:
            break
        payload = [_build_payload_row(r) for r in rows]
        logger.info("本轮预计迁移 %d 条 legacy rule → grant", len(payload))
        if dry_run:
            total += len(payload)
            # dry-run 模式不 INSERT，跳出循环避免 WHERE NOT EXISTS 死循环
            logger.info("dry-run 模式，已打印第一批 %d 条，退出", len(payload))
            break
        await session.execute(_grants_insert_stmt(), payload)
        if commit_each_batch:
            await session.commit()
        else:
            # 让调用方的 INSERT 对后续 WHERE NOT EXISTS 可见
            await session.flush()
        total += len(payload)
        if len(rows) < batch:
            break
    return total


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli.backfill_approval_grants",
        description="Migrate legacy tool_approval_rules to R5 tool_approval_grants.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help="每轮读取的 legacy 行数上限（默认 1000）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只读取、不写入；打印第一批记录数后退出",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="日志级别（DEBUG/INFO/WARNING/ERROR）",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    settings = get_settings()
    engine = create_async_engine(
        settings.sqlalchemy_database_url,
        echo=False,
        pool_size=1,
        max_overflow=1,
    )
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    try:
        async with session_factory() as session:
            count = await run_backfill(session, args.batch, args.dry_run)
    finally:
        await engine.dispose()

    logger.info(
        "Backfill %s，处理 %d 条 legacy rule",
        "dry-run 完成" if args.dry_run else "完成",
        count,
    )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
