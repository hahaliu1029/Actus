"""Per-session memory-save 写入限流（M1 PR-3）。

对 ``memory_save`` 工具**专用**的第三层限流——补充于：
  1. ``rate_limit_write`` dependency（IP/用户短窗速率，~/endpoint）
  2. ``memory_quota.check_and_increment_user_daily``（跨入口每日 500 次）
本层只拦 Agent 在**同一个 session**里频繁调用 memory_save 的情况，硬上限
20/session。防止 Agent 在一次对话里无脑保存，把用户 memory 打成噪声池。

Key schema：``memory:session_save:{session_id}:{YYYY-MM-DD}``
TTL：90000s (25h) 覆盖跨时区边界，与 user_daily 对齐。选日维度是为了保证
长 session（持续多天）的 counter 会自然 rotate，不会被一次失控调用永久锁死。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from redis.asyncio import Redis

from app.application.errors.exceptions import QuotaExceededError

logger = logging.getLogger(__name__)

_SESSION_KEY_TTL_SECONDS = 90000


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _session_key(session_id: str) -> str:
    return f"memory:session_save:{session_id}:{_today_utc()}"


async def check_and_increment_session_save(
    redis: Redis,
    session_id: str,
    *,
    cap: int,
) -> int:
    """原子 INCR+EXPIRE——超过 ``cap`` 抛 ``QuotaExceededError``。

    返回 INCR 后的当前计数。Redis 异常 fail-open（返回 0 + warn），和
    ``memory_quota.check_and_increment_user_daily`` 同源设计——不因 Redis
    瞬时故障把 memory_save 工具整条打挂。

    pipeline MULTI/EXEC 保证 INCR 和 EXPIRE 要么都落库要么都对 client 不
    可见，防止 "计数递增了但没有 TTL" 的永久 key 泄漏。
    """
    key = _session_key(session_id)
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, _SESSION_KEY_TTL_SECONDS)
            results = await pipe.execute()
        current = int(results[0])
    except Exception as exc:
        logger.warning(
            "memory session save limit 检查失败，fail-open 放行: session_id=%s err=%s",
            session_id,
            exc,
        )
        return 0

    if current > cap:
        raise QuotaExceededError(
            msg=f"当前 session memory_save 次数已达上限（{cap}），"
            f"请结束当前任务后再试或直接通过 UI 编辑记忆",
            limit=cap,
            bucket="memory_session_save",
        )
    return current


async def refund_session_save(redis: Redis, session_id: str) -> None:
    """调用 ``check_and_increment_session_save`` 之后，如果 caller 发现写入实
    际没有落库（ConflictError 去重 / 用户 daily cap 打穿 / service 异常），用
    本函数 DECR 把 session 计数还回去——否则 20 次 cap 会被失败请求消耗掉。

    Redis 异常同样 fail-open（仅记 warn）——refund 本身失败不该传染主调用方的
    返回语义，session counter 最多当日偏高，24h TTL 自动归零。
    """
    key = _session_key(session_id)
    try:
        await redis.decr(key)
    except Exception as exc:
        logger.warning(
            "memory session save refund 失败，counter 保留过量值: session_id=%s err=%s",
            session_id,
            exc,
        )
