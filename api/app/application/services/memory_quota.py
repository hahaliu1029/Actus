"""Per-user daily memory-write quota (M1).

对应设计文档 L546 的"第二层配额"——与 ``rate_limit_write`` dependency（第一层：
IP / 用户维度的短窗口限速）正交。本模块专管 memory 写入的**跨入口**上限：
memory_save 工具、``POST /v2/memories``、未来的文件导入通通走同一个 Redis
counter，防止某个入口被滥用绕开总量限制。

Key schema：``memory:user_daily:{user_id}:{YYYY-MM-DD}``
TTL：首次 INCR 时设置 90000 秒（25h）覆盖跨时区边界，不等 key 到 UTC 午夜过期。

**Fail mode**（设计文档 L406-411）：Redis 异常时 **fail-open** —— 记 warning
log 并放行写入。之所以不 fail-closed：``rate_limit_write`` dependency 已经在
Redis 挂掉时抛 503，真正到这一层时前置限流已经生效；此处再 fail-closed 会
让单点 Redis 故障把 memory 写入端点整体打挂，得不偿失。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from redis.asyncio import Redis

from app.application.errors.exceptions import QuotaExceededError

logger = logging.getLogger(__name__)

# 25h = 90000s，覆盖任意时区下 "今日" → "明日" 的 key 过期对齐问题
_QUOTA_KEY_TTL_SECONDS = 90000


def _today_utc() -> str:
    """YYYY-MM-DD（UTC）。所有 quota key 用同一时间基准，避免多 worker 偏移。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _quota_key(user_id: str, bucket: str = "user_daily") -> str:
    return f"memory:{bucket}:{user_id}:{_today_utc()}"


async def check_and_increment_user_daily(
    redis: Redis,
    user_id: str,
    *,
    daily_cap: int,
) -> int:
    """原子地 INCR 并校验——超过 ``daily_cap`` 抛 ``QuotaExceededError``。

    返回 INCR 后的当前计数（供调用方写日志或计算剩余额度）。

    使用 ``INCR`` → 读返回值 → 视情况抛异常的模式。不走 ``decr`` 回滚：
    1) 超限时 INCR 已经把计数推过 cap，下一次请求会继续拦；
    2) 同一 user 的并发 INCR 本身是原子的，不会出现 "两个请求都看到 cap-1 然后
       都放行把实际值冲到 cap+1" 的经典 race。
    溢出累积的 "技术误差" 会在 24h 后 key 过期自然归零。

    INCR 和 EXPIRE 用 pipeline MULTI/EXEC 打包成一次 round-trip 下发，两条命令
    在 server 端原子执行——避免 "INCR 成功但 EXPIRE 因瞬时网络失败" 留下永不
    过期的 quota key 的 bug（会让计数跨天不归零，产生假限流）。
    每次 INCR 都重设 TTL 是可接受的：key 本身带 ``YYYY-MM-DD`` 后缀按日 rotate，
    sliding TTL 只影响 "当天最后一次写入后的 25h 内 key 是否被提前回收"，不影响
    跨天语义。
    """
    key = _quota_key(user_id)
    try:
        async with redis.pipeline(transaction=True) as pipe:
            pipe.incr(key)
            pipe.expire(key, _QUOTA_KEY_TTL_SECONDS)
            results = await pipe.execute()
        current = int(results[0])
    except Exception as exc:
        # Redis 挂了 → fail-open 放行 + warn；rate_limit_write 会在 Redis 全挂
        # 时先抛 503，这里实际极少命中。pipeline 语义下 INCR/EXPIRE 要么都落库
        # 要么都对 client 不可见，不会出现 "计数递增了但没有 TTL" 的泄漏。
        logger.warning(
            "memory daily quota 检查失败，fail-open 放行: user_id=%s err=%s",
            user_id,
            exc,
        )
        return 0

    # cap 是**包含上界**：current == daily_cap 通过（允许刚好 500 次写入），
    # current == daily_cap + 1 抛错（第 501 次拦下）。调整 `>` ↔ `>=` 会改语义。
    if current > daily_cap:
        # 不清理 counter——让当天剩余请求继续被拦；24h TTL 自动 reset
        raise QuotaExceededError(
            msg=f"每日 memory 写入上限（{daily_cap}）已达到，请明日再试",
            limit=daily_cap,
            bucket="memory_user_daily",
        )
    return current


async def refund_user_daily(redis: Redis, user_id: str) -> None:
    """`check_and_increment_user_daily` 之后若 caller 发现不应计数（如 INSERT
    撞到 ON CONFLICT DO NOTHING 属于"重复提交"而非"成功写入"），调用本函数
    DECR 把之前 INCR 掉的额度还回去。

    使用 Redis DECR——原子、快、允许短暂负值（同一 key 下一次 INCR 会补回）。
    Redis 异常同样 fail-open（仅记 warn，不向上抛），避免 refund 失败传染主
    调用方的返回逻辑。"""
    key = _quota_key(user_id)
    try:
        await redis.decr(key)
    except Exception as exc:
        logger.warning(
            "memory daily quota refund 失败，counter 保留过量值: user_id=%s err=%s",
            user_id,
            exc,
        )
