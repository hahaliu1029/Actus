import asyncio
import logging
import math
import time
import uuid
from dataclasses import dataclass
from enum import Enum

from app.application.errors.exceptions import ServiceUnavailableError, TooManyRequestsError
from app.domain.models.user import User
from app.infrastructure.storage.redis import RedisClient, get_redis
from core.config import get_settings
from fastapi import Depends
from redis.asyncio import Redis

from .auth import CurrentUser

logger = logging.getLogger(__name__)


class RateLimitBucket(str, Enum):
    READ = "read"
    WRITE = "write"
    CHAT = "chat"


class RateLimitChannel(str, Enum):
    SSE = "sse"
    WS = "ws"


def _get_limit(bucket: RateLimitBucket) -> int:
    settings = get_settings()
    if bucket == RateLimitBucket.READ:
        return settings.rate_limit_read_per_minute
    if bucket == RateLimitBucket.WRITE:
        return settings.rate_limit_write_per_minute
    return settings.rate_limit_chat_per_minute


def _get_connection_limit(channel: RateLimitChannel) -> int:
    settings = get_settings()
    if channel == RateLimitChannel.SSE:
        return settings.rate_limit_sse_concurrent
    return settings.rate_limit_ws_concurrent


async def enforce_request_limit(
    bucket: RateLimitBucket,
    current_user: User,
    redis_client: RedisClient,
) -> None:
    """请求级限流（固定窗口）"""
    settings = get_settings()
    limit = _get_limit(bucket)
    window_seconds = settings.rate_limit_window_seconds
    window = int(time.time() // window_seconds)
    key = f"rl:req:{bucket.value}:{current_user.id}:{window}"
    redis = redis_client.client

    try:
        current = await redis.incr(key)
        if current == 1:
            await redis.expire(key, window_seconds + 1)
        if current > limit:
            ttl = await redis.ttl(key)
            retry_after = ttl if isinstance(ttl, int) and ttl > 0 else window_seconds
            raise TooManyRequestsError(
                retry_after=retry_after,
                limit=limit,
                window_seconds=window_seconds,
                bucket=bucket.value,
            )
    except TooManyRequestsError:
        raise
    except Exception as exc:
        logger.error(f"请求限流失败: {exc}")
        raise ServiceUnavailableError("限流服务不可用，请稍后重试")


@dataclass
class ConnectionLease:
    """连接并发限制租约"""

    redis: Redis
    key: str
    member: str
    heartbeat_seconds: int
    ttl_seconds: int
    _heartbeat_task: asyncio.Task | None = None
    _released: bool = False

    async def touch(self) -> None:
        await self.redis.zadd(self.key, {self.member: time.time()})
        await self.redis.expire(self.key, self.ttl_seconds)

    def start_heartbeat(self) -> None:
        if self._heartbeat_task is None or self._heartbeat_task.done():
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while not self._released:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await self.touch()
            except Exception as exc:
                logger.warning(f"连接限流心跳更新失败: {exc}")
                break

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        try:
            await self.redis.zrem(self.key, self.member)
        except Exception as exc:
            logger.warning(f"连接限流释放失败: {exc}")


async def acquire_connection_limit(
    channel: RateLimitChannel,
    user_id: str,
    redis_client: RedisClient,
) -> ConnectionLease:
    """连接并发限制（zset + 过期清理）"""
    settings = get_settings()
    limit = _get_connection_limit(channel)
    ttl_seconds = settings.rate_limit_connection_ttl_seconds
    heartbeat_seconds = settings.rate_limit_heartbeat_seconds
    now = time.time()
    key = f"rl:conn:{channel.value}:{user_id}"
    redis = redis_client.client

    try:
        await redis.zremrangebyscore(key, 0, now - ttl_seconds)
        count = await redis.zcard(key)
        if count >= limit:
            oldest = await redis.zrange(key, 0, 0, withscores=True)
            retry_after = heartbeat_seconds
            if oldest and isinstance(oldest[0], (tuple, list)) and len(oldest[0]) == 2:
                oldest_ts = float(oldest[0][1])
                retry_after = max(1, math.ceil(oldest_ts + ttl_seconds - now))
            raise TooManyRequestsError(
                retry_after=retry_after,
                limit=limit,
                window_seconds=ttl_seconds,
                bucket=channel.value,
                msg="并发连接数已达上限，请稍后重试",
            )

        member = str(uuid.uuid4())
        lease = ConnectionLease(
            redis=redis,
            key=key,
            member=member,
            heartbeat_seconds=heartbeat_seconds,
            ttl_seconds=ttl_seconds,
        )
        await lease.touch()
        return lease
    except TooManyRequestsError:
        raise
    except Exception as exc:
        logger.error(f"连接限流失败: {exc}")
        raise ServiceUnavailableError("限流服务不可用，请稍后重试")


async def rate_limit_read(
    current_user: CurrentUser,
    redis_client: RedisClient = Depends(get_redis),
) -> None:
    await enforce_request_limit(
        bucket=RateLimitBucket.READ,
        current_user=current_user,
        redis_client=redis_client,
    )


async def rate_limit_write(
    current_user: CurrentUser,
    redis_client: RedisClient = Depends(get_redis),
) -> None:
    await enforce_request_limit(
        bucket=RateLimitBucket.WRITE,
        current_user=current_user,
        redis_client=redis_client,
    )


async def rate_limit_chat(
    current_user: CurrentUser,
    redis_client: RedisClient = Depends(get_redis),
) -> None:
    await enforce_request_limit(
        bucket=RateLimitBucket.CHAT,
        current_user=current_user,
        redis_client=redis_client,
    )
