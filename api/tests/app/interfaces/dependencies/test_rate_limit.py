import asyncio
import time

import pytest
from app.application.errors.exceptions import ServiceUnavailableError, TooManyRequestsError
from app.domain.models.user import User
from app.interfaces.dependencies.rate_limit import (
    RateLimitBucket,
    RateLimitChannel,
    acquire_connection_limit,
    enforce_request_limit,
)


class FakeRedis:
    def __init__(self) -> None:
        self.fail = False
        self.force_incr: int | None = None
        self.counters: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def incr(self, key: str) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        if self.force_incr is not None:
            return self.force_incr
        value = self.counters.get(key, 0) + 1
        self.counters[key] = value
        return value

    async def expire(self, key: str, seconds: int) -> bool:
        if self.fail:
            raise RuntimeError("redis unavailable")
        self.ttls[key] = seconds
        return True

    async def ttl(self, key: str) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        return self.ttls.get(key, 60)

    async def zremrangebyscore(self, key: str, min_score: float, max_score: float) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        zset = self.zsets.setdefault(key, {})
        to_remove = [m for m, s in zset.items() if min_score <= s <= max_score]
        for member in to_remove:
            del zset[member]
        return len(to_remove)

    async def zcard(self, key: str) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        return len(self.zsets.get(key, {}))

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        zset = self.zsets.setdefault(key, {})
        zset.update(mapping)
        return 1

    async def zrange(self, key: str, start: int, stop: int, withscores: bool = False):
        if self.fail:
            raise RuntimeError("redis unavailable")
        zset = self.zsets.get(key, {})
        items = sorted(zset.items(), key=lambda item: item[1])
        sliced = items[start : stop + 1 if stop >= 0 else None]
        if withscores:
            return sliced
        return [item[0] for item in sliced]

    async def zrem(self, key: str, member: str) -> int:
        if self.fail:
            raise RuntimeError("redis unavailable")
        zset = self.zsets.setdefault(key, {})
        existed = 1 if member in zset else 0
        zset.pop(member, None)
        return existed


class FakeRedisClient:
    def __init__(self, redis: FakeRedis) -> None:
        self.client = redis


def test_request_rate_limit_exceeded_returns_429() -> None:
    redis = FakeRedis()
    redis.force_incr = 121
    user = User(id="u-read")

    with pytest.raises(TooManyRequestsError) as exc:
        asyncio.run(
            enforce_request_limit(
                bucket=RateLimitBucket.READ,
                current_user=user,
                redis_client=FakeRedisClient(redis),
            )
        )

    assert exc.value.status_code == 429
    assert exc.value.data
    assert exc.value.data["bucket"] == "read"


def test_request_rate_limit_redis_failure_returns_503() -> None:
    redis = FakeRedis()
    redis.fail = True
    user = User(id="u-read")

    with pytest.raises(ServiceUnavailableError):
        asyncio.run(
            enforce_request_limit(
                bucket=RateLimitBucket.READ,
                current_user=user,
                redis_client=FakeRedisClient(redis),
            )
        )


def test_connection_limit_exceeded_returns_429() -> None:
    redis = FakeRedis()
    user_id = "u-sse"
    key = f"rl:conn:{RateLimitChannel.SSE.value}:{user_id}"
    now = time.time()
    redis.zsets[key] = {f"conn-{idx}": now for idx in range(10)}

    with pytest.raises(TooManyRequestsError):
        asyncio.run(
            acquire_connection_limit(
                channel=RateLimitChannel.SSE,
                user_id=user_id,
                redis_client=FakeRedisClient(redis),
            )
        )
