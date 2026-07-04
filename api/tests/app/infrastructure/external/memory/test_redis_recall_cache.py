"""B8 PR-3: RedisRecallCache 序列化契约（P-5）。

镜像 test_redis_embedding_cache 的 fail-soft 模式；额外锁定：
payload_version 门 / datetime tz-aware roundtrip（spec R3#2）/ 空 items
payload / key 形态。
"""
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.memory_recall import RecallCachePayload, RecalledMemoryItem
from app.infrastructure.external.memory.redis_recall_cache import (
    PAYLOAD_VERSION,
    RedisRecallCache,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _item(**overrides):
    defaults = dict(
        chunk_id="c1",
        category="fact",
        content="数据库是 PostgreSQL 17",
        created_at=datetime(2026, 6, 1, 10, 30, tzinfo=timezone(timedelta(hours=8))),
        score=0.4231,
    )
    defaults.update(overrides)
    return RecalledMemoryItem(**defaults)


def _fake_redis(store: dict):
    redis = MagicMock()
    redis.get = AsyncMock(side_effect=lambda k: store.get(k))

    async def _set(k, v, ex=None):
        store[k] = v

    redis.set = AsyncMock(side_effect=_set)
    return redis


class TestRoundtrip:
    async def test_set_get_roundtrip_preserves_payload(self):
        store: dict = {}
        cache = RedisRecallCache(_fake_redis(store), ttl=123)
        payload = RecallCachePayload(items=(_item(), _item(chunk_id="c2", category=None)), candidate_count=7)
        await cache.set("sid", "qh", payload)
        got = await cache.get("sid", "qh")
        assert got == payload  # dataclass eq：tz-aware datetime 按时刻比较

    async def test_key_format_and_ttl(self):
        store: dict = {}
        redis = _fake_redis(store)
        cache = RedisRecallCache(redis, ttl=123)
        await cache.set("sid", "qh", RecallCachePayload(items=(), candidate_count=0))
        assert "mem_recall:sid:qh" in store
        assert redis.set.await_args.kwargs["ex"] == 123

    async def test_empty_items_payload_roundtrip(self):
        store: dict = {}
        cache = RedisRecallCache(_fake_redis(store), ttl=60)
        payload = RecallCachePayload(items=(), candidate_count=0)
        await cache.set("sid", "qh", payload)
        assert await cache.get("sid", "qh") == payload


class TestMissAndFailSoft:
    async def test_absent_key_is_none(self):
        cache = RedisRecallCache(_fake_redis({}), ttl=60)
        assert await cache.get("sid", "nope") is None

    async def test_payload_version_mismatch_is_miss(self):
        store = {
            "mem_recall:sid:qh": json.dumps(
                {"payload_version": PAYLOAD_VERSION + 999, "candidate_count": 1, "items": []},
            ),
        }
        cache = RedisRecallCache(_fake_redis(store), ttl=60)
        assert await cache.get("sid", "qh") is None

    async def test_malformed_json_is_miss(self):
        store = {"mem_recall:sid:qh": "{not json"}
        cache = RedisRecallCache(_fake_redis(store), ttl=60)
        assert await cache.get("sid", "qh") is None

    async def test_redis_get_error_is_miss(self):
        redis = MagicMock()
        redis.get = AsyncMock(side_effect=RuntimeError("redis down"))
        cache = RedisRecallCache(redis, ttl=60)
        assert await cache.get("sid", "qh") is None

    async def test_redis_set_error_swallowed(self):
        redis = MagicMock()
        redis.set = AsyncMock(side_effect=RuntimeError("redis down"))
        cache = RedisRecallCache(redis, ttl=60)
        await cache.set("sid", "qh", RecallCachePayload(items=(), candidate_count=0))  # 不抛
