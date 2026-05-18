"""Tests for atomic Redis Lua quota: probe slot acquire / release / stale cleanup.

Scope: unit tests with mocks. Real-Redis Lua atomicity (HGETALL iteration,
HEXISTS idempotency, HSET+EXPIRE ordering, fakeredis or live Redis) is
tracked as a follow-up integration test — see PR-3 self-review notes.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.infrastructure.cache.probe_quota import (
    ProbeQuotaService,
    MAX_ACTIVE_PROBES_PER_USER_DEFAULT,
    PROBE_QUOTA_TTL_SECONDS,
)

pytestmark = pytest.mark.anyio  # project convention is anyio (NOT asyncio)


@pytest.fixture
def mock_redis():
    """Mock RedisClient.

    Production code calls `self._redis.client.eval(...)` /
    `self._redis.client.hdel(...)` (RedisClient.client is a @property →
    redis.asyncio.Redis). Tests configure those same attributes — no
    parent-level mirror tricks.
    """
    redis = MagicMock()
    redis.client = MagicMock()
    redis.client.eval = AsyncMock()
    redis.client.hdel = AsyncMock()
    return redis


@pytest.fixture
def quota_service(mock_redis):
    return ProbeQuotaService(
        redis_client=mock_redis,
        max_active=2,
        ttl_seconds=900,
    )


async def test_acquire_empty_key_succeeds(quota_service, mock_redis):
    """Empty key → first acquire returns True; Lua key includes user_id."""
    mock_redis.client.eval.return_value = 1

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-1")

    assert acquired is True
    mock_redis.client.eval.assert_called_once()
    call_args = mock_redis.client.eval.call_args
    assert "actus:active_probes:u-1" in str(call_args)


async def test_acquire_max_active_returns_false(quota_service, mock_redis):
    """When 2 active probes already exist, 3rd acquire returns False."""
    mock_redis.client.eval.return_value = 0  # Lua returns 0 when full

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-3")

    assert acquired is False


@pytest.mark.parametrize(
    "lua_return",
    [
        1,          # int (decode_responses=False edge case)
        "1",        # str (decode_responses=True default)
        b"1",       # bytes (decode_responses=False)
    ],
    ids=["int", "str", "bytes"],
)
async def test_acquire_handles_all_lua_truthy_return_forms(
    quota_service, mock_redis, lua_return
):
    """`bool(int(result))` must coerce 1 / "1" / b"1" → True equally."""
    mock_redis.client.eval.return_value = lua_return

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-1")

    assert acquired is True


@pytest.mark.parametrize(
    "lua_return",
    [0, "0", b"0"],
    ids=["int", "str", "bytes"],
)
async def test_acquire_handles_all_lua_falsy_return_forms(
    quota_service, mock_redis, lua_return
):
    """`bool(int(result))` must coerce 0 / "0" / b"0" → False equally."""
    mock_redis.client.eval.return_value = lua_return

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-1")

    assert acquired is False


async def test_release_calls_hdel(quota_service, mock_redis):
    """Release removes specific probe_run_id from user's active set."""
    await quota_service.release(user_id="u-1", probe_run_id="p-1")

    mock_redis.client.hdel.assert_called_once_with(
        "actus:active_probes:u-1", "p-1"
    )


async def test_acquire_stale_entry_cleaned_then_acquires(quota_service, mock_redis):
    """Stale entry cleanup is verified by Lua logic; service-level contract:
    when Lua returns 1, the service propagates success regardless of any
    pre-existing entries (Lua already cleaned them).
    """
    mock_redis.client.eval.return_value = 1

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-new")

    assert acquired is True


async def test_service_propagates_lua_result_for_each_concurrent_call(
    quota_service, mock_redis
):
    """When 5 concurrent service.acquire() calls happen and the underlying
    Lua mock returns 1 for the first 2 calls and 0 thereafter, the service
    must return exactly 2 True results.

    NOTE: This is a SERVICE-LEVEL propagation test, NOT a Lua atomicity
    test. Real Lua atomicity (HLEN+HSET race-freedom) requires a real
    Redis and is covered as a follow-up integration test.
    """
    call_count = [0]

    async def mock_eval(*args, **kwargs):
        call_count[0] += 1
        return 1 if call_count[0] <= 2 else 0

    mock_redis.client.eval = mock_eval

    results = await asyncio.gather(*[
        quota_service.acquire(user_id="u-1", probe_run_id=f"p-{i}")
        for i in range(5)
    ])

    assert sum(results) == 2  # exactly 2 succeed


async def test_acquire_redis_exception_fail_closed(quota_service, mock_redis):
    """If Redis is down, acquire returns False (fail closed) and logs error.

    Better to deny a legit probe than allow unlimited probes during outage.
    """
    mock_redis.client.eval.side_effect = Exception("Redis connection failed")

    acquired = await quota_service.acquire(user_id="u-1", probe_run_id="p-1")

    assert acquired is False


async def test_release_redis_exception_swallowed(quota_service, mock_redis):
    """release() must swallow Redis errors (best-effort; TTL self-cleans)."""
    mock_redis.client.hdel.side_effect = Exception("Redis connection failed")

    # Must NOT raise. Stale entries self-expire via hash TTL.
    await quota_service.release(user_id="u-1", probe_run_id="p-1")


async def test_constants_match_documented_defaults():
    """Sanity: defaults match the documented contract (DI overrideable)."""
    assert MAX_ACTIVE_PROBES_PER_USER_DEFAULT == 2
    assert PROBE_QUOTA_TTL_SECONDS == 900  # 15 min (>D5 watchdog 600s)
