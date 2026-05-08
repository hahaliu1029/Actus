"""B3-core PR-1: RedisStreamMessageQueue.put MAXLEN + first-write EXPIRE.

Spec v3 §3.2 row `task:output:{session_id}`:
- MAXLEN ~2000 (approximate trim)
- TTL 24h (set at first XADD only)

Unit coverage for C-Redis-1 (call shape via mocks); the integration anchor at
tests/integration/test_supervisor_wire_contract.py::test_C_Redis_1_stream_has_maxlen_2000
remains xfail until PR-2 wires the `agent_service_with_redis` fixture.

End-to-end MAXLEN roundtrip lives in
tests/integration/test_long_session_reconnect.py (T10, runs in CI).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def fake_redis():
    """Redis client double exposing exists / xadd / expire as AsyncMocks."""
    fake = MagicMock()
    fake.client = MagicMock()
    fake.client.exists = AsyncMock(return_value=0)
    fake.client.xadd = AsyncMock(return_value=b"1712345678901-0")
    fake.client.expire = AsyncMock()
    fake.client.ttl = AsyncMock(return_value=3600)
    return fake


async def test_put_passes_maxlen_2000_approximate_true(fake_redis):
    """Producer XADD must include `maxlen=2000, approximate=True` (§3.2)."""
    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:abc")
        await queue.put('{"x":1}')

    fake_redis.client.xadd.assert_awaited_once()
    args, kwargs = fake_redis.client.xadd.call_args
    assert args[0] == "task:output:abc"
    assert args[1] == {"data": '{"x":1}'}
    assert kwargs.get("maxlen") == 2000
    assert kwargs.get("approximate") is True


async def test_put_arms_24h_expire_on_first_write(fake_redis):
    """First write (EXISTS=0) → EXPIRE armed at 86400 seconds (§3.2)."""
    fake_redis.client.exists = AsyncMock(return_value=0)

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:first")
        await queue.put('{"first":true}')

    fake_redis.client.exists.assert_awaited_once_with("task:output:first")
    fake_redis.client.expire.assert_awaited_once_with("task:output:first", 86400)


async def test_put_does_not_re_arm_expire_on_subsequent_writes(fake_redis):
    """Subsequent writes with a valid TTL do not refresh stream expiry."""
    fake_redis.client.exists = AsyncMock(return_value=1)  # already exists
    fake_redis.client.ttl = AsyncMock(return_value=3600)

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:existing")
        await queue.put('{"second":true}')

    fake_redis.client.expire.assert_not_awaited()


async def test_put_re_arms_expire_when_existing_stream_has_no_ttl(fake_redis):
    """Existing streams with TTL=-1 are repaired on the next producer write."""
    fake_redis.client.exists = AsyncMock(return_value=1)
    fake_redis.client.ttl = AsyncMock(return_value=-1)

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:no-ttl")
        await queue.put('{"repair":true}')

    fake_redis.client.ttl.assert_awaited_once_with("task:output:no-ttl")
    fake_redis.client.expire.assert_awaited_once_with("task:output:no-ttl", 86400)


async def test_put_returns_xadd_id(fake_redis):
    """put still returns the Redis Stream message id from XADD."""
    fake_redis.client.xadd = AsyncMock(return_value=b"1712345678901-7")

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:return")
        result = await queue.put('{"x":1}')

    assert result == b"1712345678901-7"


async def test_put_continues_when_expire_fails(fake_redis):
    """EXPIRE failure logged but does NOT crash the producer (non-fatal)."""
    fake_redis.client.exists = AsyncMock(return_value=0)
    fake_redis.client.expire = AsyncMock(side_effect=RuntimeError("redis down"))

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:expire-fail")
        # Must not raise:
        result = await queue.put('{"x":1}')

    assert result is not None
    fake_redis.client.expire.assert_awaited_once()


# ---- B3-core PR-1 T9: B5 metric actus_supervisor_event_stream_size_bytes ----

async def test_put_works_with_meter_none(fake_redis):
    """meter=None (legacy callers) — put still works without metric emission."""
    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:no-meter", meter=None)
        rid = await queue.put('{"x":1}')
    assert rid is not None
    assert queue._stream_size_histogram is None


async def test_meter_create_histogram_called_when_meter_provided(fake_redis):
    """meter parameter triggers create_histogram once per instance construction.

    Histogram is cached on the instance, not the class — no reset boilerplate
    required, and parallel tests don't race on shared state.
    """
    fake_meter = MagicMock()
    fake_meter.create_histogram = MagicMock(return_value=MagicMock())

    with patch(
        "app.infrastructure.external.message_queue.redis_stream_message_queue.get_redis",
        return_value=fake_redis,
    ):
        from app.infrastructure.external.message_queue.redis_stream_message_queue import (
            RedisStreamMessageQueue,
        )

        queue = RedisStreamMessageQueue("task:output:test", meter=fake_meter)

    fake_meter.create_histogram.assert_called_once()
    args, kwargs = fake_meter.create_histogram.call_args
    assert kwargs.get("name") == "actus_supervisor_event_stream_size_bytes"
    assert queue._stream_size_histogram is not None
