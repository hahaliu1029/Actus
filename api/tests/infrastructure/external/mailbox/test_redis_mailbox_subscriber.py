"""C2 PR-3 §8.5.3 — RedisMailboxSubscriber unit tests.

Mock Redis; verifies:
- subscribe creates the consumer group (BUSYGROUP idempotent)
- consume yields only predicate-accepted envelopes, always XACKs
- bad JSON → XACK + skip
- max_iterations test seam halts the loop
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from app.infrastructure.external.mailbox.redis_mailbox_subscriber import (
    RedisMailboxSubscriber,
)


@pytest.mark.anyio
async def test_subscribes_root_scoped_stream_creates_group() -> None:
    redis = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)
    await sub.subscribe(
        stream_key="actus:child:root1:mailbox",
        consumer_group="coordinator:child:c1",
        consumer_name="c1-listener",
    )
    redis.xgroup_create.assert_awaited_once()
    kwargs = redis.xgroup_create.await_args.kwargs
    assert kwargs["name"] == "actus:child:root1:mailbox"
    assert kwargs["groupname"] == "coordinator:child:c1"
    assert kwargs["mkstream"] is True
    assert kwargs["id"] == "$"


@pytest.mark.anyio
async def test_subscribe_default_start_id_is_dollar() -> None:
    """Default ``start_id='$'`` is forward-only (skip backlog)."""
    redis = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)
    await sub.subscribe(
        stream_key="actus:child:root1:mailbox",
        consumer_group="g1",
        consumer_name="c1",
    )
    assert redis.xgroup_create.await_args.kwargs["id"] == "$"


@pytest.mark.anyio
async def test_subscribe_with_start_id_zero_reads_backlog() -> None:
    """[r4 P1-1 fix] Rehydrate path uses ``start_id='0'`` so the new consumer
    group sees terminal envelopes already buffered in the stream."""
    redis = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)
    await sub.subscribe(
        stream_key="actus:child:root1:mailbox",
        consumer_group="coordinator:waiter:c1",
        consumer_name="waiter-c1",
        start_id="0",
    )
    assert redis.xgroup_create.await_args.kwargs["id"] == "0"


@pytest.mark.anyio
async def test_subscribe_swallows_busygroup() -> None:
    redis = AsyncMock()
    redis.xgroup_create.side_effect = Exception("BUSYGROUP Consumer Group exists")
    sub = RedisMailboxSubscriber(redis=redis)
    # Must not raise — idempotent re-subscribe.
    await sub.subscribe(
        stream_key="s", consumer_group="g", consumer_name="c",
    )


@pytest.mark.anyio
async def test_subscribe_propagates_other_errors() -> None:
    redis = AsyncMock()
    redis.xgroup_create.side_effect = Exception("connection refused")
    sub = RedisMailboxSubscriber(redis=redis)
    with pytest.raises(Exception) as ei:
        await sub.subscribe(
            stream_key="s", consumer_group="g", consumer_name="c",
        )
    assert "connection refused" in str(ei.value)


@pytest.mark.anyio
async def test_consume_yields_only_matching_envelopes_and_acks_all() -> None:
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        [(
            "actus:child:root1:mailbox",
            [
                ("1-0", {b"envelope": b'{"child_session_id":"c1","type":"CANCEL_REQUEST"}'}),
                ("2-0", {b"envelope": b'{"child_session_id":"c2","type":"CANCEL_REQUEST"}'}),
            ],
        )],
    ])
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return env.get("child_session_id") == "c1"

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="actus:child:root1:mailbox",
        consumer_group="coordinator:child:c1",
        consumer_name="c1-listener",
        predicate=predicate,
        max_iterations=1,
    ):
        received.append(env)

    assert len(received) == 1
    assert received[0]["child_session_id"] == "c1"
    # Both envelopes XACKed regardless of match.
    assert redis.xack.await_count == 2


@pytest.mark.anyio
async def test_consume_string_field_name_works() -> None:
    """redis.asyncio sometimes returns str-keyed fields when decode_responses=True."""
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        [("s", [("1-0", {"envelope": '{"child_session_id":"c1","type":"CANCEL_REQUEST"}'})])],
    ])
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return True

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=predicate, max_iterations=1,
    ):
        received.append(env)
    assert len(received) == 1


@pytest.mark.anyio
async def test_consume_skips_and_acks_bad_json() -> None:
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        [("s", [("9-0", {b"envelope": b"{not json"})])],
    ])
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return True

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=predicate, max_iterations=1,
    ):
        received.append(env)
    assert received == []
    redis.xack.assert_awaited_once_with("s", "g", "9-0")


@pytest.mark.anyio
async def test_consume_max_iterations_halts_loop_when_empty_responses() -> None:
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(return_value=[])  # always empty
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return True

    async for _ in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=predicate, max_iterations=3,
    ):
        pass  # never yields

    # Loop ran exactly 3 iterations then stopped.
    assert redis.xreadgroup.await_count == 3


@pytest.mark.anyio
async def test_consume_isolates_predicate_exception() -> None:
    """[r5 P1-3 fix] A predicate that raises must NOT kill the consume loop.
    The bad message is treated as unmatched (skipped); next message processes normally."""
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        [("s", [
            ("1-0", {b"envelope": b'{"child_session_id":"c1","type":"BOOM"}'}),
            ("2-0", {b"envelope": b'{"child_session_id":"c1","type":"OK"}'}),
        ])],
    ])
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)
    call_count = 0

    async def crashy_predicate(env: dict) -> bool:
        nonlocal call_count
        call_count += 1
        if env.get("type") == "BOOM":
            raise RuntimeError("predicate crashed")
        return True

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=crashy_predicate, max_iterations=1,
    ):
        received.append(env)
    assert call_count == 2  # both predicates ran
    assert len(received) == 1  # only the OK one yielded
    assert received[0]["type"] == "OK"
    # Both messages were XACKed (loop didn't die mid-iteration).
    assert redis.xack.await_count == 2


@pytest.mark.anyio
async def test_consume_isolates_xack_exception() -> None:
    """[r5 P1-3 fix] An xack failure must NOT kill the consume loop."""
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        [("s", [
            ("1-0", {b"envelope": b'{"type":"OK1"}'}),
            ("2-0", {b"envelope": b'{"type":"OK2"}'}),
        ])],
    ])
    # First xack raises, second succeeds.
    redis.xack = AsyncMock(side_effect=[RuntimeError("xack down"), None])
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return True

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=predicate, max_iterations=1,
    ):
        received.append(env)
    # Both messages yielded (predicate True) even though first xack failed.
    assert len(received) == 2


@pytest.mark.anyio
async def test_consume_swallows_xreadgroup_error_and_continues() -> None:
    redis = AsyncMock()
    redis.xreadgroup = AsyncMock(side_effect=[
        Exception("transient redis error"),
        [("s", [("1-0", {b"envelope": b'{"child_session_id":"c1"}'})])],
    ])
    redis.xack = AsyncMock()
    sub = RedisMailboxSubscriber(redis=redis)

    async def predicate(env: dict) -> bool:
        return True

    received: list[dict] = []
    async for env in sub.consume(
        stream_key="s", consumer_group="g", consumer_name="c",
        predicate=predicate, max_iterations=2,
    ):
        received.append(env)
    assert len(received) == 1
    assert received[0]["child_session_id"] == "c1"
