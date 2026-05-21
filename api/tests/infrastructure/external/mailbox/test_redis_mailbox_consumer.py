"""C3 PR-2 — RedisMailboxConsumer thin wrapper (spec §5.2-§5.4 + §5.6)."""

import pytest
from datetime import datetime, timezone

from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.external.mailbox.redis_mailbox_consumer import (
    RedisMailboxConsumer,
)
from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
    RedisMailboxPublisher,
)


def _make_env(eid="01HSPYU0000000000000000001"):
    return MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id="01HSPYU0000000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "visibility": "hidden"},
    )


@pytest.mark.anyio
async def test_ensure_group_creates_group_with_mkstream(fake_redis):
    consumer = RedisMailboxConsumer(
        fake_redis, root_session_id="root-1", pod_id="pod-a", instance_id="i1"
    )
    await consumer.ensure_group()
    groups = await fake_redis.xinfo_groups("actus:child:root-1:mailbox")
    # redis-py + fakeredis both return dicts with str keys for the outer
    # xinfo_groups response (the underlying RESP CommandsParser coerces
    # field names to str regardless of decode_responses). Inner values
    # are bytes when decode_responses=False — so the group ``name``
    # value is bytes that needs decoding.
    names = [
        g["name"].decode() if isinstance(g["name"], bytes) else g["name"]
        for g in groups
    ]
    assert "actus:mailbox-supervisor:v1" in names


@pytest.mark.anyio
async def test_ensure_group_idempotent(fake_redis):
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()
    await consumer.ensure_group()  # no raise


@pytest.mark.anyio
async def test_xreadgroup_returns_envelopes(fake_redis):
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    entries = await consumer.read(count=10, block_ms=0)
    assert len(entries) == 1
    _redis_id, parsed = entries[0]
    assert parsed.envelope_id == env.envelope_id


@pytest.mark.anyio
async def test_xack_clears_pel(fake_redis):
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    entries = await consumer.read(count=10, block_ms=0)
    redis_id, _ = entries[0]
    await consumer.ack(redis_id)
    pending = await fake_redis.xpending(
        "actus:child:root-1:mailbox", "actus:mailbox-supervisor:v1"
    )
    # redis-py xpending returns dict with str keys for the summary form
    # (``pending`` / ``min`` / ``max`` / ``consumers``) regardless of
    # ``decode_responses``. After XACK, the pending count is 0.
    assert pending["pending"] == 0


@pytest.mark.anyio
async def test_xautoclaim_returns_idle_entries(fake_redis):
    """Spec §5.6 — startup XAUTOCLAIM picks up PEL idle > threshold."""
    consumer_a = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    consumer_b = RedisMailboxConsumer(fake_redis, "root-1", "pod-b", "i2")
    await consumer_a.ensure_group()
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    # Consumer A reads but doesn't ACK — entry sits in A's PEL slot
    await consumer_a.read(count=1, block_ms=0)
    # Consumer B claims idle entries (min_idle_ms=0 for test)
    claimed = await consumer_b.autoclaim(min_idle_ms=0, count=10)
    assert len(claimed) >= 1
    _, env_b = claimed[0]
    assert env_b.envelope_id == env.envelope_id


@pytest.mark.anyio
async def test_destroy_stream_cleans_up(fake_redis):
    """Spec §5.10 — root terminal cleanup."""
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()
    pub = RedisMailboxPublisher(fake_redis)
    await pub.publish(_make_env())
    await consumer.destroy_stream()
    assert await fake_redis.exists("actus:child:root-1:mailbox") == 0


@pytest.mark.anyio
async def test_destroy_stream_raises_on_wrongtype(fake_redis):
    """Spec §5.10 — destroy_stream MUST NOT silently DEL a key that's
    been corrupted into a non-stream type. WRONGTYPE propagates so the
    integrity violation is visible."""
    # Pre-stage a string key at the stream path (simulates corruption /
    # someone else writing to the same key namespace).
    await fake_redis.set("actus:child:root-corrupt:mailbox", "not-a-stream")

    consumer = RedisMailboxConsumer(
        fake_redis, root_session_id="root-corrupt", pod_id="pod-a", instance_id="i1"
    )
    # Skip ensure_group — we're testing the corruption-detection path.
    from redis.exceptions import ResponseError
    with pytest.raises(ResponseError):
        await consumer.destroy_stream()

    # Verify we did NOT DEL the corrupted key (caller must investigate).
    assert await fake_redis.exists("actus:child:root-corrupt:mailbox") == 1


@pytest.mark.anyio
async def test_read_skips_poison_entries(fake_redis):
    """Spec §5.6 poison path — malformed entries are logged + skipped,
    valid entries in the same batch are still returned. _parse_entries
    must NOT raise out of the batch."""
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()

    # XADD a poison entry directly (no 'envelope' field) — simulates a
    # producer skew or corrupted stream.
    await fake_redis.xadd(
        "actus:child:root-1:mailbox",
        fields={"corrupted": "no envelope field"},
    )

    # Then publish a real envelope.
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)

    entries = await consumer.read(count=10, block_ms=0)
    # Only the valid envelope should come back; poison was logged + dropped.
    assert len(entries) == 1
    _, parsed = entries[0]
    assert parsed.envelope_id == env.envelope_id


@pytest.mark.anyio
async def test_read_normalizes_ids_under_decode_responses_true(fake_redis_decoded):
    """Production shape — RedisClient.client uses decode_responses=True
    (str ids, str field keys). Consumer wrapper must normalize redis_id
    to bytes so the public type contract is stable regardless of the
    underlying Redis client config."""
    consumer = RedisMailboxConsumer(fake_redis_decoded, "root-1", "pod-a", "i1")
    await consumer.ensure_group()
    pub = RedisMailboxPublisher(fake_redis_decoded)
    env = _make_env()
    await pub.publish(env)

    entries = await consumer.read(count=10, block_ms=0)
    assert len(entries) == 1
    redis_id, parsed = entries[0]
    assert isinstance(redis_id, bytes), (
        f"expected bytes id from consumer wrapper, got {type(redis_id).__name__}"
    )
    assert parsed.envelope_id == env.envelope_id

    # And ack should accept the bytes id without conversion.
    await consumer.ack(redis_id)


@pytest.mark.anyio
async def test_autoclaim_normalizes_ids_under_decode_responses_true(fake_redis_decoded):
    """Same invariant for XAUTOCLAIM path."""
    consumer_a = RedisMailboxConsumer(fake_redis_decoded, "root-1", "pod-a", "i1")
    consumer_b = RedisMailboxConsumer(fake_redis_decoded, "root-1", "pod-b", "i2")
    await consumer_a.ensure_group()
    pub = RedisMailboxPublisher(fake_redis_decoded)
    await pub.publish(_make_env())
    await consumer_a.read(count=1, block_ms=0)
    claimed = await consumer_b.autoclaim(min_idle_ms=0, count=10)
    assert len(claimed) >= 1
    redis_id, _ = claimed[0]
    assert isinstance(redis_id, bytes)


@pytest.mark.anyio
async def test_read_skips_malformed_json(fake_redis):
    """Spec §5.6 poison path — malformed JSON in ``envelope`` field is
    caught by model_validate_json and skipped, not propagated."""
    consumer = RedisMailboxConsumer(fake_redis, "root-1", "pod-a", "i1")
    await consumer.ensure_group()

    # XADD with envelope field that is not valid JSON.
    await fake_redis.xadd(
        "actus:child:root-1:mailbox",
        fields={"envelope": b"{this is not json"},
    )

    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)

    entries = await consumer.read(count=10, block_ms=0)
    assert len(entries) == 1
    _, parsed = entries[0]
    assert parsed.envelope_id == env.envelope_id
