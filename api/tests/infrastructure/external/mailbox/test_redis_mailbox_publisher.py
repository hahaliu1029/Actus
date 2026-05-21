"""C3 PR-2 — RedisMailboxPublisher (spec §5.1 + §5.8 Layer 1 + §5.9)."""

import json

import pytest
from datetime import datetime, timezone

from app.domain.external.mailbox_publisher import MailboxPublishOversize
from app.domain.models.mailbox_envelope import (
    APPROVAL_PAYLOAD_MAX_BYTES,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
    RedisMailboxPublisher,
)


def _make_env(envelope_id: str = "01HSPYU0000000000000000001", **kw):
    base = dict(
        envelope_id=envelope_id,
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id="01HSPYU0000000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "visibility": "hidden"},
    )
    base.update(kw)
    return MailboxEnvelope(**base)


@pytest.mark.anyio
async def test_publish_first_time_does_xadd(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    assert len(entries) == 1
    payload = json.loads(entries[0][1][b"envelope"])
    assert payload["envelope_id"] == env.envelope_id


@pytest.mark.anyio
async def test_publish_duplicate_envelope_id_skips_xadd(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    await pub.publish(env)
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    assert len(entries) == 1, "publisher-side dedup must prevent duplicate XADD"


@pytest.mark.anyio
async def test_publish_dedup_key_has_ttl(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    ttl = await fake_redis.ttl(
        f"actus:mailbox:dedup:{env.parent_session_id}:{env.envelope_id}"
    )
    assert 0 < ttl <= 86_400


@pytest.mark.anyio
async def test_publish_uses_maxlen_approx(fake_redis):
    pub = RedisMailboxPublisher(fake_redis, maxlen_approx=100)
    for i in range(200):
        env = _make_env(envelope_id=f"01HSPYU000000000000000{i:04d}")
        await pub.publish(env)
    length = await fake_redis.xlen("actus:child:root-1:mailbox")
    assert length <= 150


@pytest.mark.anyio
async def test_publish_passes_approximate_true_to_xadd(fake_redis, monkeypatch):
    """Hard lock: xadd MUST be called with approximate=True (spec §5.9
    — exact MAXLEN is ~10x more CPU per XADD). Regression test against
    accidental approximate=False."""
    captured: dict[str, object] = {}

    original_xadd = fake_redis.xadd

    async def spy(*args, **kwargs):
        captured["maxlen"] = kwargs.get("maxlen")
        captured["approximate"] = kwargs.get("approximate")
        return await original_xadd(*args, **kwargs)

    monkeypatch.setattr(fake_redis, "xadd", spy)

    pub = RedisMailboxPublisher(fake_redis, maxlen_approx=50)
    await pub.publish(_make_env())

    assert captured["maxlen"] == 50
    assert captured["approximate"] is True


@pytest.mark.anyio
async def test_publish_oversize_payload_raises(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    huge = "x" * (APPROVAL_PAYLOAD_MAX_BYTES + 1)
    # ProgressUpdate schema forbids extra fields, so put oversize into
    # an MailboxEnvelopeType with looser shape OR pick PROGRESS_UPDATE
    # and overflow via tool_call_id / partial_summary. partial_summary is
    # a free-form str on ProgressUpdatePayload — use that.
    env = _make_env(
        payload={
            "kind": "heartbeat",
            "visibility": "hidden",
            "partial_summary": huge,
        }
    )
    with pytest.raises(MailboxPublishOversize):
        await pub.publish(env)

    # Side-effect assertion: size guard fires BEFORE SET NX / XADD, so
    # no Redis state should change. Regression that moves the size check
    # after SET NX would leak a poison TTL key, suppressing a later
    # corrected retry — this assertion locks that ordering.
    assert (
        await fake_redis.exists(
            f"actus:mailbox:dedup:{env.parent_session_id}:{env.envelope_id}"
        )
        == 0
    )
    assert (
        await fake_redis.xlen(f"actus:child:{env.parent_session_id}:mailbox") == 0
    )


@pytest.mark.anyio
async def test_publish_different_root_uses_different_stream(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    env_a = _make_env(parent_session_id="root-a", envelope_id="01HSPYU0a00000000000000000")
    env_b = _make_env(parent_session_id="root-b", envelope_id="01HSPYU0b00000000000000000")
    await pub.publish(env_a)
    await pub.publish(env_b)
    a = await fake_redis.xlen("actus:child:root-a:mailbox")
    b = await fake_redis.xlen("actus:child:root-b:mailbox")
    assert a == 1 and b == 1


@pytest.mark.anyio
async def test_publish_serialization_roundtrip(fake_redis):
    pub = RedisMailboxPublisher(fake_redis)
    env = _make_env()
    await pub.publish(env)
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    payload = json.loads(entries[0][1][b"envelope"])
    rebuilt = MailboxEnvelope.model_validate(payload)
    assert rebuilt.envelope_id == env.envelope_id
    assert rebuilt.producer_role == env.producer_role
    assert rebuilt.type == env.type
