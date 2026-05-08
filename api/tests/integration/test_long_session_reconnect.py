"""B3-core PR-1 integration: 5000-event reconnect via since_seq.

Validates the full event flow:
  producer (RedisStreamMessageQueue.put with MAXLEN) →
  consumer (RedisEventRecovery.get_recent_events with after_seq) →
  cursor (last_seq derived in agent_service.get_events_since)

The MAXLEN ~2000 trim means events 0..4999 are written but only ~2000 survive.
A reconnecting client with since_seq=4500 should get events 4501..5000.

Spec v3 §3.2 + §3.3 + §6.7.

NOTE on stream reuse: 5000 events at ~150 bytes = ~750KB; well under default Redis
maxmemory. MAXLEN ~2000 keeps actual size bounded.
"""

from __future__ import annotations

import json
import uuid

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_5000_event_reconnect_returns_tail_via_since_seq(redis_client):
    """Push 5000 events; with MAXLEN~2000 only ~2000 latest survive; reconnect with
    since_seq=4500 returns events 4501..5000.
    """
    from app.infrastructure.external.event_recovery.redis_event_recovery import (
        RedisEventRecovery,
    )
    from app.infrastructure.external.message_queue.redis_stream_message_queue import (
        RedisStreamMessageQueue,
    )

    task_id = str(uuid.uuid4())
    stream = f"task:output:{task_id}"
    queue = RedisStreamMessageQueue(stream)

    # Push 5000 events with monotonic seq.
    for i in range(5000):
        evt_json = json.dumps({
            "id": f"e{i}",
            "type": "message",
            "created_at": "2026-05-07T00:00:00",
            "role": "assistant",
            "message": f"m-{i}",
            "seq": i + 1,  # 1..5000
        })
        await queue.put(evt_json)

    # MAXLEN ~2000 — verify trim worked.
    length = await redis_client.xlen(stream)
    assert 1900 <= length <= 2100, f"unexpected trimmed length: {length}"

    # Reconnect at seq=4500 — expect events with seq in 4501..5000 (500 events,
    # well within MAXLEN survivors).
    recovery = RedisEventRecovery(max_count=10000)
    result = await recovery.get_recent_events(
        task_id=task_id, after_event_id=None, after_seq=4500,
    )
    seqs = sorted(e.seq for e in result.events if e.seq is not None)
    # All returned events must have seq > 4500.
    assert all(s > 4500 for s in seqs), f"filter leaked: {seqs[:10]}"
    # And all events ≤ 5000 — no fabrication.
    assert all(s <= 5000 for s in seqs)
    # We expect to recover the tail 4501..5000 entirely (500 events).
    assert seqs == list(range(4501, 5001)), (
        f"missing tail events: got {len(seqs)} starting at {seqs[:3]}"
    )


async def test_reconnect_skips_legacy_seq_none_events(redis_client):
    """Legacy events with seq=None are filtered out when after_seq is given."""
    from app.infrastructure.external.event_recovery.redis_event_recovery import (
        RedisEventRecovery,
    )
    from app.infrastructure.external.message_queue.redis_stream_message_queue import (
        RedisStreamMessageQueue,
    )

    task_id = str(uuid.uuid4())
    stream = f"task:output:{task_id}"
    queue = RedisStreamMessageQueue(stream)

    # Mixed: 5 legacy + 5 with seq.
    for i in range(5):
        await queue.put(json.dumps({
            "id": f"legacy-{i}", "type": "message",
            "created_at": "2026-05-07T00:00:00",
            "role": "assistant", "message": f"legacy-{i}",
        }))
    for i in range(5):
        await queue.put(json.dumps({
            "id": f"new-{i}", "type": "message",
            "created_at": "2026-05-07T00:00:00",
            "role": "assistant", "message": f"new-{i}",
            "seq": i + 1,
        }))

    recovery = RedisEventRecovery(max_count=100)
    result = await recovery.get_recent_events(
        task_id=task_id, after_event_id=None, after_seq=2,
    )
    seqs = sorted(e.seq for e in result.events if e.seq is not None)
    assert seqs == [3, 4, 5]  # exactly the seq>2 with non-None seq
    # No legacy events leak.
    msgs = [e.message for e in result.events if hasattr(e, "message")]
    assert all(not m.startswith("legacy-") for m in msgs)
