"""B3-core PR-1: RedisEventRecovery.get_recent_events seq filter.

Spec v3 §3.3 + §6.7. Pure-mock unit-level coverage; end-to-end MAXLEN+seq
roundtrip lives in tests/integration/test_long_session_reconnect.py (T10).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.domain.models.event import MessageEvent
from app.infrastructure.external.event_recovery.redis_event_recovery import (
    RedisEventRecovery,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _evt_json(seq: int | None, message: str = "m") -> str:
    """Build a synthetic event JSON with an optional seq."""
    return MessageEvent(role="assistant", message=message, seq=seq).model_dump_json()


def _mock_queue_with_entries(entries):
    """Build a queue mock whose get_range yields the given (id, json) entries."""
    mock_queue = AsyncMock()

    async def mock_get_range(start_id="-", end_id="+", count=100):
        for mid, payload in entries:
            yield mid, payload

    mock_queue.get_range = mock_get_range
    return mock_queue


async def test_after_seq_filters_to_strictly_greater_seq():
    """after_seq=5 returns only events with event.seq > 5; legacy seq=None excluded."""
    entries = [
        (f"170000{i:04d}-0", _evt_json(seq=i + 1, message=f"hello-{i}"))
        for i in range(10)  # seq 1..10
    ] + [
        ("9999999999-0", _evt_json(seq=None, message="legacy")),
    ]
    mock_queue = _mock_queue_with_entries(entries)

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        recovery = RedisEventRecovery(max_count=100)
        result = await recovery.get_recent_events(
            task_id="task-123", after_event_id=None, after_seq=5
        )

    seqs = sorted(e.seq for e in result.events if e.seq is not None)
    assert seqs == [6, 7, 8, 9, 10], seqs


async def test_after_seq_filters_seq_events_when_event_id_is_also_present():
    """When both are given, after_seq filters sequenced events."""
    cursor_id = "1700000000-0"
    entries = [
        (f"170000000{i}-0", _evt_json(seq=i + 1, message=f"m-{i}"))
        for i in range(5)  # seq 1..5
    ]
    mock_queue = _mock_queue_with_entries(entries)

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        recovery = RedisEventRecovery(max_count=100)
        result = await recovery.get_recent_events(
            task_id="task-123",
            after_event_id=cursor_id,
            after_seq=2,
        )
    assert sorted(e.seq for e in result.events) == [3, 4, 5]


async def test_after_seq_with_event_id_keeps_legacy_events_after_cursor():
    """Mixed streams keep seq=None entries after the Redis Stream cursor."""
    cursor_id = "1700000000-0"
    entries = [
        (cursor_id, _evt_json(seq=1, message="cursor")),
        ("1700000001-0", _evt_json(seq=None, message="legacy-after-cursor")),
        ("1700000002-0", _evt_json(seq=2, message="seq-after-cursor")),
    ]
    mock_queue = _mock_queue_with_entries(entries)

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        recovery = RedisEventRecovery(max_count=100)
        result = await recovery.get_recent_events(
            task_id="task-123", after_event_id=cursor_id, after_seq=1
        )

    assert [getattr(e, "message", "") for e in result.events] == [
        "legacy-after-cursor",
        "seq-after-cursor",
    ]
    assert result.events[0].seq is None
    assert result.events[1].seq == 2


async def test_after_seq_none_falls_back_to_legacy_event_id():
    """after_seq=None (default) preserves legacy behavior — uses after_event_id."""
    cursor_id = "1700000000-0"
    entries = [
        (cursor_id, _evt_json(seq=1, message="cursor")),  # SKIPPED (xrange inclusive)
        ("1700000001-0", _evt_json(seq=2, message="m-1")),
        ("1700000002-0", _evt_json(seq=3, message="m-2")),
    ]
    mock_queue = _mock_queue_with_entries(entries)

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        recovery = RedisEventRecovery(max_count=100)
        result = await recovery.get_recent_events(
            task_id="task-123", after_event_id=cursor_id
        )

    assert len(result.events) == 2
    assert result.events[0].id == "1700000001-0"
    assert result.events[1].id == "1700000002-0"


async def test_after_seq_filters_legacy_seq_none_events():
    """When after_seq is given, all event.seq=None entries are filtered out (no leak)."""
    entries = [
        ("1700000000-0", _evt_json(seq=None, message="legacy-1")),
        ("1700000001-0", _evt_json(seq=1, message="m-1")),
        ("1700000002-0", _evt_json(seq=None, message="legacy-2")),
        ("1700000003-0", _evt_json(seq=2, message="m-2")),
    ]
    mock_queue = _mock_queue_with_entries(entries)

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        recovery = RedisEventRecovery(max_count=100)
        result = await recovery.get_recent_events(
            task_id="task-123", after_event_id=None, after_seq=0
        )

    seqs = sorted(e.seq for e in result.events if e.seq is not None)
    assert seqs == [1, 2]
    msgs = [getattr(e, "message", "") for e in result.events]
    assert all(not m.startswith("legacy-") for m in msgs)
