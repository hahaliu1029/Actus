from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.domain.models.event import MessageEvent
from app.infrastructure.external.event_recovery.redis_event_recovery import (
    RedisEventRecovery,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


async def test_get_recent_events_returns_events_with_backfilled_id():
    """Redis 返回事件时，event.id 应回填为 Redis message ID"""
    recovery = RedisEventRecovery()
    event_json = MessageEvent(
        role="assistant", message="hello"
    ).model_dump_json()

    mock_queue = AsyncMock()

    async def mock_get_range(start_id="-", end_id="+", count=100):
        yield ("1712345678901-0", event_json)
        yield ("1712345678901-1", event_json)

    mock_queue.get_range = mock_get_range

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        result = await recovery.get_recent_events("task-123", None)

    assert len(result.events) == 2
    assert result.events[0].id == "1712345678901-0"
    assert result.events[1].id == "1712345678901-1"
    assert result.has_more is False


async def test_get_recent_events_skips_matching_start_id():
    """xrange inclusive 语义：跳过 ID == after_event_id 的首条消息"""
    recovery = RedisEventRecovery()
    event_json = MessageEvent(
        role="assistant", message="hello"
    ).model_dump_json()

    mock_queue = AsyncMock()

    async def mock_get_range(start_id="-", end_id="+", count=100):
        yield ("1712345678901-0", event_json)  # should be skipped
        yield ("1712345678901-1", event_json)

    mock_queue.get_range = mock_get_range

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        result = await recovery.get_recent_events(
            "task-123", "1712345678901-0"
        )

    assert len(result.events) == 1
    assert result.events[0].id == "1712345678901-1"


async def test_get_recent_events_empty_stream():
    """stream 不存在或为空时返回空列表"""
    recovery = RedisEventRecovery()
    mock_queue = AsyncMock()

    async def mock_get_range(start_id="-", end_id="+", count=100):
        return
        yield  # make it an async generator

    mock_queue.get_range = mock_get_range

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        result = await recovery.get_recent_events("task-123", "some-id")

    assert len(result.events) == 0
    assert result.has_more is False


async def test_get_recent_events_has_more_when_count_reached():
    """达到 count 上限时 has_more=True"""
    recovery = RedisEventRecovery(max_count=2)
    event_json = MessageEvent(
        role="assistant", message="hello"
    ).model_dump_json()

    mock_queue = AsyncMock()

    async def mock_get_range(start_id="-", end_id="+", count=100):
        yield ("1-0", event_json)
        yield ("1-1", event_json)

    mock_queue.get_range = mock_get_range

    with patch(
        "app.infrastructure.external.event_recovery.redis_event_recovery.RedisStreamMessageQueue",
        return_value=mock_queue,
    ):
        result = await recovery.get_recent_events("task-123", None)

    assert len(result.events) == 2
    assert result.has_more is True
