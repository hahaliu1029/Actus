from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.external.event_recovery import EventRecoveryResult
from app.domain.models.event import MessageEvent, HealthEvent, HealthStatus
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_event(role="assistant", message="hello", event_id=None):
    e = MessageEvent(role=role, message=message)
    if event_id:
        e.id = event_id
    return e


def _make_session(events=None, task_id=None, status=SessionStatus.RUNNING):
    return Session(
        id="session-1",
        user_id="user-1",
        title="test",
        status=status,
        events=events or [],
        task_id=task_id,
    )


async def test_get_events_since_pg_only_no_task():
    """无 task_id 时仅返回 PG 事件"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")
    e3 = _make_event(event_id="evt-3")
    session = _make_session(events=[e1, e2, e3], task_id=None)

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = None

    result = await svc.get_events_since("session-1", "evt-1", "user-1")

    assert len(result["events"]) == 2
    assert result["events"][0].id == "evt-2"
    assert result["events"][1].id == "evt-3"
    assert result["session_status"] == SessionStatus.RUNNING
    assert result["has_more"] is False


async def test_get_events_since_pg_not_found_returns_all():
    """PG 中找不到 event_id 时返回全量"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")
    session = _make_session(events=[e1, e2], task_id=None)

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = None

    result = await svc.get_events_since("session-1", "nonexistent", "user-1")

    assert len(result["events"]) == 2


async def test_get_events_since_no_since_returns_all_pg_events():
    """since_event_id=None 时返回全量 PG 事件（关标签页重开场景）"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")
    session = _make_session(events=[e1, e2], task_id=None)

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = None

    result = await svc.get_events_since("session-1", None, "user-1")

    assert len(result["events"]) == 2
    assert result["events"][0].id == "evt-1"
    assert result["events"][1].id == "evt-2"


async def test_get_events_since_no_since_with_redis_supplements():
    """since_event_id=None + task_id 存在 → 全量 PG + Redis 补充"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    session = _make_session(events=[e1], task_id="task-abc")

    redis_event = _make_event(event_id="redis-2", message="from redis")

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(
        return_value=EventRecoveryResult(events=[redis_event], has_more=False)
    )

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    result = await svc.get_events_since("session-1", None, "user-1")

    assert len(result["events"]) == 2
    assert result["events"][0].id == "evt-1"
    assert result["events"][1].id == "redis-2"
    event_recovery.get_recent_events.assert_called_once_with(
        task_id="task-abc",
        after_event_id=None,
        after_seq=None,  # B3-core PR-1 §3.3 — additive kwarg, default None on legacy paths
    )


async def test_get_events_since_redis_supplements_pg():
    """Redis 补充 PG 中尚未持久化的事件"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="evt-2")
    session = _make_session(events=[e1, e2], task_id="task-abc")

    redis_event = _make_event(event_id="redis-3", message="from redis")

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(
        return_value=EventRecoveryResult(events=[redis_event], has_more=False)
    )

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    result = await svc.get_events_since("session-1", "evt-1", "user-1")

    # PG: evt-2, Redis: redis-3
    assert len(result["events"]) == 2
    assert result["events"][0].id == "evt-2"
    assert result["events"][1].id == "redis-3"
    event_recovery.get_recent_events.assert_called_once_with(
        task_id="task-abc",
        after_event_id=None,
        after_seq=None,  # B3-core PR-1 §3.3
    )


async def test_get_events_since_prefers_last_valid_redis_stream_id():
    """PG 增量中若存在合法 Redis stream id，应优先用它作为 recovery 起点。"""
    from app.application.services.agent_service import AgentService

    e1 = _make_event(event_id="evt-1")
    e2 = _make_event(event_id="1713264000000-0", message="persisted via redis")
    session = _make_session(events=[e1, e2], task_id="task-abc")

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(
        return_value=EventRecoveryResult(events=[], has_more=False)
    )

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    await svc.get_events_since("session-1", "evt-1", "user-1")

    event_recovery.get_recent_events.assert_called_once_with(
        task_id="task-abc",
        after_event_id="1713264000000-0",
        after_seq=None,  # B3-core PR-1 §3.3
    )


async def test_get_events_since_seq_does_not_let_pg_tail_skip_redis_gap():
    """since_seq must recover Redis-only gaps before a later PG-persisted tail."""
    from app.application.services.agent_service import AgentService

    cursor = _make_event(event_id="1000-2", message="cursor")
    cursor.seq = 2
    pg_tail = _make_event(event_id="1000-6", message="pg-tail")
    pg_tail.seq = 6
    session = _make_session(events=[cursor, pg_tail], task_id="task-abc")

    redis_gap = []
    for seq in range(3, 6):
        event = _make_event(event_id=f"1000-{seq}", message=f"gap-{seq}")
        event.seq = seq
        redis_gap.append(event)
    redis_duplicate_tail = _make_event(event_id="1000-6", message="pg-tail")
    redis_duplicate_tail.seq = 6
    redis_events = redis_gap + [redis_duplicate_tail]

    def _stream_id_gt(left: str, right: str | None) -> bool:
        if right is None:
            return True
        left_ms, left_seq = (int(part) for part in left.split("-", 1))
        right_ms, right_seq = (int(part) for part in right.split("-", 1))
        return (left_ms, left_seq) > (right_ms, right_seq)

    async def recover(task_id: str, after_event_id: str | None, after_seq: int | None):
        return EventRecoveryResult(
            events=[
                event
                for event in redis_events
                if _stream_id_gt(event.id, after_event_id)
                and after_seq is not None
                and event.seq is not None
                and event.seq > after_seq
            ],
            has_more=False,
        )

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(side_effect=recover)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    result = await svc.get_events_since(
        "session-1",
        "1000-2",
        "user-1",
        since_seq=2,
    )

    assert [event.id for event in result["events"]] == [
        "1000-3",
        "1000-4",
        "1000-5",
        "1000-6",
    ]
    event_recovery.get_recent_events.assert_called_once_with(
        task_id="task-abc",
        after_event_id="1000-2",
        after_seq=2,
    )


async def test_get_events_since_seq_sorts_non_stream_pg_event_by_seq():
    """Recovered sequenced events are ordered by seq even if PG ids are UUIDs."""
    from app.application.services.agent_service import AgentService

    cursor = _make_event(event_id="1000-2", message="cursor")
    cursor.seq = 2
    pg_middle = _make_event(event_id="uuid-pg-4", message="pg-middle")
    pg_middle.seq = 4
    session = _make_session(events=[cursor, pg_middle], task_id="task-abc")

    redis_before = _make_event(event_id="1000-3", message="redis-before")
    redis_before.seq = 3
    redis_after = _make_event(event_id="1000-5", message="redis-after")
    redis_after.seq = 5

    async def recover(task_id: str, after_event_id: str | None, after_seq: int | None):
        return EventRecoveryResult(
            events=[redis_before, redis_after],
            has_more=False,
        )

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(side_effect=recover)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    result = await svc.get_events_since(
        "session-1",
        "1000-2",
        "user-1",
        since_seq=2,
    )

    assert [(event.id, event.seq) for event in result["events"]] == [
        ("1000-3", 3),
        ("uuid-pg-4", 4),
        ("1000-5", 5),
    ]


async def test_get_events_since_invalid_since_id_does_not_reach_redis():
    """显式 since_event_id 为无效值时，不能直接拿去做 Redis stream 游标。"""
    from app.application.services.agent_service import AgentService

    session = _make_session(events=[], task_id="task-abc")

    uow_mock = AsyncMock()
    uow_mock.session.get_by_id = AsyncMock(return_value=session)
    uow_factory = AsyncMock(return_value=uow_mock)
    uow_factory.__aenter__ = AsyncMock(return_value=uow_mock)
    uow_factory.__aexit__ = AsyncMock(return_value=False)

    event_recovery = AsyncMock()
    event_recovery.get_recent_events = AsyncMock(
        return_value=EventRecoveryResult(events=[], has_more=False)
    )

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = lambda: uow_factory
    svc._event_recovery = event_recovery

    await svc.get_events_since("session-1", "1", "user-1")

    event_recovery.get_recent_events.assert_called_once_with(
        task_id="task-abc",
        after_event_id=None,
        after_seq=None,  # B3-core PR-1 §3.3
    )
