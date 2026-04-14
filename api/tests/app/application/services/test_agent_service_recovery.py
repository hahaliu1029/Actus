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
        after_event_id="evt-1",
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
