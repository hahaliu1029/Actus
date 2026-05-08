"""B3-core PR-1: AgentService._emit_event + get_session helpers.

Spec v3 §3.2 (session:seq counter), §6.5/§6.6 (helpers), §6.7 (seq plumbing).
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_redis_client():
    """Lightweight redis_client double exposing client.incr / expire / exists."""
    rc = MagicMock()
    rc.client = MagicMock()
    rc.client.incr = AsyncMock(return_value=1)
    rc.client.expire = AsyncMock()
    rc.client.exists = AsyncMock(return_value=0)
    return rc


def _build_svc_with_no_task(redis_client):
    """Build AgentService whose uow.session.get_by_id returns None (no task)."""
    from app.application.services.agent_service import AgentService

    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock()
    uow.session = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=None)
    uow_factory = MagicMock(return_value=uow)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = uow_factory
    svc._redis_client = redis_client
    svc._task_cls = MagicMock()
    return svc, uow


async def test_get_session_returns_none_for_missing(mock_redis_client):
    """get_session returns None (NOT raises) when session doesn't exist."""
    svc, uow = _build_svc_with_no_task(mock_redis_client)

    result = await svc.get_session("nonexistent")
    assert result is None
    uow.session.get_by_id.assert_awaited_once_with("nonexistent")


async def test_get_session_returns_session_when_present(mock_redis_client):
    """get_session forwards the session object from uow.session.get_by_id."""
    from app.application.services.agent_service import AgentService

    fake_session = MagicMock(id="sess-1", task_id="task-1")
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock()
    uow.session = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=fake_session)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = MagicMock(return_value=uow)
    svc._redis_client = mock_redis_client
    svc._task_cls = MagicMock()

    result = await svc.get_session("sess-1")
    assert result is fake_session


async def test_emit_event_stamps_seq_via_incr(mock_redis_client):
    """First emit calls INCR session:seq:{sid}, returns 1 → event.seq=1, EXPIRE armed."""
    from app.domain.models.event import MessageEvent

    sid = str(uuid.uuid4())
    svc, _uow = _build_svc_with_no_task(mock_redis_client)  # session None → backlog skip

    evt = MessageEvent(role="assistant", message="hi")
    await svc._emit_event(sid, evt)

    mock_redis_client.client.incr.assert_awaited_once_with(f"session:seq:{sid}")
    # First INCR returns 1 → 24h EXPIRE armed.
    mock_redis_client.client.expire.assert_awaited_once_with(
        f"session:seq:{sid}", 86400
    )
    assert evt.seq == 1


async def test_emit_event_subsequent_seq_does_not_re_arm_expire(mock_redis_client):
    """When INCR returns N>1 (subsequent emit), EXPIRE is NOT called again."""
    from app.domain.models.event import MessageEvent

    mock_redis_client.client.incr = AsyncMock(return_value=5)  # not first emit

    svc, _uow = _build_svc_with_no_task(mock_redis_client)
    evt = MessageEvent(role="assistant", message="continue")
    await svc._emit_event("sess-1", evt)

    mock_redis_client.client.incr.assert_awaited_once()
    mock_redis_client.client.expire.assert_not_awaited()
    assert evt.seq == 5


async def test_emit_event_seq_stamp_failure_does_not_crash(mock_redis_client):
    """If INCR fails, _emit_event logs + continues (event.seq stays None)."""
    from app.domain.models.event import MessageEvent

    mock_redis_client.client.incr = AsyncMock(side_effect=RuntimeError("redis down"))

    svc, _uow = _build_svc_with_no_task(mock_redis_client)
    evt = MessageEvent(role="assistant", message="ok")
    await svc._emit_event("sess-1", evt)

    assert evt.seq is None  # graceful degrade


async def test_emit_event_no_redis_client_skips_emission():
    """When redis_client is None (test fallback), emit returns None and does not crash."""
    from app.application.services.agent_service import AgentService
    from app.domain.models.event import MessageEvent

    svc = AgentService.__new__(AgentService)
    svc._redis_client = None
    svc._uow_factory = MagicMock()
    svc._task_cls = MagicMock()

    evt = MessageEvent(role="assistant", message="x")
    result = await svc._emit_event("sess-X", evt)
    assert result is None
    assert evt.seq is None


async def test_emit_event_emits_via_task_output_stream_when_task_present(
    mock_redis_client,
):
    """When session.task_id resolves to a live task, _emit_event uses task.output_stream.put."""
    from app.application.services.agent_service import AgentService
    from app.domain.models.event import MessageEvent

    fake_session = MagicMock(id="sess-X", task_id="task-X", user_id="u")
    uow = MagicMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock()
    uow.session = MagicMock()
    uow.session.get_by_id = AsyncMock(return_value=fake_session)

    fake_task = MagicMock()
    fake_task.output_stream = MagicMock()
    fake_task.output_stream.put = AsyncMock(return_value="9999-0")

    task_cls = MagicMock()
    task_cls.get = MagicMock(return_value=fake_task)

    svc = AgentService.__new__(AgentService)
    svc._uow_factory = MagicMock(return_value=uow)
    svc._redis_client = mock_redis_client
    svc._task_cls = task_cls

    evt = MessageEvent(role="assistant", message="via-task")
    result = await svc._emit_event("sess-X", evt)

    assert result == "9999-0"
    fake_task.output_stream.put.assert_awaited_once()
    payload = fake_task.output_stream.put.call_args.args[0]
    assert '"seq":1' in payload
