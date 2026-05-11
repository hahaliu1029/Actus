"""B3-core PR-1: AgentService.get_events_since since_seq cursor + last_seq.

Spec v3 §3.3 + §6.7.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.session import SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeHotRedis:
    def __init__(self, hot_hash):
        self.hot_hash = hot_hash
        self.keys = []

    async def hgetall(self, key: str):
        self.keys.append(key)
        return self.hot_hash


class _FailingHotRedis:
    async def hgetall(self, key: str):
        raise RuntimeError("redis unavailable")


class _RedisWrapper:
    def __init__(self, redis):
        self.client = redis


def _build_svc_with_session(
    events_with_seq,
    *,
    redis_client=None,
    terminal_reason=None,
):
    """Construct a minimal AgentService whose _get_accessible_session returns
    a session with the given events (each event having .id, .seq, .task_id)."""
    from app.application.services.agent_service import AgentService

    expires_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    svc = AgentService.__new__(AgentService)

    session = MagicMock()
    session.events = events_with_seq
    session.task_id = None  # no Redis path
    session.status = SessionStatus.RUNNING
    session.user_id = "u-1"
    session.id = "sess-1"
    session.execution_mode = "background"
    session.execution_phase = "running"
    session.background_reason = "explicit"
    session.expires_at = expires_at
    session.retry_budget_remaining = 2
    session.suspended_reason = None
    session.terminal_reason = terminal_reason

    svc._get_accessible_session = AsyncMock(return_value=session)
    svc._event_recovery = None
    svc._redis_client = redis_client
    svc._is_valid_redis_stream_id = AgentService._is_valid_redis_stream_id  # static
    return svc


def _assert_base_snapshot_fields(snapshot):
    assert set(snapshot.model_dump()) == {
        "execution_mode",
        "execution_phase",
        "background_reason",
        "expires_at",
        "retry_budget_remaining",
        "suspended_reason",
        "terminal_reason",
        "last_progress_at",
        "is_alive",
        "cancellation_state",
    }
    assert snapshot.execution_mode == "background"
    assert snapshot.execution_phase == "running"
    assert snapshot.background_reason == "explicit"
    assert snapshot.expires_at == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert snapshot.retry_budget_remaining == 2
    assert snapshot.suspended_reason is None


async def test_since_seq_logs_when_event_id_fallback_is_also_present(caplog):
    """Both given → since_seq is preferred for sequenced events, warning logged."""
    svc = _build_svc_with_session([])

    with caplog.at_level(logging.WARNING):
        result = await svc.get_events_since(
            session_id="sess-1",
            since_event_id="legacy-id-1",
            user_id="u-1",
            since_seq=10,
        )
    assert result["events"] == []
    assert result["last_seq"] == 10  # falls back to since_seq when no events
    assert any("using since_seq" in rec.message for rec in caplog.records)


async def test_since_seq_with_event_id_keeps_legacy_events_after_cursor():
    """Mixed streams keep seq=None legacy events after the event-id cursor."""
    cursor = MagicMock()
    cursor.id = "cursor-id"
    cursor.seq = 1
    legacy_after_cursor = MagicMock()
    legacy_after_cursor.id = "legacy-after-cursor"
    legacy_after_cursor.seq = None
    seq_after_cursor = MagicMock()
    seq_after_cursor.id = "seq-after-cursor"
    seq_after_cursor.seq = 2
    svc = _build_svc_with_session([cursor, legacy_after_cursor, seq_after_cursor])

    result = await svc.get_events_since(
        session_id="sess-1",
        since_event_id="cursor-id",
        user_id="u-1",
        since_seq=1,
    )

    assert result["events"] == [legacy_after_cursor, seq_after_cursor]
    assert result["last_seq"] == 2


async def test_last_seq_derives_from_max_event_seq():
    """last_seq = max(event.seq for events with seq)."""
    e1 = MagicMock()
    e1.id = "a"
    e1.seq = 3
    e2 = MagicMock()
    e2.id = "b"
    e2.seq = 7
    e3 = MagicMock()
    e3.id = "c"
    e3.seq = None  # legacy event
    svc = _build_svc_with_session([e1, e2, e3])

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )
    assert result["last_seq"] == 7
    assert result["supervisor_snapshot"] is not None


async def test_last_seq_falls_back_to_since_seq_when_no_events_have_seq():
    """If no event has seq, last_seq = since_seq (or 0)."""
    e1 = MagicMock()
    e1.id = "x"
    e1.seq = None
    svc = _build_svc_with_session([e1])

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1", since_seq=42,
    )
    assert result["last_seq"] == 42


async def test_last_seq_zero_when_no_events_and_no_since_seq():
    """No events + no since_seq → last_seq = 0 (default)."""
    svc = _build_svc_with_session([])

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )
    assert result["last_seq"] == 0


async def test_supervisor_snapshot_uses_session_fields_and_hot_activity():
    last_activity_ts = int(
        (datetime.now(timezone.utc) - timedelta(seconds=5)).timestamp()
    )
    hot_redis = _FakeHotRedis(
        {
            "last_activity_at": str(last_activity_ts),
            "cancellation_pending": "0",
        }
    )
    svc = _build_svc_with_session([], redis_client=_RedisWrapper(hot_redis))

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )
    snapshot = result["supervisor_snapshot"]
    _assert_base_snapshot_fields(snapshot)
    assert snapshot.terminal_reason is None
    assert snapshot.last_progress_at == datetime.fromtimestamp(
        last_activity_ts, timezone.utc
    )
    assert snapshot.is_alive is True
    assert snapshot.cancellation_state == "none"
    assert hot_redis.keys == ["supervisor:hot:sess-1"]


async def test_supervisor_snapshot_future_hot_activity_is_not_alive():
    future_activity_ts = int(
        (datetime.now(timezone.utc) + timedelta(seconds=30)).timestamp()
    )
    hot_redis = _FakeHotRedis({"last_activity_at": str(future_activity_ts)})
    svc = _build_svc_with_session([], redis_client=_RedisWrapper(hot_redis))

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )

    snapshot = result["supervisor_snapshot"]
    assert snapshot.last_progress_at == datetime.fromtimestamp(
        future_activity_ts, timezone.utc
    )
    assert snapshot.is_alive is False


async def test_supervisor_snapshot_reports_cancelling_from_hot_hash():
    hot_redis = _FakeHotRedis({"cancellation_pending": "1"})
    svc = _build_svc_with_session([], redis_client=hot_redis)

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )

    snapshot = result["supervisor_snapshot"]
    assert snapshot.cancellation_state == "cancelling"


async def test_supervisor_snapshot_prefers_terminal_user_cancel():
    hot_redis = _FakeHotRedis({"cancellation_pending": "1"})
    svc = _build_svc_with_session(
        [],
        redis_client=hot_redis,
        terminal_reason="user_cancel",
    )

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )

    snapshot = result["supervisor_snapshot"]
    assert snapshot.terminal_reason == "user_cancel"
    assert snapshot.cancellation_state == "cancelled"


async def test_supervisor_snapshot_survives_missing_redis_client():
    svc = _build_svc_with_session([], redis_client=None)

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )

    snapshot = result["supervisor_snapshot"]
    _assert_base_snapshot_fields(snapshot)
    assert snapshot.last_progress_at is None
    assert snapshot.is_alive is False
    assert snapshot.cancellation_state == "none"


async def test_supervisor_snapshot_survives_redis_read_failure():
    svc = _build_svc_with_session([], redis_client=_FailingHotRedis())

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )

    snapshot = result["supervisor_snapshot"]
    _assert_base_snapshot_fields(snapshot)
    assert snapshot.last_progress_at is None
    assert snapshot.is_alive is False
    assert snapshot.cancellation_state == "none"
