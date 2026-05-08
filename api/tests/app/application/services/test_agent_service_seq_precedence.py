"""B3-core PR-1: AgentService.get_events_since since_seq cursor + last_seq.

Spec v3 §3.3 + §6.7.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _build_svc_with_session(events_with_seq):
    """Construct a minimal AgentService whose _get_accessible_session returns
    a session with the given events (each event having .id, .seq, .task_id)."""
    from app.application.services.agent_service import AgentService

    svc = AgentService.__new__(AgentService)

    session = MagicMock()
    session.events = events_with_seq
    session.task_id = None  # no Redis path
    session.status = MagicMock()
    session.user_id = "u-1"
    session.id = "sess-1"

    svc._get_accessible_session = AsyncMock(return_value=session)
    svc._event_recovery = None
    svc._is_valid_redis_stream_id = AgentService._is_valid_redis_stream_id  # static
    return svc


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
    assert result["supervisor_snapshot"] is None  # PR-1 placeholder


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


async def test_supervisor_snapshot_is_none_in_pr1():
    """PR-1 placeholder: supervisor_snapshot is always None.
    PR-3c/PR-4 will populate this with real snapshot data.
    """
    svc = _build_svc_with_session([])

    result = await svc.get_events_since(
        session_id="sess-1", since_event_id=None, user_id="u-1",
    )
    assert result["supervisor_snapshot"] is None
