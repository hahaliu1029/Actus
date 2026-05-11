from __future__ import annotations

import logging
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services import memory_notification_emitter as emitter_module
from app.application.services.memory_notification_emitter import (
    DBMemoryNotificationEmitter,
)

pytestmark = pytest.mark.anyio

B3_EVENT_TYPES = frozenset(
    {
        "bg_completed",
        "bg_cancelled",
        "bg_failed_resume",
        "bg_failed_watchdog",
        "bg_terminal_server_restart",
        "bg_suspended_timeout",
        "bg_suspended_server_restart",
        "bg_retry_exhausted",
    }
)

M1_EVENT_TYPES = frozenset(
    {
        "memory_gate_paused",
        "quota_exceeded",
        "fs_permanent_failure",
    }
)


def _make_emitter():
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    session_factory = MagicMock(return_value=mock_session)
    repo = AsyncMock()
    repo_factory = MagicMock(return_value=repo)
    emitter = DBMemoryNotificationEmitter(
        session_factory=session_factory,
        repo_factory=repo_factory,
    )
    return emitter, session_factory, mock_session, repo_factory, repo


def test_b3_event_type_literal_matches_spec() -> None:
    assert frozenset(
        get_args(emitter_module.B3CoreNotificationEventType)
    ) == B3_EVENT_TYPES
    assert emitter_module._B3_CORE_EVENT_TYPES == B3_EVENT_TYPES
    assert len(emitter_module._B3_CORE_EVENT_TYPES) == 8


def test_m1_event_types_remain_valid() -> None:
    assert emitter_module._M1_EVENT_TYPES == M1_EVENT_TYPES
    assert emitter_module.ALL_VALID_EVENT_TYPES == M1_EVENT_TYPES | B3_EVENT_TYPES
    assert len(emitter_module.ALL_VALID_EVENT_TYPES) == 11


@pytest.mark.parametrize("event_type", sorted(M1_EVENT_TYPES | B3_EVENT_TYPES))
async def test_known_event_types_are_persisted(event_type: str) -> None:
    emitter, session_factory, mock_session, repo_factory, repo = _make_emitter()

    await emitter.emit(
        user_id="u-1",
        event_type=event_type,
        payload={"session_id": "s-1"},
    )

    session_factory.assert_called_once_with()
    repo_factory.assert_called_once_with(mock_session)
    repo.create.assert_awaited_once()
    assert repo.create.await_args.kwargs["event_type"] == event_type
    mock_session.commit.assert_awaited_once()


async def test_unknown_event_type_is_dropped_and_logged(caplog) -> None:
    caplog.set_level(logging.WARNING, logger=emitter_module.__name__)
    emitter, session_factory, mock_session, repo_factory, repo = _make_emitter()

    await emitter.emit(
        user_id="u-1",
        event_type="not_a_real_event",
        payload={"session_id": "s-1"},
    )

    session_factory.assert_not_called()
    repo_factory.assert_not_called()
    repo.create.assert_not_called()
    mock_session.commit.assert_not_called()
    assert "unknown event_type=not_a_real_event" in caplog.text
