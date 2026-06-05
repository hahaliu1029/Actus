from datetime import datetime

import pytest
from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
)
from app.interfaces.schemas.event import ControlSSEEvent, EventMapper


def test_event_mapper_maps_control_event_to_control_sse_event() -> None:
    expires_at = datetime(2026, 2, 27, 12, 0, 0)
    event = ControlEvent(
        action=ControlAction.REQUESTED,
        scope=ControlScope.SHELL,
        source=ControlSource.AGENT,
        request_status="starting",
        takeover_id="tk_123",
        expires_at=expires_at,
    )

    EventMapper._cache_mapping = None
    sse_event = EventMapper.event_to_sse_event(event)

    assert isinstance(sse_event, ControlSSEEvent)
    assert sse_event.event == "control"
    assert sse_event.data.action == ControlAction.REQUESTED
    assert sse_event.data.scope == ControlScope.SHELL
    assert sse_event.data.source == ControlSource.AGENT
    assert sse_event.data.request_status == "starting"
    assert sse_event.data.takeover_id == "tk_123"
    assert sse_event.data.expires_at == int(expires_at.timestamp())


def test_control_event_requested_requires_scope() -> None:
    with pytest.raises(ValueError):
        ControlEvent(
            action=ControlAction.REQUESTED,
            source=ControlSource.AGENT,
        )


def test_event_mapper_maps_session_mode_changed_to_typed_sse_event() -> None:
    from app.domain.models.event import SessionModeChangedEvent
    from app.interfaces.schemas.event import (
        EventMapper,
        SessionModeChangedSSEEvent,
    )

    event = SessionModeChangedEvent(
        to="takeover",
        from_mode="running",
        reason="takeover_started",
        mode_revision=12,
    )

    EventMapper._cache_mapping = None
    sse_event = EventMapper.event_to_sse_event(event)

    assert isinstance(sse_event, SessionModeChangedSSEEvent)
    assert sse_event.event == "session_mode_changed"
    assert sse_event.data.to == "takeover"
    assert sse_event.data.from_mode == "running"
    assert sse_event.data.reason == "takeover_started"
    assert sse_event.data.mode_revision == 12


def test_session_mode_changed_sse_optional_fields_default_none() -> None:
    from app.domain.models.event import SessionModeChangedEvent
    from app.interfaces.schemas.event import EventMapper, SessionModeChangedSSEEvent

    event = SessionModeChangedEvent(to="waiting", reason="wait")
    EventMapper._cache_mapping = None
    sse_event = EventMapper.event_to_sse_event(event)
    assert isinstance(sse_event, SessionModeChangedSSEEvent)
    assert sse_event.data.from_mode is None
    assert sse_event.data.mode_revision is None
