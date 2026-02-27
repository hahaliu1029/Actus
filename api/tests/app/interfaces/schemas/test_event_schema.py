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
