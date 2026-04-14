"""Test FinishingEvent model and SSE serialization."""
import pytest
from pydantic import TypeAdapter


def test_finishing_status_exists():
    from app.domain.models.session import SessionStatus
    assert hasattr(SessionStatus, "FINISHING")
    assert SessionStatus.FINISHING.value == "finishing"


def test_finishing_event_type_discriminator():
    from app.domain.models.event import FinishingEvent, Event
    evt = FinishingEvent()
    assert evt.type == "finishing"
    # Must be deserializable via Event union
    adapter = TypeAdapter(Event)
    json_str = evt.model_dump_json()
    restored = adapter.validate_json(json_str)
    assert restored.type == "finishing"


def test_finishing_sse_event_mapper():
    from app.domain.models.event import FinishingEvent
    from app.interfaces.schemas.event import EventMapper
    # Clear cache to pick up new union members
    EventMapper._cache_mapping = None
    evt = FinishingEvent()
    sse = EventMapper.event_to_sse_event(evt)
    assert sse.event == "finishing"
