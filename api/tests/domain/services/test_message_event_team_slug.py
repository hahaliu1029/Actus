from app.domain.models.event import MessageEvent


def test_message_event_team_slug_defaults_none():
    # MessageEvent inherits all required fields from BaseEvent with defaults;
    # `message` defaults to "" so a no-arg ctor is valid, but we set it for clarity.
    ev = MessageEvent(message="hi")
    assert getattr(ev, "team_slug", "MISSING") is None


def test_message_event_team_slug_roundtrips_through_dump():
    ev = MessageEvent(message="hi", team_slug="squad")
    dumped = ev.model_dump()
    assert dumped["team_slug"] == "squad"


def test_message_event_team_slug_survives_json_union_roundtrip():
    # The production queue seam writes MessageEvent.model_dump_json()
    # (agent_service.py) and reads via TypeAdapter(Event).validate_json(...)
    # (agent_task_runner.py:961). The dict-dump test above does NOT exercise the
    # discriminated-union JSON carrier — this one does, end to end.
    from pydantic import TypeAdapter

    from app.domain.models.event import Event

    ev = MessageEvent(message="hi", team_slug="squad")
    restored = TypeAdapter(Event).validate_json(ev.model_dump_json())
    assert isinstance(restored, MessageEvent)
    assert restored.team_slug == "squad"
