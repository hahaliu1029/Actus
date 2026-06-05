"""A4-0: domain SessionModeChangedEvent — defaults, type literal, union membership,
and the import-cycle / replay round-trip (T-MODEL domain part + T-IMPORT)."""
from pydantic import TypeAdapter


def test_session_mode_changed_event_defaults_and_type_literal() -> None:
    from app.domain.models.event import SessionModeChangedEvent

    event = SessionModeChangedEvent(to="waiting", reason="wait", mode_revision=5)
    assert event.type == "session_mode_changed"
    assert event.to == "waiting"
    assert event.reason == "wait"
    assert event.mode_revision == 5
    # best-effort context fields default to None
    assert event.from_mode is None

    full = SessionModeChangedEvent(
        to="takeover", from_mode="running", reason="takeover_started", mode_revision=9
    )
    assert full.from_mode == "running"


def test_session_mode_changed_event_json_round_trip() -> None:
    from app.domain.models.event import SessionModeChangedEvent

    event = SessionModeChangedEvent(to="takeover_pending", reason="takeover_requested")
    restored = SessionModeChangedEvent.model_validate_json(event.model_dump_json())
    assert restored.to == "takeover_pending"
    assert restored.mode_revision is None


def test_no_import_cycle_between_event_and_session() -> None:
    # R2#1: event.py must NOT import session.py (session.py imports event.py).
    import importlib

    importlib.import_module("app.domain.models.event")
    importlib.import_module("app.domain.models.session")


def test_session_mode_changed_round_trips_through_event_union() -> None:
    # R1#2: replay rehydrates via TypeAdapter(Event); the new event MUST be a
    # discriminated-union member or this raises on the discriminator.
    from app.domain.models.event import Event, SessionModeChangedEvent

    event = SessionModeChangedEvent(to="waiting", reason="wait", mode_revision=3)
    parsed = TypeAdapter(Event).validate_json(event.model_dump_json())
    assert isinstance(parsed, SessionModeChangedEvent)
    assert parsed.to == "waiting"
    assert parsed.mode_revision == 3
