"""B3-core PR-1: new domain event payload tests.

Covers spec v3 §3.3 — BaseEvent.seq field + ExecutionStateChangedEvent +
OwnerConflictEvent. PR-0 anchor flipped: C-Wire-1.
"""

from __future__ import annotations

import json


def test_base_event_seq_field_default_none():
    """BaseEvent.seq is Optional[int] = None — backward-compatible default."""
    from app.domain.models.event import MessageEvent

    evt = MessageEvent(role="assistant", message="hi")
    assert evt.seq is None  # default


def test_base_event_seq_field_round_trip_int():
    """seq round-trips through model_dump_json()."""
    from app.domain.models.event import MessageEvent

    evt = MessageEvent(role="assistant", message="hi", seq=42)
    raw = evt.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["seq"] == 42

    rebuilt = MessageEvent.model_validate_json(raw)
    assert rebuilt.seq == 42


def test_base_event_seq_field_legacy_json_parses_with_none():
    """JSON written before PR-1 (no `seq` key) still parses — default-None protects PG/Redis backlog."""
    from app.domain.models.event import MessageEvent

    legacy_json = '{"id":"e1","type":"message","created_at":"2026-05-07T00:00:00","role":"assistant","message":"hello"}'
    evt = MessageEvent.model_validate_json(legacy_json)
    assert evt.seq is None


def test_execution_state_changed_event_uses_type_discriminator():
    """C-Wire-1: discriminator field is `type`, value `execution_state_changed`."""
    from app.domain.models.event import ExecutionStateChangedEvent, ExecutionStatePayload

    evt = ExecutionStateChangedEvent(
        payload=ExecutionStatePayload(
            execution_mode="foreground",
            execution_phase="running",
            retry_budget_remaining=3,
        )
    )
    raw = evt.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "execution_state_changed"
    assert "event_type" not in parsed  # never use event_type discriminator on domain events
    assert parsed["payload"]["execution_mode"] == "foreground"
    assert parsed["payload"]["execution_phase"] == "running"
    assert parsed["payload"]["retry_budget_remaining"] == 3


def test_execution_state_changed_event_round_trip_via_event_union():
    """Discriminated union dispatches `type=execution_state_changed` → ExecutionStateChangedEvent."""
    from pydantic import TypeAdapter

    from app.domain.models.event import (
        Event,
        ExecutionStateChangedEvent,
        ExecutionStatePayload,
    )

    adapter = TypeAdapter(Event)
    evt = ExecutionStateChangedEvent(
        payload=ExecutionStatePayload(
            execution_mode="background",
            execution_phase="suspended",
            background_reason="explicit",
            retry_budget_remaining=2,
            suspended_reason="bg_idle_timeout",
            transition_reason="idle watchdog T8",
        ),
        seq=17,
    )
    raw = evt.model_dump_json()
    rebuilt = adapter.validate_json(raw)

    assert isinstance(rebuilt, ExecutionStateChangedEvent)
    assert rebuilt.seq == 17
    assert rebuilt.payload.background_reason == "explicit"
    assert rebuilt.payload.suspended_reason == "bg_idle_timeout"


def test_execution_state_changed_event_validates_phase_literal():
    """Invalid execution_phase values are rejected by Pydantic Literal."""
    import pytest as _pytest
    from pydantic import ValidationError

    from app.domain.models.event import ExecutionStateChangedEvent, ExecutionStatePayload

    with _pytest.raises(ValidationError):
        ExecutionStateChangedEvent(
            payload=ExecutionStatePayload(
                execution_mode="foreground",
                execution_phase="not_a_real_phase",  # invalid
                retry_budget_remaining=3,
            )
        )


def test_owner_conflict_event_uses_type_discriminator():
    """spec v3 §3.3 — OwnerConflictEvent uses type=`owner_conflict`."""
    from app.domain.models.event import OwnerConflictEvent, OwnerConflictPayload

    evt = OwnerConflictEvent(
        payload=OwnerConflictPayload(
            current_owner_connection_id="conn-1",
            conflicting_connection_id="conn-2",
            session_id="sess-1",
        )
    )
    raw = evt.model_dump_json()
    parsed = json.loads(raw)
    assert parsed["type"] == "owner_conflict"
    assert parsed["payload"]["current_owner_connection_id"] == "conn-1"
    assert parsed["payload"]["suggested_action"] == "wait_lease_expire"


def test_owner_conflict_event_round_trip_via_event_union():
    """Discriminated union dispatches `type=owner_conflict` → OwnerConflictEvent."""
    from pydantic import TypeAdapter

    from app.domain.models.event import Event, OwnerConflictEvent, OwnerConflictPayload

    adapter = TypeAdapter(Event)
    evt = OwnerConflictEvent(
        payload=OwnerConflictPayload(
            current_owner_connection_id="conn-A",
            conflicting_connection_id="conn-B",
            session_id="sess-X",
            suggested_action="request_takeover",
        ),
        seq=99,
    )
    rebuilt = adapter.validate_json(evt.model_dump_json())
    assert isinstance(rebuilt, OwnerConflictEvent)
    assert rebuilt.payload.suggested_action == "request_takeover"
    assert rebuilt.seq == 99
