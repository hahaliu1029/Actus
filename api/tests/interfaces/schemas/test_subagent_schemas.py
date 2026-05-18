"""Schemas: probe events extend BaseEvent with `type` field (not event_type)."""
import pytest
from pydantic import ValidationError

from app.interfaces.schemas.subagent import (
    ResearchSubagentRequest,
    ChildStartedEvent,
    ChildDoneEvent,
    JoinedSummaryEvent,
    ChildOutcome,
)


def test_request_field_constraints():
    """prompts: list[str] with min_length=1 max_length=3."""
    req = ResearchSubagentRequest(prompts=["a"], max_children=3)
    assert req.prompts == ["a"]
    assert req.max_children == 3

    with pytest.raises(ValidationError):
        ResearchSubagentRequest(prompts=[], max_children=3)

    with pytest.raises(ValidationError):
        ResearchSubagentRequest(prompts=["a", "b", "c", "d"], max_children=3)

    with pytest.raises(ValidationError):
        ResearchSubagentRequest(prompts=["a"], max_children=10)


def test_child_started_event_has_type_field():
    """Probe events extend BaseEvent with `type` (matches EventMapper)."""
    ev = ChildStartedEvent(
        id="ev-1",
        probe_run_id="p-1",
        child_session_id="c-1",
        prompt="hello",
    )
    assert ev.type == "child_started"
    assert ev.id == "ev-1"
    d = ev.model_dump()
    assert d["type"] == "child_started"


def test_child_done_event_outcome_enum():
    """ChildOutcome is one of completed/failed/timed_out/waiting/cancelled."""
    ev = ChildDoneEvent(
        id="ev-2",
        probe_run_id="p-1",
        child_session_id="c-1",
        outcome=ChildOutcome.COMPLETED,
        final_answer="result",
        transcript_tokens=1500,
        error_summary=None,
    )
    assert ev.outcome == ChildOutcome.COMPLETED

    with pytest.raises(ValidationError):
        ChildDoneEvent(
            id="ev-3",
            probe_run_id="p-1",
            child_session_id="c-1",
            outcome="invalid",
            final_answer=None,
            transcript_tokens=0,
            error_summary="x",
        )


def test_joined_summary_event_validation_warnings_default_empty():
    ev = JoinedSummaryEvent(
        id="ev-4",
        probe_run_id="p-1",
        summary="result",
        summary_tokens=200,
        completed_children=["c-1", "c-2"],
        dropped_children=[],
        metrics={"compression_ratio": 0.08},
    )
    assert ev.validation_warnings == []
