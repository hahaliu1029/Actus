"""[C2 PR-8 Task 8.2] EventMapper coverage for the five coordinator SSE events.

These tests exercise the reflection-based ``EventMapper`` dispatch path for the
five coordinator events introduced by PR-8 §13. The mapper builds its
``event_type → EventMapping`` cache by inspecting ``AgentSSEEvent``'s union
arguments, so the act of *registering* the new SSE classes is what makes them
flow through ``event_to_sse_event``. The cache is class-level, so each test
resets it (``EventMapper._cache_mapping = None``) before exercising the mapper
to guarantee any registration changes are picked up after import.
"""
from __future__ import annotations

import json

import pytest

from app.domain.models.event import (
    CoordinatorApplyEvent,
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome
from app.domain.models.patch_apply_plan import GroupOutcome
from app.interfaces.schemas.event import (
    CoordinatorApplyEventData,
    CoordinatorApplySSEEvent,
    CoordinatorDispatchEventData,
    CoordinatorDispatchSSEEvent,
    CoordinatorReduceEventData,
    CoordinatorReduceSSEEvent,
    CoordinatorSiblingCancelEventData,
    CoordinatorSiblingCancelSSEEvent,
    CoordinatorWorkerSpawnedEventData,
    CoordinatorWorkerSpawnedSSEEvent,
    EventMapper,
)


@pytest.fixture(autouse=True)
def _reset_event_mapper_cache() -> None:
    """Reset the class-level mapping cache so newly-registered SSE classes
    in ``AgentSSEEvent`` are visible regardless of import order across the
    pytest session.
    """
    EventMapper._cache_mapping = None
    yield
    EventMapper._cache_mapping = None


def _lineage_kwargs() -> dict[str, str]:
    return {
        "root_session_id": "root-1",
        "parent_session_id": "parent-1",
        "child_session_id": "child-1",
        "coordinator_run_id": "r1",
        "work_unit_id": "wu-1",
    }


def test_dispatch_maps_to_sse() -> None:
    event = CoordinatorDispatchEvent(
        **_lineage_kwargs(),
        step_id="step-9",
        work_unit_count=2,
        work_unit_ids=["wu-1", "wu-2"],
        phases=["exploration", "write"],
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorDispatchSSEEvent)
    assert sse.event == "coordinator_dispatch"
    assert isinstance(sse.data, CoordinatorDispatchEventData)
    assert sse.data.coordinator_run_id == "r1"
    assert sse.data.root_session_id == "root-1"
    assert sse.data.parent_session_id == "parent-1"
    assert sse.data.child_session_id == "child-1"
    assert sse.data.work_unit_id == "wu-1"
    assert sse.data.step_id == "step-9"
    assert sse.data.work_unit_count == 2
    assert sse.data.work_unit_ids == ["wu-1", "wu-2"]
    assert sse.data.phases == ["exploration", "write"]
    assert sse.data.event_id == event.id


def test_worker_spawned_maps_to_sse() -> None:
    event = CoordinatorWorkerSpawnedEvent(
        **_lineage_kwargs(),
        objective="refactor module x",
        phase="write",
        allowed_tools=["file_read", "file_write"],
        write_lease_count=1,
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorWorkerSpawnedSSEEvent)
    assert sse.event == "coordinator_worker_spawned"
    assert isinstance(sse.data, CoordinatorWorkerSpawnedEventData)
    assert sse.data.coordinator_run_id == "r1"
    assert sse.data.objective == "refactor module x"
    assert sse.data.phase == "write"
    assert sse.data.allowed_tools == ["file_read", "file_write"]
    assert sse.data.write_lease_count == 1


def test_reduce_maps_to_sse() -> None:
    cost = CostAggregate(
        total_input_tokens=10,
        total_output_tokens=20,
        total_usd=0.5,
        tool_call_count=3,
    )
    event = CoordinatorReduceEvent(
        **_lineage_kwargs(),
        group_outcome=GroupOutcome.SUCCESS,
        per_worker_outcomes={
            "wu-1": ResultReadyOutcome.SUCCESS,
            "wu-2": ResultReadyOutcome.SUCCESS,
        },
        diagnostics_summary="all green",
        conflict_paths=[],
        cost_total=cost,
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorReduceSSEEvent)
    assert sse.event == "coordinator_reduce"
    assert isinstance(sse.data, CoordinatorReduceEventData)
    assert sse.data.coordinator_run_id == "r1"
    assert sse.data.group_outcome == "success"
    assert sse.data.per_worker_outcomes == {"wu-1": "success", "wu-2": "success"}
    assert sse.data.diagnostics_summary == "all green"
    assert sse.data.conflict_paths == []
    # CostAggregate re-validates from its dict form, so the wire shape stays
    # a CostAggregate after round-tripping through BaseEventData.from_event.
    assert isinstance(sse.data.cost_total, CostAggregate)
    assert sse.data.cost_total.total_input_tokens == 10
    assert sse.data.cost_total.total_output_tokens == 20
    assert sse.data.cost_total.total_usd == 0.5
    assert sse.data.cost_total.tool_call_count == 3


def test_apply_maps_to_sse() -> None:
    event = CoordinatorApplyEvent(
        **_lineage_kwargs(),
        apply_status="success",
        file_count=4,
        total_bytes=1024,
        failed_at_path=None,
        rollback_status=None,
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorApplySSEEvent)
    assert sse.event == "coordinator_apply"
    assert isinstance(sse.data, CoordinatorApplyEventData)
    assert sse.data.coordinator_run_id == "r1"
    assert sse.data.apply_status == "success"
    assert sse.data.file_count == 4
    assert sse.data.total_bytes == 1024
    assert sse.data.failed_at_path is None
    assert sse.data.rollback_status is None


def test_sibling_cancel_maps_to_sse() -> None:
    event = CoordinatorSiblingCancelEvent(
        **_lineage_kwargs(),
        triggered_by_work_unit_id="wu-1",
        triggered_by_outcome=ResultReadyOutcome.FAILED,
        cancelled_work_unit_ids=["wu-2", "wu-3"],
        reason="fail-fast on first hard failure",
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorSiblingCancelSSEEvent)
    assert sse.event == "coordinator_sibling_cancel"
    assert isinstance(sse.data, CoordinatorSiblingCancelEventData)
    assert sse.data.coordinator_run_id == "r1"
    assert sse.data.triggered_by_work_unit_id == "wu-1"
    assert sse.data.triggered_by_outcome == "failed"
    assert sse.data.cancelled_work_unit_ids == ["wu-2", "wu-3"]
    assert sse.data.reason == "fail-fast on first hard failure"


def test_lineage_when_none() -> None:
    """Root-session events (all lineage = None) still serialize cleanly.

    All five lineage tags default to ``None`` on the domain mixin and on the
    SSE data class so events emitted before any coordinator context is
    established still validate end-to-end.
    """
    event = CoordinatorDispatchEvent(
        step_id="step-1",
        work_unit_count=1,
        work_unit_ids=["wu-1"],
        phases=["exploration"],
    )

    sse = EventMapper.event_to_sse_event(event)

    assert isinstance(sse, CoordinatorDispatchSSEEvent)
    assert sse.event == "coordinator_dispatch"
    assert sse.data.root_session_id is None
    assert sse.data.parent_session_id is None
    assert sse.data.child_session_id is None
    assert sse.data.coordinator_run_id is None
    assert sse.data.work_unit_id is None
    assert sse.data.step_id == "step-1"
    assert sse.data.work_unit_count == 1


def test_reduce_model_dump_json_roundtrip() -> None:
    """SSE wire serialization (model_dump_json) is parseable and preserves
    enum values, lineage fields, and the nested CostAggregate.
    """
    cost = CostAggregate(
        total_input_tokens=7,
        total_output_tokens=11,
        total_usd=0.12,
        tool_call_count=2,
    )
    event = CoordinatorReduceEvent(
        **_lineage_kwargs(),
        group_outcome=GroupOutcome.MIXED,
        per_worker_outcomes={
            "wu-1": ResultReadyOutcome.SUCCESS,
            "wu-2": ResultReadyOutcome.FAILED,
        },
        diagnostics_summary="one worker failed",
        conflict_paths=["src/a.py"],
        cost_total=cost,
    )

    sse = EventMapper.event_to_sse_event(event)
    payload = json.loads(sse.to_sse_data_json())

    assert payload["coordinator_run_id"] == "r1"
    assert payload["group_outcome"] == "mixed"
    assert payload["per_worker_outcomes"] == {"wu-1": "success", "wu-2": "failed"}
    assert payload["diagnostics_summary"] == "one worker failed"
    assert payload["conflict_paths"] == ["src/a.py"]
    assert payload["cost_total"]["total_input_tokens"] == 7
    assert payload["cost_total"]["total_output_tokens"] == 11
    assert payload["cost_total"]["total_usd"] == 0.12
    assert payload["cost_total"]["tool_call_count"] == 2
