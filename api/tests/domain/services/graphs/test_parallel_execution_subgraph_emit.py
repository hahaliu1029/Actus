"""C2 PR-8 §13 Task 8.4 — emit tests for parallel_execution_subgraph.

Asserts that `_first_time_dispatch` and `reducer_node` emit the new lineage-
tagged coordinator events onto the `configurable.event_queue`:
  - 1 CoordinatorDispatchEvent + N CoordinatorWorkerSpawnedEvent on dispatch
  - 1 CoordinatorReduceEvent on reducer success
  - 0 events when dispatch raises mid-way (rollback path)
"""
from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.event import (
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorWorkerSpawnedEvent,
)
from app.domain.models.mailbox_envelope import (
    CostAggregate,
    ResultReadyOutcome,
)
from app.domain.models.work_unit import ProposedPath, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    dispatch_node,
    reducer_node,
)


def _mk_session(sid: str) -> MagicMock:
    m = MagicMock()
    m.id = sid
    return m


def _base_state(work_unit_requests=None) -> dict:
    return {
        "coordinator_run_id": None,
        "step_id": "step-abc",
        "work_unit_requests": work_unit_requests or [
            WorkUnitRequest(
                objective="explore X", phase="exploration",
                allowed_tools=["file_read"],
            ),
            WorkUnitRequest(
                objective="explore Y", phase="exploration",
                allowed_tools=["file_read"],
            ),
        ],
        "work_units": [],
        "parent_session_id": "parent1",
        "user_id": "u1",
        "root_session_id": "root1",
        "child_session_ids": {},
        "orchestrator_task": None,
        "worker_results": [],
        "apply_plan": None,
        "group_outcome": None,
        "step_result_candidate": None,
    }


def _base_config(*, peek_returns: int | None = None) -> dict:
    rehydrate = AsyncMock()
    rehydrate.detect_existing_run = AsyncMock(return_value=None)
    session_service = AsyncMock()
    session_service.peek_coordinator_attempt = AsyncMock(return_value=peek_returns)
    session_service.bump_coordinator_attempt = AsyncMock(return_value=1)
    session_service.create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1"), _mk_session("c2")],
    )
    runner_starter = AsyncMock()
    runner_starter.start = AsyncMock()
    publisher = AsyncMock()
    artifact = AsyncMock()
    artifact.put_content_addressed_bytes = AsyncMock(
        return_value="minio://manifest-ref",
    )
    parent_sandbox = AsyncMock()
    parent_sandbox.compute_digest = AsyncMock(return_value="sha256_abc")
    parent_sandbox.read_file = AsyncMock(return_value=b"content")
    orchestrator = AsyncMock()
    orchestrator.run = AsyncMock()
    orchestrator_factory = MagicMock()
    orchestrator_factory.build = MagicMock(return_value=orchestrator)
    subscriber = AsyncMock()
    subscriber.subscribe = AsyncMock()
    return {
        "configurable": {
            "rehydrate_service": rehydrate,
            "session_service": session_service,
            "child_runner_starter": runner_starter,
            "mailbox_publisher": publisher,
            "mailbox_subscriber": subscriber,
            "artifact_storage": artifact,
            "parent_sandbox": parent_sandbox,
            "orchestrator_factory": orchestrator_factory,
            "cancel_event": asyncio.Event(),
        }
    }


def _drain(queue: asyncio.Queue) -> list:
    """Synchronously pull every item currently on the queue."""
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


@pytest.mark.anyio
async def test_first_time_dispatch_emits_dispatch_plus_spawned_events() -> None:
    config = _base_config(peek_returns=None)
    queue: asyncio.Queue = asyncio.Queue()
    config["configurable"]["event_queue"] = queue
    state = _base_state()

    await dispatch_node(state, config)

    events = _drain(queue)
    # 1 dispatch + 2 spawned
    assert len(events) == 3
    dispatch_evs = [e for e in events if isinstance(e, CoordinatorDispatchEvent)]
    spawned_evs = [e for e in events if isinstance(e, CoordinatorWorkerSpawnedEvent)]
    assert len(dispatch_evs) == 1
    assert len(spawned_evs) == 2

    dispatch_ev = dispatch_evs[0]
    assert dispatch_ev.step_id == "step-abc"
    assert dispatch_ev.work_unit_count == 2
    assert len(dispatch_ev.work_unit_ids) == 2
    assert all(p == "exploration" for p in dispatch_ev.phases)
    assert dispatch_ev.coordinator_run_id is not None
    assert dispatch_ev.root_session_id == "root1"
    assert dispatch_ev.parent_session_id == "parent1"

    # Each spawned event has lineage + work_unit_id + child_session_id
    child_ids = {ev.child_session_id for ev in spawned_evs}
    assert child_ids == {"c1", "c2"}
    for ev in spawned_evs:
        assert ev.phase == "exploration"
        assert ev.objective in {"explore X", "explore Y"}
        assert ev.allowed_tools == ["file_read"]
        assert ev.write_lease_count == 0
        assert ev.coordinator_run_id == dispatch_ev.coordinator_run_id
        assert ev.root_session_id == "root1"
        assert ev.parent_session_id == "parent1"


@pytest.mark.anyio
async def test_dispatch_failure_mid_way_emits_no_events() -> None:
    """When dispatch raises after the preflight (e.g. parent_sandbox digest
    fails) we MUST NOT emit dispatch/spawned events — emit happens only
    after the body completes."""
    config = _base_config(peek_returns=None)
    queue: asyncio.Queue = asyncio.Queue()
    config["configurable"]["event_queue"] = queue
    # Modify lease forces a digest computation; make it fail
    state = _base_state(work_unit_requests=[
        WorkUnitRequest(
            objective="touch existing", phase="write",
            allowed_tools=["file_write"],
            proposed_paths=[ProposedPath(path="existing.py", op="modify")],
        ),
    ])
    config["configurable"]["session_service"].create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1")],
    )
    config["configurable"]["parent_sandbox"].compute_digest = AsyncMock(
        return_value=None,
    )
    with pytest.raises(ValueError):
        await dispatch_node(state, config)

    assert _drain(queue) == []


@pytest.mark.anyio
async def test_first_time_dispatch_no_event_queue_does_not_raise() -> None:
    """No event_queue in configurable → emits skipped, dispatch happy."""
    config = _base_config(peek_returns=None)
    # NB: no event_queue key at all
    state = _base_state()
    cmd = await dispatch_node(state, config)
    assert cmd is not None


@pytest.mark.anyio
async def test_reducer_node_emits_reduce_event() -> None:
    """Reducer happy-path emits exactly one CoordinatorReduceEvent with
    lineage + group_outcome + cost_total fallback to CostAggregate()."""
    from app.application.services.patch_reducer_service import (
        ReducerDiagnostics,
        ReducerOutput,
    )
    from app.domain.models.patch_apply_plan import GroupOutcome
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

    wu = WorkUnit(
        work_unit_id="wu-1",
        objective="explore X", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )
    wr = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
        patch_manifest=None,
        needs_authorization_details=None,
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "event_queue": queue,
        },
    }
    state = {
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "parent_session_id": "parent1",
        "user_id": "u1",
        "work_units": [wu],
        "worker_results": [wr],
        "quota_acquired": False,
    }

    await reducer_node(state, config)

    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    assert ev.group_outcome == GroupOutcome.SUCCESS
    assert ev.per_worker_outcomes == {"wu-1": ResultReadyOutcome.SUCCESS}
    assert ev.diagnostics_summary == "ok"  # diagnostics object is non-None
    assert ev.conflict_paths == []
    assert isinstance(ev.cost_total, CostAggregate)
    assert ev.coordinator_run_id == "r1"
    assert ev.root_session_id == "root1"
    assert ev.parent_session_id == "parent1"


@pytest.mark.anyio
async def test_reducer_node_emit_failure_does_not_mask_command() -> None:
    """An emit failure must not abort reducer_node."""
    from app.application.services.patch_reducer_service import (
        ReducerDiagnostics,
        ReducerOutput,
    )
    from app.domain.models.patch_apply_plan import GroupOutcome
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

    wu = WorkUnit(
        work_unit_id="wu-1",
        objective="explore X", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )
    wr = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
        patch_manifest=None,
        needs_authorization_details=None,
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    # Bad queue: put_nowait raises (codex R4 P2 — emit now uses
    # put_nowait for cancellation safety; mock the actual call site).
    queue = MagicMock()
    queue.put_nowait = MagicMock(side_effect=RuntimeError("queue exploded"))
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "event_queue": queue,
        },
    }
    state = {
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "parent_session_id": "parent1",
        "user_id": "u1",
        "work_units": [wu],
        "worker_results": [wr],
        "quota_acquired": False,
    }

    cmd = await reducer_node(state, config)
    # We don't crash, and the Command(update=...) still carries reducer outputs.
    assert cmd is not None
