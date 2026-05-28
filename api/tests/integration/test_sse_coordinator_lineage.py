"""C2 PR-8 §13 Task 8.4 — integration: 5 coordinator events on event_queue.

End-to-end-style coverage: exercise the actual subgraph dispatch + reducer
nodes wired to an asyncio.Queue (the production event_queue surface) and
assert the 5 lineage-tagged events arrive in the right order. SSE-side
reconnect is already covered by ``redis_event_recovery`` infra; this file
pins the emit-side contract that feeds it.

NB: This file lives under tests/integration/ per the C2 PR-8 spec
(``tests/integration/test_sse_coordinator_lineage.py``) but does NOT
require Postgres/Redis — collaborators are mocked. We override the
auto-use ``_migrate`` fixture so collection doesn't drag in alembic.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture(scope="module", autouse=True)
def _migrate():  # noqa: PT004
    """Override the parent integration conftest's autouse migration so
    this DB-free file does not require a running Postgres."""
    yield

from app.application.services.patch_reducer_service import (
    ReducerDiagnostics,
    ReducerOutput,
)
from app.domain.models.event import (
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorWorkerSpawnedEvent,
)
from app.domain.models.mailbox_envelope import (
    CostAggregate,
    ResultReadyOutcome,
)
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.work_unit import WorkUnit, WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    WorkerResult,
    dispatch_node,
    reducer_node,
)


def _mk_session(sid: str) -> MagicMock:
    m = MagicMock()
    m.id = sid
    return m


def _dispatch_config(*, event_queue: asyncio.Queue) -> dict:
    rehydrate = AsyncMock()
    rehydrate.detect_existing_run = AsyncMock(return_value=None)
    session_service = AsyncMock()
    session_service.peek_coordinator_attempt = AsyncMock(return_value=None)
    session_service.bump_coordinator_attempt = AsyncMock(return_value=1)
    session_service.create_session_with_parent = AsyncMock(
        side_effect=[_mk_session("c1"), _mk_session("c2")],
    )
    runner_starter = AsyncMock()
    publisher = AsyncMock()
    artifact = AsyncMock()
    artifact.put_content_addressed_bytes = AsyncMock(
        return_value="minio://manifest-ref",
    )
    parent_sandbox = AsyncMock()
    orchestrator = AsyncMock()
    orchestrator_factory = MagicMock()
    orchestrator_factory.build = MagicMock(return_value=orchestrator)
    subscriber = AsyncMock()
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
            "event_queue": event_queue,
        }
    }


@pytest.mark.integration
@pytest.mark.anyio
async def test_dispatch_and_reducer_flow_emits_lineage_events_on_queue() -> None:
    """Full happy-path: dispatch → reducer drains 1 + N + 1 = N+2 events."""
    queue: asyncio.Queue = asyncio.Queue()
    config = _dispatch_config(event_queue=queue)
    state = {
        "coordinator_run_id": None,
        "step_id": "step-integ",
        "work_unit_requests": [
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
        "parent_session_id": "p-integ",
        "user_id": "u-integ",
        "root_session_id": "root-integ",
        "child_session_ids": {},
        "orchestrator_task": None,
        "worker_results": [],
        "apply_plan": None,
        "group_outcome": None,
        "step_result_candidate": None,
    }

    dispatch_cmd = await dispatch_node(state, config)
    # Dispatch produced N WorkUnit objects in state-update.
    enriched_units: list[WorkUnit] = dispatch_cmd.update["work_units"]
    coordinator_run_id = dispatch_cmd.update["coordinator_run_id"]

    # Now wire a reducer that returns SUCCESS.
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))
    reducer_config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "event_queue": queue,
        },
    }
    reducer_state = {
        "coordinator_run_id": coordinator_run_id,
        "root_session_id": "root-integ",
        "parent_session_id": "p-integ",
        "user_id": "u-integ",
        "work_units": enriched_units,
        "worker_results": [
            WorkerResult(
                work_unit_id=enriched_units[0].work_unit_id,
                child_session_id="c1",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
                summary="ok",
                patch_manifest=None,
                needs_authorization_details=None,
            ),
            WorkerResult(
                work_unit_id=enriched_units[1].work_unit_id,
                child_session_id="c2",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
                summary="ok",
                patch_manifest=None,
                needs_authorization_details=None,
            ),
        ],
        "quota_acquired": False,
    }
    await reducer_node(reducer_state, reducer_config)

    # Drain queue and validate the 4-event sequence.
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert len(drained) == 4

    # First event is the dispatch.
    assert isinstance(drained[0], CoordinatorDispatchEvent)
    assert drained[0].work_unit_count == 2
    assert drained[0].root_session_id == "root-integ"
    assert drained[0].parent_session_id == "p-integ"

    # Next two are the per-WorkUnit spawn events.
    spawned = drained[1:3]
    assert all(isinstance(ev, CoordinatorWorkerSpawnedEvent) for ev in spawned)
    assert {ev.child_session_id for ev in spawned} == {"c1", "c2"}

    # Last is the reducer event.
    assert isinstance(drained[3], CoordinatorReduceEvent)
    assert drained[3].group_outcome == GroupOutcome.SUCCESS
    assert drained[3].coordinator_run_id == coordinator_run_id
    assert drained[3].root_session_id == "root-integ"
    assert drained[3].parent_session_id == "p-integ"
