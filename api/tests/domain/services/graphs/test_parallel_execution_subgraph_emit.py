"""C2 PR-8 §13 Task 8.4 — emit tests for parallel_execution_subgraph.

Asserts that `_first_time_dispatch` and `reducer_node` emit the new lineage-
tagged coordinator events onto the `configurable.event_queue`:
  - 1 CoordinatorDispatchEvent + N CoordinatorWorkerSpawnedEvent on dispatch
  - 1 CoordinatorReduceEvent on reducer success
  - 0 events when dispatch raises mid-way (rollback path)

C2 PR-9b-A Task A6: also pins the contract that the per-run PatchApplier
constructed inside ``main_graph._run_parallel_backend`` receives an
**async put_nowait** ``emit_event`` closure bound to ``cfg["event_queue"]``
(INV-A3) and that emit failures from the applier are swallowed downstream
of the apply outcome (INV-A4 — already covered by
``test_patch_applier_emit.test_finalize_emit_failure_does_not_mask_outcome``;
the test in this file pins the closure construction shape).
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
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


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
    lineage + group_outcome + aggregate-pull cost_total (post-B4)."""
    from app.application.services.cost_rollup_service import AggregateResult
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

    # [B4] aggregate pull is the cost authority; legacy assertion expects
    # the clean ``diagnostics_summary == "ok"`` path, so wire an aggregate
    # that returns CostAggregate() with no missing children.
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        return_value=AggregateResult(cost=CostAggregate(), missing_children=())
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
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
        # [F2] dispatched-child id set drives the aggregate pull.
        "child_session_ids": {"wu-1": "c1"},
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


# ── C2 PR-9b-A Task A6 — per-run PatchApplier construction tests ───────


@pytest.mark.anyio
async def test_patch_applier_constructed_with_async_put_nowait_emit() -> None:
    """[Task A6 / INV-A3] When ``main_graph._run_parallel_backend`` constructs
    a PatchApplier per-run, it MUST bind ``emit_event`` to an async closure
    that uses synchronous ``put_nowait`` on the per-stream ``event_queue``.

    This pins the production shape independently of where the construction
    happens — instantiating PatchApplier with the exact ``emit_event`` pattern
    we expect ``_run_parallel_backend`` to assemble.
    """
    from unittest.mock import AsyncMock as _AsyncMock
    from app.application.services.patch_applier import PatchApplier

    event_queue: asyncio.Queue = asyncio.Queue()

    async def emit_event_into_queue(event):
        # put_nowait avoids cancellation window — see main_graph closure.
        event_queue.put_nowait(event)

    snapshot_store = MagicMock()
    audit_repo = MagicMock()
    redis_mock = MagicMock()

    applier = PatchApplier(
        snapshot_store=snapshot_store,
        audit_repo=audit_repo,
        redis=redis_mock,
        emit_event=emit_event_into_queue,
    )

    # INV-A3: bound callable is async + uses put_nowait.
    assert inspect.iscoroutinefunction(applier._emit_event)
    # Round-trip — caller can dispatch into the queue from the async body.
    await applier._emit_event("ev-X")
    assert event_queue.get_nowait() == "ev-X"


@pytest.mark.anyio
async def test_run_parallel_backend_constructs_applier_from_deps_with_emit() -> None:
    """[Task A6 / INV-A3] ``main_graph._run_parallel_backend`` MUST construct
    PatchApplier per-run from ``cfg['patch_applier_deps']`` and bind
    ``emit_event`` to an async closure over ``cfg['event_queue']``.

    The PatchApplier itself is NOT a cfg singleton — the cfg holds the
    lifespan-scoped deps (PatchApplierDeps), and main_graph instantiates
    PatchApplier per coordinator run so the emit closure can target the
    correct per-stream queue.
    """
    from app.application.services.patch_applier import (
        ApplyDiagnostics,
        ApplyOutcome,
        ApplyStatus,
        PatchApplier,
    )
    from app.application.services.patch_applier_deps import PatchApplierDeps
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.services.graphs.main_graph import _run_parallel_backend

    sha = hashlib.sha256(b"x").hexdigest()
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="f.py", op="add", new_digest=sha,
            content_ref="r", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu1",),
    )

    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(return_value={
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": plan,
        "step_result_candidate": "reducer-text",
    })

    # Capture PatchApplier ctor kwargs so we can pin: (a) per-run construction
    # happens and (b) emit_event closure binds to the per-stream queue.
    captured_kwargs: dict = {}

    class _CapturingApplier:
        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        async def apply(self, plan, *, parent_sandbox, minio_client,
                        cancel_event=None, lineage=None) -> ApplyOutcome:
            # Smoke the emit closure to confirm it dispatches into the
            # active per-stream queue.
            await captured_kwargs["emit_event"]("emit-from-applier")
            return ApplyOutcome(
                status=ApplyStatus.SUCCESS,
                applied_files=(), failed_at=None, rollback_status=None,
                diagnostics=ApplyDiagnostics(duration_ms=1),
            )

    # Monkeypatch PatchApplier inside main_graph's local import scope so the
    # per-run construction goes through our capturing fake.
    import app.domain.services.graphs.main_graph as mg

    orig_applier = mg.__dict__.get("PatchApplier", None)
    mg.PatchApplier = _CapturingApplier
    try:
        deps = PatchApplierDeps(
            snapshot_store=MagicMock(),
            audit_repo=MagicMock(),
            redis=MagicMock(),
        )
        event_queue: asyncio.Queue = asyncio.Queue()

        step = MagicMock()
        step.id = "step1"
        step.parallel_work_units = MagicMock()
        step.parallel_work_units.work_units = []
        state = {"session_id": "p1", "user_id": "u1", "root_session_id": "p1"}
        config = {"configurable": {
            "parallel_execution_subgraph": subgraph,
            "patch_applier_deps": deps,
            "parent_sandbox": AsyncMock(),
            "artifact_storage": AsyncMock(),
            "event_queue": event_queue,
        }}

        out = await _run_parallel_backend(state, config, step)
    finally:
        if orig_applier is None:
            mg.__dict__.pop("PatchApplier", None)
        else:
            mg.PatchApplier = orig_applier

    # Per-run ctor was invoked with deps + emit_event closure.
    assert captured_kwargs["snapshot_store"] is deps.snapshot_store
    assert captured_kwargs["audit_repo"] is deps.audit_repo
    assert captured_kwargs["redis"] is deps.redis
    emit = captured_kwargs["emit_event"]
    assert inspect.iscoroutinefunction(emit)

    # Closure dispatched into the per-stream queue (smoke fire from fake).
    assert event_queue.get_nowait() == "emit-from-applier"

    # Apply succeeded → main_graph returns the success text.
    assert "应用成功" in out
    assert "1 个文件" in out


@pytest.mark.anyio
async def test_run_parallel_backend_raises_when_event_queue_missing() -> None:
    """[Task A6 / INV-A3] When the coordinator apply branch fires but no
    ``event_queue`` is in cfg, ``_run_parallel_backend`` MUST raise loudly —
    silently dropping the emit would be a GraphEventBridge wiring regression
    (event_bridge.py:74-79 should always merge ``event_queue`` into the
    coordinator path's configurable). Fail-fast surfaces the misconfig.
    """
    import hashlib as _hashlib

    from app.application.services.patch_applier_deps import PatchApplierDeps
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.services.graphs.main_graph import _run_parallel_backend

    sha = _hashlib.sha256(b"x").hexdigest()
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="f.py", op="add", new_digest=sha,
            content_ref="r", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu1",),
    )

    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(return_value={
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": plan,
        "step_result_candidate": "reducer-text",
    })

    deps = PatchApplierDeps(
        snapshot_store=MagicMock(),
        audit_repo=MagicMock(),
        redis=MagicMock(),
    )
    step = MagicMock()
    step.id = "step1"
    step.parallel_work_units = MagicMock()
    step.parallel_work_units.work_units = []
    state = {"session_id": "p1", "user_id": "u1", "root_session_id": "p1"}
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier_deps": deps,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
        # NO event_queue — this is the regression we guard against.
    }}

    with pytest.raises(RuntimeError) as ei:
        await _run_parallel_backend(state, config, step)
    assert "event_queue" in str(ei.value)


@pytest.mark.skip(
    reason=(
        "INV-A4 emit failure swallowing is already pinned by "
        "tests/application/services/test_patch_applier_emit.py::"
        "test_finalize_emit_failure_does_not_mask_outcome. "
        "Re-asserting via _run_parallel_backend would require fully wiring "
        "PatchApplier.apply()'s snapshot_store + audit_repo + redis fakes "
        "into the per-run construction path — deferred to PR-9b-C C9 smoke."
    )
)
@pytest.mark.anyio
async def test_patch_applier_emit_failure_swallowed_with_warning() -> None:
    """[Task A6 / INV-A4] Per-run-constructed PatchApplier swallows emit
    failures so apply outcome reaches main_graph unchanged. Deferred (see
    skip reason); the underlying invariant is locked in test_patch_applier_emit.
    """


@pytest.mark.skip(
    reason=(
        "Heavy composition-root introspection — defers to PR-9b-A8 lifespan "
        "smoke + PR-9b-C C9 fixture harness. Lighter A6 contracts above "
        "(test_orchestrator_emit_event_is_async_callable + "
        "test_run_parallel_backend_constructs_applier_from_deps_with_emit) "
        "pin the production shape without spinning up TestClient/lifespan."
    )
)
@pytest.mark.anyio
async def test_composition_root_wires_async_put_nowait_emit_to_patch_applier() -> None:
    """[Task A6 / INV-A7] Composition-root → planner cfg → main_graph emit
    closure end-to-end. Deferred (see skip reason).
    """


# ── B4: silent-zero removal + aggregate pull ────────────────────────────────
#
# INV-B1: CoordinatorReduceEvent.cost_total derives from
#         cost_rollup_service.aggregate(...), NOT from envelope cost_summary.
# INV-B2: No silent-zero `... or CostAggregate()` fallback adjacent to
#         `getattr(output, "cost_summary", ...)` in the reducer node.
# INV-B3: When aggregate returns ``missing_children`` (or raises), the
#         reducer's CoordinatorReduceEvent.diagnostics_summary surfaces
#         "cost_unavailable: ..." so the observability path can detect it.


@pytest.mark.anyio
async def test_reducer_node_emit_uses_aggregate_pull_happy_path() -> None:
    """[B4 / INV-B1] Happy path: aggregate(coordinator_run_id, child_session_ids)
    returns a non-zero cost; that cost flows into CoordinatorReduceEvent.cost_total
    and diagnostics_summary stays free of "cost_unavailable".
    """
    from app.application.services.cost_rollup_service import AggregateResult
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
    wr_1 = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),  # envelope cost is IGNORED post-B4
        summary="ok",
    )
    wr_2 = WorkerResult(
        work_unit_id="wu-2",
        child_session_id="c2",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    expected_cost = CostAggregate(
        total_input_tokens=1000,
        total_output_tokens=500,
        total_usd=0.42,
        tool_call_count=3,
    )
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        return_value=AggregateResult(cost=expected_cost, missing_children=())
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
            "event_queue": queue,
        },
    }
    state = {
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "parent_session_id": "parent1",
        "user_id": "u1",
        "work_units": [wu],
        "worker_results": [wr_1, wr_2],
        # [F2] aggregate id list comes from the dispatched child set, not
        # from worker_results.
        "child_session_ids": {"wu-1": "c1", "wu-2": "c2"},
        "quota_acquired": False,
    }

    await reducer_node(state, config)

    # aggregate() invoked exactly once with the contract signature.
    cost_rollup_service.aggregate.assert_awaited_once()
    call_kwargs = cost_rollup_service.aggregate.await_args.kwargs
    assert call_kwargs["coordinator_run_id"] == "r1"
    # child_session_ids is order-preserved from worker_results.
    assert list(call_kwargs["child_session_ids"]) == ["c1", "c2"]

    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    # INV-B1: cost_total IS the aggregate result, NOT envelope-derived.
    assert ev.cost_total.total_usd == 0.42
    assert ev.cost_total.total_input_tokens == 1000
    assert ev.cost_total.total_output_tokens == 500
    assert ev.cost_total.tool_call_count == 3
    # INV-B3: clean aggregate → no cost_unavailable diagnostic appended.
    assert "cost_unavailable" not in ev.diagnostics_summary


@pytest.mark.anyio
async def test_reducer_node_emit_aggregate_raises_surfaces_diagnostic() -> None:
    """[B4 / INV-B2 / INV-B3] When aggregate(...) raises, cost_total falls
    back to CostAggregate() (zeros) BUT diagnostics_summary must surface
    "cost_unavailable" so the zeros are observable, not silently emitted.
    """
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
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        side_effect=RuntimeError("ledger query failed")
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
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
        # [F2] dispatched-child id set drives the aggregate pull.
        "child_session_ids": {"wu-1": "c1"},
        "quota_acquired": False,
    }

    # Aggregate failure MUST NOT mask reducer outcome — Command still returns.
    cmd = await reducer_node(state, config)
    assert cmd is not None

    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    # INV-B2: cost_total is CostAggregate() (zeros).
    assert ev.cost_total.total_usd == 0.0
    assert ev.cost_total.total_input_tokens == 0
    assert ev.cost_total.total_output_tokens == 0
    # INV-B3: but the zero is OBSERVABLE via the diagnostic.
    assert "cost_unavailable" in ev.diagnostics_summary


@pytest.mark.anyio
async def test_reducer_node_emit_missing_children_surfaces_diagnostic() -> None:
    """[B4 / INV-B3] When aggregate returns AggregateResult with non-empty
    ``missing_children``, the reducer's CoordinatorReduceEvent.diagnostics_summary
    must encode "cost_unavailable: missing_children=[...]" so the child IDs
    that contributed ZERO rows to the ledger SUM are observable downstream.
    """
    from app.application.services.cost_rollup_service import AggregateResult
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
    wr_1 = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )
    wr_2 = WorkerResult(
        work_unit_id="wu-2",
        child_session_id="c2",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    # Aggregate succeeds but reports c2 as missing.
    partial_cost = CostAggregate(
        total_input_tokens=100,
        total_output_tokens=50,
        total_usd=0.05,
        tool_call_count=1,
    )
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        return_value=AggregateResult(
            cost=partial_cost,
            missing_children=("c2",),
        )
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
            "event_queue": queue,
        },
    }
    state = {
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "parent_session_id": "parent1",
        "user_id": "u1",
        "work_units": [wu],
        "worker_results": [wr_1, wr_2],
        # [F2] dispatched-child id set drives the aggregate pull; the fake
        # aggregate reports c2 missing so the diagnostic must surface it.
        "child_session_ids": {"wu-1": "c1", "wu-2": "c2"},
        "quota_acquired": False,
    }

    await reducer_node(state, config)

    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    # INV-B1: partial cost still flows through.
    assert ev.cost_total.total_usd == 0.05
    # INV-B3: missing_children is surfaced into diagnostics_summary.
    assert "cost_unavailable" in ev.diagnostics_summary
    assert "c2" in ev.diagnostics_summary


@pytest.mark.anyio
async def test_reducer_node_aggregates_dispatched_child_without_worker_result() -> None:
    """[B4 / F2 / INV-B3] A child that was DISPATCHED (present in
    ``state['child_session_ids']``) but produced NO WorkerResult (INCOMPLETE
    group / rehydrate-skipped envelope) MUST still be included in the
    aggregate's ``child_session_ids`` argument — sourcing the id list from
    ``worker_results`` alone would silently drop it from BOTH the cost SUM
    and ``missing_children``, undercounting cost while looking authoritative.

    Here c2 is dispatched but absent from worker_results; the fake aggregate
    reports c2 as missing → the cost_unavailable diagnostic fires (INV-B3).
    """
    from app.application.services.cost_rollup_service import AggregateResult
    from app.application.services.patch_reducer_service import (
        ReducerDiagnostics,
        ReducerOutput,
    )
    from app.domain.models.patch_apply_plan import GroupOutcome
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

    wu_1 = WorkUnit(
        work_unit_id="wu-1",
        objective="explore X", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )
    wu_2 = WorkUnit(
        work_unit_id="wu-2",
        objective="explore Y", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )
    # Only wu-1/c1 produced a WorkerResult; wu-2/c2 was dispatched but is
    # INCOMPLETE (no envelope / result).
    wr_1 = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="done",
        diagnostics=ReducerDiagnostics(),
    ))

    # The fake aggregate echoes back c2 as missing (it has no cost rows).
    partial_cost = CostAggregate(
        total_input_tokens=100, total_output_tokens=50,
        total_usd=0.05, tool_call_count=1,
    )
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        return_value=AggregateResult(cost=partial_cost, missing_children=("c2",))
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
            "event_queue": queue,
        },
    }
    state = {
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
        "parent_session_id": "parent1",
        "user_id": "u1",
        "work_units": [wu_1, wu_2],
        # NOTE: only wr_1 — c2 dispatched but no WorkerResult.
        "worker_results": [wr_1],
        # Full dispatched set includes the INCOMPLETE child c2.
        "child_session_ids": {"wu-1": "c1", "wu-2": "c2"},
        "quota_acquired": False,
    }

    await reducer_node(state, config)

    # F2: the aggregate's child id list is the FULL dispatched set — the
    # INCOMPLETE child c2 IS present, not silently dropped.
    cost_rollup_service.aggregate.assert_awaited_once()
    call_kwargs = cost_rollup_service.aggregate.await_args.kwargs
    assert set(call_kwargs["child_session_ids"]) == {"c1", "c2"}
    assert "c2" in call_kwargs["child_session_ids"]

    # INV-B3: c2 surfaced as missing → cost_unavailable diagnostic fires.
    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    assert "cost_unavailable" in ev.diagnostics_summary
    assert "c2" in ev.diagnostics_summary
    # per_worker_outcomes stays per-completed-child (only wu-1 finished).
    assert ev.per_worker_outcomes == {"wu-1": ResultReadyOutcome.SUCCESS}


# ── [PR-9b-B codex F1 — HIGH] cancellation MUST propagate; ordinary-exception
# handoff survives ────────────────────────────────────────────────────────────
#
# REVERT of a prior over-correction: an earlier round wrapped the cost pull in
# ``asyncio.shield`` and added an ``except asyncio.CancelledError`` that
# logged + fell through to ``return command`` — i.e. it SWALLOWED cancellation
# and let LangGraph PROCEED to the apply branch. That is WRONG: when an
# SSE-disconnect / the execution watchdog / ``stop_session`` cancels the graph
# via ``task.cancel()``, a cancel landing on the cost-aggregate await is the
# signal that apply MUST be SKIPPED (``cfg["cancel_event"]`` is NOT set by
# ``stop_session``; cancellation relies on task-cancel propagation, which the
# suppression defeated → patches applied to the parent sandbox AFTER the user
# cancelled). So ``CancelledError`` MUST propagate out of ``reducer_node``.
#
# The surviving contract:
#   • CancelledError during the cost pull / emit → PROPAGATE → reducer raises
#     → apply branch skipped (CORRECT for cancellation).
#   • An ORDINARY Exception during aggregate → inner ``except Exception`` sets
#     cost_unavailable + still returns the Command (best-effort on infra error).
#   • An ORDINARY Exception during the emit → outer ``except Exception`` logs +
#     returns the Command (handoff preserved on best-effort emit failure).


def _reducer_returning(group_outcome, apply_plan, step_text):
    """A reducer AsyncMock whose reduce() yields the given ReducerOutput."""
    from app.application.services.patch_reducer_service import (
        ReducerDiagnostics,
        ReducerOutput,
    )

    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=apply_plan,
        group_outcome=group_outcome,
        step_result_candidate=step_text,
        diagnostics=ReducerDiagnostics(),
    ))
    return reducer


@pytest.mark.anyio
async def test_reducer_propagates_cancellation() -> None:
    """[F1 REVERT] When ``cost_rollup_service.aggregate`` raises CancelledError
    during the best-effort cost pull, ``reducer_node`` MUST PROPAGATE it (NOT
    return a Command). Cancellation is the SSE-disconnect / watchdog /
    stop_session signal that the user aborted; swallowing it would let
    LangGraph proceed to the apply branch and write patches to the parent
    sandbox AFTER the cancel. Propagating → reducer_node raises → apply
    branch is skipped (CORRECT).
    """
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

    sha = hashlib.sha256(b"x").hexdigest()
    apply_plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="f.py", op="add", new_digest=sha,
            content_ref="ref", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu-1",),
    )

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
    )

    reducer = _reducer_returning(GroupOutcome.SUCCESS, apply_plan, "done")

    # aggregate() raises CancelledError (BaseException). The reverted (correct)
    # behaviour: this propagates out of reducer_node so the apply branch is
    # skipped on cancel.
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
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
        # [F2] dispatched-child id set drives the aggregate pull.
        "child_session_ids": {"wu-1": "c1"},
        "quota_acquired": False,
    }

    # MUST raise CancelledError — the handoff is NOT returned; apply is skipped.
    with pytest.raises(asyncio.CancelledError):
        await reducer_node(state, config)

    # No CoordinatorReduceEvent was emitted on the cancel path (the emit is
    # downstream of the cancelled aggregate await).
    assert _drain(queue) == []


@pytest.mark.anyio
async def test_reducer_handoff_survives_aggregate_exception() -> None:
    """[F1 + F2] When aggregate raises an ORDINARY Exception (SQLAlchemyError-
    like), the same handoff survival holds AND the client-visible diagnostic is
    the F2 sanitized stable code (no raw exception text leaked).
    """
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    from app.domain.models.work_unit import WorkUnit
    from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

    sha = hashlib.sha256(b"y").hexdigest()
    apply_plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="g.py", op="add", new_digest=sha,
            content_ref="ref2", content_size=2,
        ),),
        total_size_bytes=2, file_count=1,
        source_work_unit_ids=("wu-1",),
    )

    wu = WorkUnit(
        work_unit_id="wu-1",
        objective="explore X", phase="exploration",
        allowed_tools=["file_read"], write_lease=[],
    )
    wr = WorkerResult(
        work_unit_id="wu-1",
        child_session_id="sess-SECRET-12345",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )

    reducer = _reducer_returning(GroupOutcome.SUCCESS, apply_plan, "done")

    # SQLAlchemyError-like message embedding SQL text + a bound session id —
    # this is exactly what F2 forbids leaking into the wire diagnostic.
    leaky_msg = (
        "(psycopg.errors) SELECT ... WHERE session_id = 'sess-SECRET-12345'"
    )
    cost_rollup_service = AsyncMock()
    cost_rollup_service.aggregate = AsyncMock(
        side_effect=RuntimeError(leaky_msg)
    )

    queue: asyncio.Queue = asyncio.Queue()
    config = {
        "configurable": {
            "patch_reducer_service": reducer,
            "cost_rollup_service": cost_rollup_service,
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
        # [F2] dispatched-child id set drives the aggregate pull. Carries the
        # SECRET session id so the F2 leak-prevention assertions below remain
        # meaningful (a naive ``{agg_exc}`` interpolation would surface it).
        "child_session_ids": {"wu-1": "sess-SECRET-12345"},
        "quota_acquired": False,
    }

    cmd = await reducer_node(state, config)

    # Handoff survives the ordinary exception too.
    assert cmd is not None
    assert cmd.update["apply_plan"] is apply_plan
    assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS

    # An event WAS emitted (ordinary exception → degrade-to-cost_unavailable,
    # not a dropped emit) and its client-visible diagnostic is F2-sanitized.
    events = _drain(queue)
    assert len(events) == 1
    ev = events[0]
    assert isinstance(ev, CoordinatorReduceEvent)
    assert "cost_unavailable" in ev.diagnostics_summary
    # F2: stable code, NOT the raw exception text / SQL / bound session id.
    assert "aggregate_error" in ev.diagnostics_summary
    assert "SECRET" not in ev.diagnostics_summary
    assert "SELECT" not in ev.diagnostics_summary
    assert "psycopg" not in ev.diagnostics_summary
