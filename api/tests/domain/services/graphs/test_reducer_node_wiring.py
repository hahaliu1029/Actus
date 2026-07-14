"""C2 PR-5 Task 5.7 — reducer_node wiring test.

Spec ref: §9.6 (thin wrapper around PatchReducerService).

These tests pin the wiring contract that compositionroot relies on:
- ``reducer_node`` reads ``patch_reducer_service`` from config
- Optional ``parent_sandbox`` is threaded through to reduce()
- Returns ``Command`` with apply_plan / group_outcome /
  step_result_candidate update + goto=END
- Missing ``coordinator_run_id`` surfaces an explicit error (topology
  invariant from dispatch_node)
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.cost_rollup_service import AggregateResult
from app.application.services.patch_reducer_service import (
    ReducerDiagnostics,
    ReducerOutput,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.work_unit import PathLease, WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    WorkerResult,
    reducer_node,
)
from langgraph.constants import END

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _wu(work_unit_id: str = "wu1") -> WorkUnit:
    return WorkUnit(
        work_unit_id=work_unit_id,
        objective="test",
        phase="write",
        allowed_tools=["file_write"],
        write_lease=[PathLease(path="x.py", op="add")],
        expected_result_schema=None,
    )


async def test_reducer_node_threads_reducer_into_command() -> None:
    """The node reads ``patch_reducer_service`` from config and folds
    its ReducerOutput into the Command(update=...) dict."""
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=ReducerDiagnostics(),
    ))
    state = {
        "coordinator_run_id": "r1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
    }
    config = {"configurable": {"patch_reducer_service": reducer}}

    cmd = await reducer_node(state, config)

    reducer.reduce.assert_awaited_once()
    call_kwargs = reducer.reduce.await_args.kwargs
    assert call_kwargs["coordinator_run_id"] == "r1"
    assert call_kwargs["work_unit_ids_expected"] == frozenset({"wu1"})
    assert call_kwargs["worker_results"] == []
    assert call_kwargs["parent_sandbox"] is None
    assert cmd.goto == END
    assert cmd.update["group_outcome"] == GroupOutcome.SUCCESS
    assert cmd.update["step_result_candidate"] == "ok"


async def test_reducer_phase_is_set_before_reducer_side_effect() -> None:
    from app.application.services.coordinator_parent_execution_lease import (
        CoordinatorParentPhase,
    )

    events: list[str] = []
    reducer = AsyncMock()

    async def reduce(**_kwargs):
        events.append("reduce")
        return ReducerOutput(
            apply_plan=None,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate="ok",
            diagnostics=ReducerDiagnostics(),
        )

    reducer.reduce = AsyncMock(side_effect=reduce)
    guard = MagicMock()
    guard.set_phase.side_effect = lambda *_args: events.append("phase")
    state = {
        "step_id": "step-1",
        "coordinator_run_id": "run-1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
    }

    await reducer_node(
        state,
        {"configurable": {
            "patch_reducer_service": reducer,
            "coordinator_wait_guard": guard,
        }},
    )

    assert events[:2] == ["phase", "reduce"]
    guard.set_phase.assert_called_once_with(
        "step-1", "run-1", CoordinatorParentPhase.REDUCING,
    )


async def test_reducer_node_passes_parent_sandbox_when_present() -> None:
    """Optional parent_sandbox in config flows through to reduce()."""
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=ReducerDiagnostics(),
    ))
    sandbox = AsyncMock()
    state = {
        "coordinator_run_id": "r1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "parent_sandbox": sandbox,
    }}

    await reducer_node(state, config)

    call_kwargs = reducer.reduce.await_args.kwargs
    assert call_kwargs["parent_sandbox"] is sandbox


async def test_reducer_node_propagates_diagnostics() -> None:
    """[codex R6 P1] ReducerDiagnostics MUST flow out of reducer_node
    into the subgraph state so main_graph + PR-7 audit-persistence
    can read needs_authorization_details, empty_success_warnings,
    digest_drift_warnings, conflict_paths, missing_ids."""
    diagnostics = ReducerDiagnostics(
        empty_success_warnings=("wu1: manifest run_id mismatch — dropped",),
    )
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=diagnostics,
    ))
    state = {
        "coordinator_run_id": "r1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
    }
    config = {"configurable": {"patch_reducer_service": reducer}}

    cmd = await reducer_node(state, config)

    assert "reducer_diagnostics" in cmd.update
    assert cmd.update["reducer_diagnostics"] is diagnostics


async def test_reducer_node_missing_coordinator_run_id_returns_error() -> None:
    """Topology invariant: dispatch_node MUST fill coordinator_run_id.
    Missing value here means a buggy dispatch_node — surface clearly
    instead of crashing inside reduce() with a confusing TypeError."""
    reducer = AsyncMock()
    state = {
        "coordinator_run_id": None,
        "work_units": [_wu("wu1")],
        "worker_results": [],
    }
    config = {"configurable": {"patch_reducer_service": reducer}}

    cmd = await reducer_node(state, config)

    reducer.reduce.assert_not_called()
    assert cmd.goto == END
    assert cmd.update["apply_plan"] is None
    assert cmd.update["group_outcome"] is None
    assert "missing coordinator_run_id" in cmd.update["step_result_candidate"]


# ── [PR-9b-B Task B8] INV-B1 cost-authority wiring lock ──────────────────────
# reducer_node MUST read ``cost_rollup_service`` from cfg and await
# ``aggregate(coordinator_run_id=..., child_session_ids=[...])`` — the cost
# authority is the durable cost_records ledger, NOT the worker envelope's
# per-child ``cost_summary``. The emit-side behaviour (cost_total propagation,
# cost_unavailable diagnostics on failure/missing-children) is covered in
# ``test_parallel_execution_subgraph_emit.py``; this is the co-located wiring
# lock so a future refactor of reducer_node can't silently drop the pull.


class _CapturingCostRollup:
    """Records the kwargs ``reducer_node`` passes to ``aggregate()``.

    Concrete (not an AsyncMock) so the captured ``child_session_ids`` is a
    real list snapshot rather than a mock call-args proxy.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def aggregate(self, *, coordinator_run_id, child_session_ids):
        self.calls.append({
            "coordinator_run_id": coordinator_run_id,
            "child_session_ids": list(child_session_ids),
        })
        return AggregateResult(cost=CostAggregate(), missing_children=())


async def test_reducer_node_awaits_cost_rollup_service_aggregate() -> None:
    """[INV-B1] reducer_node pulls cost via ``cost_rollup_service.aggregate``
    with ``coordinator_run_id`` + order-preserved ``child_session_ids`` taken
    from ``state['worker_results']``.

    The aggregate() pull lives inside the event-emit branch, so an
    ``event_queue`` MUST be present in cfg for the pull to fire — matching the
    production wiring where the composition root always provides one (INV-A8).
    """
    cost_rollup_service = _CapturingCostRollup()
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=ReducerDiagnostics(),
    ))
    wr_1 = WorkerResult(
        work_unit_id="wu1",
        child_session_id="c1",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),  # envelope cost is IGNORED post-B4
        summary="ok",
    )
    wr_2 = WorkerResult(
        work_unit_id="wu2",
        child_session_id="c2",
        outcome=ResultReadyOutcome.SUCCESS,
        cost_summary=CostAggregate(),
        summary="ok",
    )
    queue: asyncio.Queue = asyncio.Queue()
    state = {
        "coordinator_run_id": "r1",
        "work_units": [_wu("wu1")],
        "worker_results": [wr_1, wr_2],
        # [F2] aggregate id list is sourced from the dispatched child set
        # (work_unit_id -> child_session_id), order-preserved.
        "child_session_ids": {"wu1": "c1", "wu2": "c2"},
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "cost_rollup_service": cost_rollup_service,
        "event_queue": queue,
    }}

    await reducer_node(state, config)

    assert len(cost_rollup_service.calls) == 1, (
        "reducer_node must call cost_rollup_service.aggregate exactly once "
        "(INV-B1 cost-authority pull)"
    )
    call = cost_rollup_service.calls[0]
    assert call["coordinator_run_id"] == "r1"
    # child_session_ids is order-preserved from worker_results.
    assert call["child_session_ids"] == ["c1", "c2"]


async def test_reducer_node_no_aggregate_when_event_queue_absent() -> None:
    """Complement: with no ``event_queue`` in cfg (legacy / non-coordinator
    path) the reducer takes the no-emit branch and never touches
    ``cost_rollup_service`` — so wiring a real service but no queue is a no-op,
    not a crash."""
    cost_rollup_service = _CapturingCostRollup()
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=ReducerDiagnostics(),
    ))
    state = {
        "coordinator_run_id": "r1",
        "work_units": [_wu("wu1")],
        "worker_results": [
            WorkerResult(
                work_unit_id="wu1",
                child_session_id="c1",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
                summary="ok",
            ),
        ],
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "cost_rollup_service": cost_rollup_service,
        # NO event_queue
    }}

    await reducer_node(state, config)

    assert cost_rollup_service.calls == []


# ── C2b rollout WS1b §3.2/§3.3/§3.4: reducer run-level metrics ────────────────


def _reducer_output_success() -> ReducerOutput:
    return ReducerOutput(
        apply_plan=None,
        group_outcome=GroupOutcome.SUCCESS,
        step_result_candidate="ok",
        diagnostics=ReducerDiagnostics(),
    )


async def test_reducer_records_run_terminal_once_on_first_time_path() -> None:
    """[§3.2/§3.3] When cfg carries a coordinator_metrics_recorder + event_queue
    + cost_rollup_service, and state carries dispatch_started_monotonic (the
    first-time-dispatch stamp), reducer_node records the run terminal exactly
    once with authoritative cost + a positive duration."""
    recorder = MagicMock()
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=_reducer_output_success())
    queue: asyncio.Queue = asyncio.Queue()
    state = {
        "coordinator_run_id": "r1",
        "user_id": "u1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
        "child_session_ids": {},
        "dispatch_started_monotonic": time.monotonic() - 1.0,
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "cost_rollup_service": _CapturingCostRollup(),
        "event_queue": queue,
        "coordinator_metrics_recorder": recorder,
    }}

    await reducer_node(state, config)

    recorder.record_run_terminal.assert_called_once()
    kw = recorder.record_run_terminal.call_args.kwargs
    assert kw["coordinator_run_id"] == "r1"
    assert kw["user_id"] == "u1"
    assert kw["cost_usd"] == 0.0  # CostAggregate() default
    assert kw["outcome"] == "success"  # GroupOutcome.SUCCESS.value
    assert kw["cost_authoritative"] is True  # no cost_unavailable diagnostic
    assert kw["duration_s"] is not None and kw["duration_s"] > 0


async def test_reducer_no_record_without_dispatch_stamp() -> None:
    """[§3.4] No dispatch_started_monotonic on state (rehydrate-shaped) → the
    recorder is NEVER called, even with a recorder wired."""
    recorder = MagicMock()
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=_reducer_output_success())
    queue: asyncio.Queue = asyncio.Queue()
    state = {
        "coordinator_run_id": "r1",
        "user_id": "u1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
        "child_session_ids": {},
        # NO dispatch_started_monotonic
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "cost_rollup_service": _CapturingCostRollup(),
        "event_queue": queue,
        "coordinator_metrics_recorder": recorder,
    }}

    await reducer_node(state, config)

    recorder.record_run_terminal.assert_not_called()


async def test_reducer_dead_metric_guard_warns_when_recorder_absent(caplog) -> None:
    """[§3.2 dead-埋点 guard] A first-time run reaches terminal (stamp present)
    but cfg has NO coordinator_metrics_recorder → a WARNING fires so a dropped
    _build_config threading is LOUD, not a silently-flat counter."""
    reducer = AsyncMock()
    reducer.reduce = AsyncMock(return_value=_reducer_output_success())
    queue: asyncio.Queue = asyncio.Queue()
    state = {
        "coordinator_run_id": "r1",
        "user_id": "u1",
        "work_units": [_wu("wu1")],
        "worker_results": [],
        "child_session_ids": {},
        "dispatch_started_monotonic": time.monotonic(),
    }
    config = {"configurable": {
        "patch_reducer_service": reducer,
        "cost_rollup_service": _CapturingCostRollup(),
        "event_queue": queue,
        # NO coordinator_metrics_recorder
    }}

    with caplog.at_level("WARNING"):
        await reducer_node(state, config)

    assert any(
        "coordinator_metrics_recorder" in r.message for r in caplog.records
    ), "expected a dead-metric guard WARNING naming coordinator_metrics_recorder"
