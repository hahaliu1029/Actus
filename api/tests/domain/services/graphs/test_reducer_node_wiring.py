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

from unittest.mock import AsyncMock

import pytest

from app.application.services.patch_reducer_service import (
    ReducerDiagnostics,
    ReducerOutput,
)
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.work_unit import PathLease, WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
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
