"""C2 PR-3 §7.2 — executor_node dispatch branch tests.

Verifies that ``step.parallel_work_units != None`` routes to the coordinator
subgraph (gated by ``ACTUS_C2_COORDINATOR_ENABLED``); legacy react_graph path
remains intact when ``parallel_work_units is None``.

These tests use ``_run_parallel_backend`` directly (the helper added in
main_graph.py) to verify wiring without spinning the full ``build_main_graph``
graph compilation.
"""
from __future__ import annotations

import os
import pytest
from unittest.mock import AsyncMock, MagicMock

from app.domain.models.work_unit import (
    ParallelWorkUnitGroupRequest, WorkUnitRequest,
)
from app.domain.services.coordinator_feature_flag import (
    assert_coordinator_enabled,
)
from app.domain.services.graphs.main_graph import _run_parallel_backend


@pytest.mark.anyio
async def test_run_parallel_backend_invokes_subgraph_with_planner_units() -> None:
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={
        "step_result_candidate": "subgraph-output",
        "group_outcome": MagicMock(value="success"),
    })

    step = MagicMock()
    step.id = "step-abc"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(
            objective="o1", phase="exploration", allowed_tools=["file_read"],
        ),
        WorkUnitRequest(
            objective="o2", phase="exploration", allowed_tools=["file_read"],
        ),
    ])

    state = {
        "session_id": "p1",
        "user_id": "u1",
        "root_session_id": "root1",
    }
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}

    result = await _run_parallel_backend(state, config, step)
    assert result == "subgraph-output"
    subgraph.ainvoke.assert_awaited_once()
    invoked_state = subgraph.ainvoke.await_args.args[0]
    assert invoked_state["step_id"] == "step-abc"
    assert invoked_state["parent_session_id"] == "p1"
    assert invoked_state["root_session_id"] == "root1"
    assert invoked_state["coordinator_run_id"] is None
    assert len(invoked_state["work_unit_requests"]) == 2


@pytest.mark.anyio
async def test_run_parallel_backend_raises_when_subgraph_unwired() -> None:
    step = MagicMock()
    step.id = "step-x"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "p", "user_id": "u", "root_session_id": "r"}
    config = {"configurable": {}}  # NO parallel_execution_subgraph
    with pytest.raises(RuntimeError) as ei:
        await _run_parallel_backend(state, config, step)
    assert "parallel_execution_subgraph" in str(ei.value)


@pytest.mark.anyio
async def test_root_session_id_derived_from_session_id_when_state_missing_key() -> None:
    """[r1 P0-3 fix] MainGraphState carries ``session_id`` not ``root_session_id``.
    Phase 1 max_subagent_depth=1 invariant: parent IS root, so derive.
    Without this fallback, the subgraph would subscribe ``actus:child:None:mailbox``."""
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={"step_result_candidate": "x"})
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "root-abc", "user_id": "u1"}  # NO root_session_id
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    await _run_parallel_backend(state, config, step)
    invoked_state = subgraph.ainvoke.await_args.args[0]
    assert invoked_state["root_session_id"] == "root-abc"
    assert invoked_state["parent_session_id"] == "root-abc"


@pytest.mark.anyio
async def test_root_session_id_explicit_wins_over_session_id() -> None:
    """If MainGraphState ever gains ``root_session_id`` (Phase 2), explicit wins."""
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={"step_result_candidate": "x"})
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {
        "session_id": "p1",
        "root_session_id": "true-root",  # explicit override
        "user_id": "u1",
    }
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    await _run_parallel_backend(state, config, step)
    invoked_state = subgraph.ainvoke.await_args.args[0]
    assert invoked_state["root_session_id"] == "true-root"
    assert invoked_state["parent_session_id"] == "p1"


@pytest.mark.anyio
async def test_run_parallel_backend_returns_empty_string_on_missing_candidate() -> None:
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={})  # no step_result_candidate key
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "p", "user_id": "u", "root_session_id": "r"}
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    result = await _run_parallel_backend(state, config, step)
    assert result == ""


@pytest.mark.anyio
async def test_user_id_falls_back_to_configurable_when_missing_from_state() -> None:
    """[r2 P1-3 fix] planner_react.py:1264-1265 places user_id in configurable
    (not state). _run_parallel_backend must fall back to cfg.get('user_id')."""
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={"step_result_candidate": "ok"})
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "p1"}  # NO user_id in state
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "user_id": "u-from-cfg",
    }}
    await _run_parallel_backend(state, config, step)
    invoked_state = subgraph.ainvoke.await_args.args[0]
    assert invoked_state["user_id"] == "u-from-cfg"


@pytest.mark.anyio
async def test_user_id_state_wins_over_cfg() -> None:
    """When state explicitly provides user_id, it wins over cfg."""
    subgraph = AsyncMock()
    subgraph.ainvoke = AsyncMock(return_value={"step_result_candidate": "ok"})
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "p1", "user_id": "u-from-state"}
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "user_id": "u-from-cfg",
    }}
    await _run_parallel_backend(state, config, step)
    invoked_state = subgraph.ainvoke.await_args.args[0]
    assert invoked_state["user_id"] == "u-from-state"


@pytest.mark.anyio
async def test_raises_when_session_id_missing() -> None:
    """[r2 P1-3 fix] explicit non-None validation for parent_session_id."""
    subgraph = AsyncMock()
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"user_id": "u1"}  # NO session_id
    config = {"configurable": {"parallel_execution_subgraph": subgraph, "user_id": "u1"}}
    with pytest.raises(RuntimeError) as ei:
        await _run_parallel_backend(state, config, step)
    assert "session_id" in str(ei.value)


@pytest.mark.anyio
async def test_raises_when_user_id_missing_from_both_state_and_cfg() -> None:
    """[r2 P1-3 fix] user_id must be in state OR cfg; both missing → raise."""
    subgraph = AsyncMock()
    step = MagicMock()
    step.id = "s"
    step.parallel_work_units = ParallelWorkUnitGroupRequest(work_units=[
        WorkUnitRequest(objective="o", phase="exploration", allowed_tools=[]),
    ])
    state = {"session_id": "p1"}  # NO user_id
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    with pytest.raises(RuntimeError) as ei:
        await _run_parallel_backend(state, config, step)
    assert "user_id" in str(ei.value)


def test_feature_flag_off_blocks_assertion() -> None:
    """ACTUS_C2_COORDINATOR_ENABLED=false → assert_coordinator_enabled raises."""
    orig = os.environ.get("ACTUS_C2_COORDINATOR_ENABLED")
    try:
        os.environ.pop("ACTUS_C2_COORDINATOR_ENABLED", None)
        with pytest.raises(RuntimeError) as ei:
            assert_coordinator_enabled()
        assert "ACTUS_C2_COORDINATOR_ENABLED" in str(ei.value)
    finally:
        if orig is not None:
            os.environ["ACTUS_C2_COORDINATOR_ENABLED"] = orig


def test_feature_flag_on_passes_assertion() -> None:
    orig = os.environ.get("ACTUS_C2_COORDINATOR_ENABLED")
    try:
        os.environ["ACTUS_C2_COORDINATOR_ENABLED"] = "true"
        # Must not raise.
        assert_coordinator_enabled()
    finally:
        if orig is None:
            os.environ.pop("ACTUS_C2_COORDINATOR_ENABLED", None)
        else:
            os.environ["ACTUS_C2_COORDINATOR_ENABLED"] = orig
