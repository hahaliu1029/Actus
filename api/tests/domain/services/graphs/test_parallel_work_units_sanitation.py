"""[C2b rollout WS0 §3A.4] Flag-off hard sanitation: parallel_work_units in the
parsed LLM response must be cleared at the parse->Step boundary so no flag-off
session (planner OR detection OR updater) can route into the coordinator."""
from __future__ import annotations

import pytest

from app.domain.models.llm_responses import PlanResponse, StepDef
from app.domain.models.work_unit import ParallelWorkUnitGroupRequest, WorkUnitRequest
from app.domain.services.graphs.main_graph import _build_plan_from_response

_FLAG = "ACTUS_C2_COORDINATOR_ENABLED"


def _response_with_pwu() -> PlanResponse:
    pwu = ParallelWorkUnitGroupRequest(
        work_units=[
            WorkUnitRequest(
                objective="analyze foo",
                phase="exploration",
                allowed_tools=["file_read"],
            )
        ]
    )
    return PlanResponse(steps=[StepDef(description="parallel step", parallel_work_units=pwu)])


def test_flag_off_clears_parallel_work_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_FLAG, raising=False)
    plan = _build_plan_from_response(_response_with_pwu())
    assert plan.steps, "expected one step"
    assert all(s.parallel_work_units is None for s in plan.steps)


def test_flag_on_preserves_parallel_work_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_FLAG, "true")
    plan = _build_plan_from_response(_response_with_pwu())
    assert plan.steps[0].parallel_work_units is not None
    assert len(plan.steps[0].parallel_work_units.work_units) == 1


def test_detection_path_shares_sanitized_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    """flows/planner_react.py:1239 calls _build_plan_from_response directly
    (detection planner bypasses planner_node). Pin that the helper — the single
    shared Step-construction point — sanitizes for that path too."""
    monkeypatch.delenv(_FLAG, raising=False)
    plan = _build_plan_from_response(_response_with_pwu(), plan_id="detection")
    assert all(s.parallel_work_units is None for s in plan.steps)


def test_executor_gate_raises_flag_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """[§3A.5] Last-line defense: assert_coordinator_enabled() raises flag-off.
    This is the invariant the dark-launch integration test used to cover via SSE;
    pinned here as a fast local unit test."""
    from app.domain.services.coordinator_feature_flag import assert_coordinator_enabled

    monkeypatch.delenv(_FLAG, raising=False)
    with pytest.raises(RuntimeError, match="ACTUS_C2_COORDINATOR_ENABLED"):
        assert_coordinator_enabled()
