"""[C2 PR-1 Task 1.9] Step.id propagation + deterministic fallback helper.

Verifies that:
- ``_build_plan_from_response`` propagates ``StepDef.id`` into ``Step.id``
  without injecting random UUIDs, and falls back to a deterministic
  ``step_{plan_id}_{NN}`` id when the LLM omits it.
- ``_apply_plan_update`` reuses the existing ``Step.id`` whose description
  matches the planner's new ``StepDef.id`` so replan does not break
  cross-step references.
- ``_assign_fallback_step_id`` is purely deterministic (no I/O, no
  randomness).
"""
from __future__ import annotations

from app.domain.models.llm_responses import PlanResponse, StepDef
from app.domain.models.plan import Plan, Step


class TestStepIdPropagation:
    def test_create_plan_propagates(self):
        from app.domain.services.graphs.main_graph import _build_plan_from_response

        resp = PlanResponse(steps=[
            StepDef(id="step_x_001", description="t1"),
            StepDef(id="step_x_002", description="t2"),
        ])
        plan = _build_plan_from_response(resp)
        assert [s.id for s in plan.steps] == ["step_x_001", "step_x_002"]

    def test_update_plan_propagates(self):
        from app.domain.services.flows.planner_react import _apply_plan_update

        old = Plan(steps=[Step(id="step_a", description="t1")])
        new = PlanResponse(steps=[
            StepDef(id="step_a", description="t1"),
            StepDef(id="step_b", description="t3"),
        ])
        updated = _apply_plan_update(old, new)
        assert [s.id for s in updated.steps] == ["step_a", "step_b"]

    def test_replan_reuses_id(self):
        from app.domain.services.flows.planner_react import _apply_plan_update

        old = Plan(steps=[Step(id="step_v1", description="auth")])
        new = PlanResponse(steps=[StepDef(id="step_v1", description="auth")])
        assert _apply_plan_update(old, new).steps[0].id == "step_v1"

    def test_fallback_deterministic(self):
        from app.domain.services.graphs.main_graph import _assign_fallback_step_id

        assert _assign_fallback_step_id("plan_x", 0) == "step_plan_x_00"
        assert _assign_fallback_step_id("plan_x", 42) == "step_plan_x_42"


class TestStepIdDedupe:
    def test_duplicate_llm_ids_fall_back_to_deterministic(self):
        # LLM sometimes copies the id from its own example ("1", "1", ...).
        # The second occurrence must get a deterministic fallback so
        # updater_node's id-based step lookup keeps each step distinct.
        from app.domain.services.graphs.main_graph import _build_plan_from_response

        resp = PlanResponse(steps=[
            StepDef(id="1", description="first"),
            StepDef(id="1", description="second"),
            StepDef(id="1", description="third"),
        ])
        plan = _build_plan_from_response(resp, plan_id="p1")
        ids = [s.id for s in plan.steps]
        assert ids[0] == "1"
        assert ids[1] == "step_p1_01"
        assert ids[2] == "step_p1_02"
        assert len(set(ids)) == 3

    def test_duplicate_then_unique_keeps_unique(self):
        from app.domain.services.graphs.main_graph import _build_plan_from_response

        resp = PlanResponse(steps=[
            StepDef(id="a", description="x"),
            StepDef(id="a", description="y"),
            StepDef(id="b", description="z"),
        ])
        plan = _build_plan_from_response(resp, plan_id="p2")
        assert [s.id for s in plan.steps] == ["a", "step_p2_01", "b"]
