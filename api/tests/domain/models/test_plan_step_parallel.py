"""C2 PR-1 Task 1.4/1.5 — Step + StepDef parallel_work_units schema tests.

Spec ref: docs/superpowers/specs (C2 spec §4.2 r7 P0-1 / P0-2 / r3 P1-1).
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.llm_responses import StepDef
from app.domain.models.plan import Step
from app.domain.models.work_unit import (
    ParallelWorkUnitGroupRequest,
    WorkUnitRequest,
)


class TestStepIdRequired:
    def test_id_must_be_supplied(self):
        with pytest.raises(ValidationError):
            Step(description="x")  # type: ignore[call-arg]

    def test_explicit_id_works(self):
        assert Step(id="step_1", description="x").id == "step_1"


class TestStepParallelWorkUnits:
    def test_default_none(self):
        assert Step(id="s1", description="x").parallel_work_units is None

    def test_assigned(self):
        s = Step(
            id="s1",
            description="x",
            parallel_work_units=ParallelWorkUnitGroupRequest(
                work_units=[
                    WorkUnitRequest(
                        objective="o",
                        phase="exploration",
                        allowed_tools=["file_read"],
                    )
                ]
            ),
        )
        assert s.parallel_work_units is not None
        assert len(s.parallel_work_units.work_units) == 1

    def test_roundtrip(self):
        s = Step(
            id="s1",
            description="x",
            parallel_work_units=ParallelWorkUnitGroupRequest(work_units=[]),
        )
        assert (
            Step.model_validate_json(s.model_dump_json()).parallel_work_units
            is not None
        )


class TestStepDefParallel:
    def test_default_none(self):
        assert StepDef(id="s1", description="x").parallel_work_units is None

    def test_with_parallel(self):
        sd = StepDef(
            id="s1",
            description="x",
            parallel_work_units=ParallelWorkUnitGroupRequest(work_units=[]),
        )
        assert sd.parallel_work_units is not None

    def test_id_optional_for_llm_fallback(self):
        # [r3 P1-1] LLM 可能漏 id；plan builder _build_plan_from_response
        # 在 PR-1 Task 1.9 补 deterministic fallback (_assign_fallback_step_id)
        sd = StepDef(description="x")
        assert sd.id is None
