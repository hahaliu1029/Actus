"""C2 PR-4 Task 4.8 — coordinator_work_unit section + assembler helper tests.

Spec ref: §8.7. Pins:
- exploration phase emits exploration guidance + read-only signal
- write phase emits write guidance + lease enforcement reminder
- expected_result_schema is appended when supplied
- assembler staticmethod composes identity + behavior + work_unit blocks
- staticmethod does NOT touch SectionRegistry / TokenEstimator / Telemetry
  (intentional: child prompt is fixed-shape, not budget-arbitrated)
"""
from __future__ import annotations

import pytest

from app.domain.services.prompts.assembler import PromptAssembler
from app.domain.services.prompts.sections.coordinator_work_unit import (
    build_coordinator_work_unit_section,
)


class TestSectionExplorationPhase:
    def test_exploration_keywords_present(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="research auth", phase="exploration",
            allowed_paths=["/a", "/b"],
        )
        assert "EXPLORATION" in s
        assert "proposed_write_plan" in s
        assert "/a" in s
        assert "/b" in s

    def test_exploration_warns_no_writes(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="exploration", allowed_paths=[],
        )
        # The exploration guidance MUST say "CANNOT write"; otherwise the
        # LLM may attempt file_write and trip ChildScopeGate denials.
        assert "CANNOT write" in s


class TestSectionWritePhase:
    def test_write_keywords_present(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="patch", phase="write", allowed_paths=["/x"],
        )
        assert "WRITE" in s
        assert "/x" in s
        assert "ChildScopeGate" in s  # enforcement reminder

    def test_write_no_paths_still_renders(self) -> None:
        """Defensive: write phase with empty allowed_paths SHOULDN'T happen
        (WorkUnit.write phase requires non-empty lease), but the section
        builder must not crash if it does."""
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=[],
        )
        assert s


class TestSectionPhaseValidation:
    """[r1 P2#5] phase narrowed to Literal — fail closed on unknown values
    so a typo can't silently emit a WRITE-tone prompt for a non-write phase."""

    def test_invalid_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="EXPLORATION",  # type: ignore[arg-type]
                allowed_paths=["/x"],
            )

    def test_empty_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="",  # type: ignore[arg-type]
                allowed_paths=["/x"],
            )

    def test_arbitrary_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="readonly",  # type: ignore[arg-type]
                allowed_paths=["/x"],
            )


class TestSectionExpectedResultSchema:
    def test_schema_appended_when_supplied(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["/x"],
            expected_result_schema='{"success": bool, "msg": str}',
        )
        assert "Expected result schema" in s
        assert '"success"' in s

    def test_schema_omitted_when_none(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["/x"],
            expected_result_schema=None,
        )
        assert "Expected result schema" not in s


class TestAssemblerStaticHelper:
    def test_returns_string(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/a"],
        )
        assert isinstance(out, str)
        assert out

    def test_includes_identity_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="exploration", allowed_paths=[],
        )
        assert "Coordinator Step Worker" in out
        assert "restricted" in out.lower()

    def test_includes_behavior_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
        )
        assert "## Behavior" in out
        assert "Stop as soon as the objective is met" in out

    def test_includes_work_unit_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="finalize patch", phase="write", allowed_paths=["/x"],
        )
        assert "finalize patch" in out
        assert "/x" in out

    def test_uses_canonical_section_separator(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
        )
        assert "\n\n---\n\n" in out

    def test_is_static_method_no_instance_state_needed(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
        )
        assert out


class TestAssemblerStaticHelperPhasesDifferContent:
    def test_exploration_and_write_produce_distinct_prompts(self) -> None:
        explor = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="exploration", allowed_paths=["/x"],
        )
        write = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
        )
        assert explor != write
        assert "EXPLORATION" in explor
        assert "WRITE" in write
