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
            allowed_paths=["/a", "/b"], work_unit_id="wu-test",
        )
        assert "EXPLORATION" in s
        assert "proposed_write_plan" in s
        assert "/a" in s
        assert "/b" in s

    def test_exploration_warns_no_writes(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="exploration", allowed_paths=[],
            work_unit_id="wu-test",
        )
        # The exploration guidance MUST say "CANNOT write"; otherwise the
        # LLM may attempt file_write and trip ChildScopeGate denials.
        assert "CANNOT write" in s


class TestSectionWritePhase:
    def test_write_keywords_present(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="patch", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert "WRITE" in s
        assert "/x" in s
        assert "ChildScopeGate" in s  # enforcement reminder

    def test_write_guidance_requires_directory_qualified_paths(self) -> None:
        """[single-path contract] The write-phase guidance must steer the child
        to write the authorized directory-qualified paths (no bare filename at
        the workspace root), so a new file lands at e.g. ``workspace/foo.py``
        and never produces a bare manifest path the apply guard rejects."""
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["workspace/x.py"],
            work_unit_id="wu-test",
        )
        assert "directory" in s.lower()

    def test_write_no_paths_still_renders(self) -> None:
        """Defensive: write phase with empty allowed_paths SHOULDN'T happen
        (WorkUnit.write phase requires non-empty lease), but the section
        builder must not crash if it does."""
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=[],
            work_unit_id="wu-test",
        )
        assert s


class TestSectionTreeLeases:
    """[C2-full S2 PR-5] A shell-mode write unit may carry tree leases
    (``write_tree_lease`` → ``allowed_trees``) authorizing ADD-only creation of
    NEW files anywhere under each prefix. The section MUST render those prefixes
    and MUST NOT mislabel a tree-only write unit as read-only exploration."""

    def test_write_tree_only_renders_tree_block_not_read_only(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="generate code", phase="write", allowed_paths=[],
            allowed_trees=["workspace/gen"], work_unit_id="wu-tree",
        )
        # The leased tree prefix is rendered.
        assert "workspace/gen" in s
        # A tree-only write unit is NOT read-only exploration.
        assert "_(none — read-only exploration)_" not in s
        # [codex PR-5 R3 P2] Lock the load-bearing ADD-only wording, not just
        # the substring "ADD": the child must be told it may create NEW files
        # but may NOT modify/delete EXISTING ones (else the tree could be read
        # as full write authority — a mislead into a guaranteed gate bounce).
        assert "ADD-ONLY" in s
        assert "create NEW files" in s
        assert "may NOT modify or delete EXISTING files" in s
        # [codex PR-5 R4 P2] the empty "Authorized paths" placeholder REDIRECTS
        # to the tree block instead of leaving "write to EXACTLY those paths"
        # pointing at nothing.
        assert "authorized directory trees below" in s

    def test_write_mixed_paths_and_trees_renders_both(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["workspace/a.py"],
            allowed_trees=["workspace/gen"], work_unit_id="wu-mixed",
        )
        assert "workspace/a.py" in s
        assert "workspace/gen" in s
        assert "_(none — read-only exploration)_" not in s

    def test_write_paths_only_no_tree_block(self) -> None:
        """An exact-paths-only write unit (no trees) must not grow a tree
        block — output unchanged from the legacy paths-only shape."""
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["workspace/a.py"],
            allowed_trees=[], work_unit_id="wu-paths",
        )
        assert "workspace/a.py" in s
        # No ADD-only directory-tree wording when there are no tree leases.
        assert "directory tree" not in s.lower()

    def test_empty_paths_and_empty_trees_byte_for_byte_unchanged(self) -> None:
        """[flag-OFF / non-shell safety] A non-shell-mode unit has
        ``write_tree_lease == []`` ⇒ ``allowed_trees`` empty ⇒ the rendered
        output MUST be BYTE-FOR-BYTE identical to the legacy call (which had no
        ``allowed_trees`` param at all). This is the flag-OFF guarantee: existing
        children see an unchanged prompt."""
        legacy = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=[],
            work_unit_id="wu-empty",
        )
        with_empty_trees = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=[],
            allowed_trees=[], work_unit_id="wu-empty",
        )
        assert with_empty_trees == legacy
        # And the read-only placeholder still applies when BOTH are empty.
        assert "_(none — read-only exploration)_" in with_empty_trees

    def test_exploration_with_empty_trees_unchanged(self) -> None:
        """Exploration units have neither paths nor trees — adding the
        ``allowed_trees`` default must not alter exploration output."""
        legacy = build_coordinator_work_unit_section(
            objective="x", phase="exploration", allowed_paths=[],
            work_unit_id="wu-explore",
        )
        with_empty_trees = build_coordinator_work_unit_section(
            objective="x", phase="exploration", allowed_paths=[],
            allowed_trees=[], work_unit_id="wu-explore",
        )
        assert with_empty_trees == legacy

    def test_exploration_with_trees_raises(self) -> None:
        # [codex PR-5 R3 P2] Fail-closed helper guard: an exploration unit is
        # read-only and must not carry tree leases (WorkUnit enforces this
        # upstream). A direct-helper misuse mixing exploration + allowed_trees
        # must raise, not emit a contradictory exploration+ADD-only prompt.
        import pytest

        with pytest.raises(ValueError, match="exploration"):
            build_coordinator_work_unit_section(
                objective="x", phase="exploration", allowed_paths=[],
                allowed_trees=["workspace/gen"], work_unit_id="wu-bad",
            )


class TestSectionPhaseValidation:
    """[r1 P2#5] phase narrowed to Literal — fail closed on unknown values
    so a typo can't silently emit a WRITE-tone prompt for a non-write phase."""

    def test_invalid_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="EXPLORATION",  # type: ignore[arg-type]
                allowed_paths=["/x"], work_unit_id="wu-test",
            )

    def test_empty_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="",  # type: ignore[arg-type]
                allowed_paths=["/x"], work_unit_id="wu-test",
            )

    def test_arbitrary_phase_raises(self) -> None:
        with pytest.raises(ValueError, match="phase must be"):
            build_coordinator_work_unit_section(
                objective="x", phase="readonly",  # type: ignore[arg-type]
                allowed_paths=["/x"], work_unit_id="wu-test",
            )


class TestSectionExpectedResultSchema:
    def test_schema_appended_when_supplied(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
            expected_result_schema='{"success": bool, "msg": str}',
        )
        assert "Expected result schema" in s
        assert '"success"' in s

    def test_schema_omitted_when_none(self) -> None:
        s = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
            expected_result_schema=None,
        )
        assert "Expected result schema" not in s


class TestAssemblerStaticHelper:
    def test_returns_string(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/a"],
            work_unit_id="wu-test",
        )
        assert isinstance(out, str)
        assert out

    def test_includes_identity_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="exploration", allowed_paths=[],
            work_unit_id="wu-test",
        )
        assert "Coordinator Step Worker" in out
        assert "restricted" in out.lower()

    def test_includes_behavior_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert "## Behavior" in out
        assert "Stop as soon as the objective is met" in out

    def test_includes_work_unit_block(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="finalize patch", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert "finalize patch" in out
        assert "/x" in out

    def test_uses_canonical_section_separator(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert "\n\n---\n\n" in out

    def test_is_static_method_no_instance_state_needed(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert out


class TestAssemblerStaticHelperPhasesDifferContent:
    def test_exploration_and_write_produce_distinct_prompts(self) -> None:
        explor = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="exploration", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        write = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["/x"],
            work_unit_id="wu-test",
        )
        assert explor != write
        assert "EXPLORATION" in explor
        assert "WRITE" in write


class TestAssemblerForwardsTrees:
    """[C2-full S2 PR-5] The assembler helper forwards ``allowed_trees`` to the
    section so a shell-mode child's tree leases reach the prompt."""

    def test_assembler_forwards_tree_prefix(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=[],
            allowed_trees=["workspace/gen"], work_unit_id="wu-tree",
        )
        assert "workspace/gen" in out
        assert "_(none — read-only exploration)_" not in out

    def test_assembler_empty_trees_byte_for_byte_unchanged(self) -> None:
        """[flag-OFF guarantee] Default/empty ``allowed_trees`` ⇒ assembler
        output identical to the legacy no-trees call."""
        legacy = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["workspace/a.py"],
            work_unit_id="wu-empty",
        )
        with_empty_trees = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["workspace/a.py"],
            allowed_trees=[], work_unit_id="wu-empty",
        )
        assert with_empty_trees == legacy


class TestWorkUnitIdRendered:
    """[F2.1 / spec §5.1.6 / INV-F1.7] work_unit_id is threaded into the child
    prompt verbatim so the PR-F4 routing fake LLM can exact-match on it. It is
    a REQUIRED kwarg (no default) — a default-empty would let the routing fake's
    exact-match be silently defeated."""

    def test_assembler_renders_work_unit_id_verbatim(self) -> None:
        out = PromptAssembler.build_minimal_for_coordinator_child(
            objective="x", phase="write", allowed_paths=["a.py"],
            work_unit_id="wu-abc-123",
        )
        assert "wu-abc-123" in out

    def test_section_renders_work_unit_id(self) -> None:
        out = build_coordinator_work_unit_section(
            objective="x", phase="write", allowed_paths=["a.py"],
            work_unit_id="wu-xyz",
        )
        assert "wu-xyz" in out
