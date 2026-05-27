"""C2 PR-5 Task 5.1 — PatchApplyPlan + GroupOutcome + priority resolution.

Spec ref: §9.2 (priority ladder) — multi-outcome → highest-priority winner.

The reducer maps a heterogeneous set of WorkerResult outcomes to a single
``GroupOutcome``. The priority order is the wire-stable shape that the
orchestrator / SSE surface depends on, so this test pins it.
"""
from __future__ import annotations

import hashlib

import pytest

from app.domain.models.patch_apply_plan import (
    GroupOutcome,
    PatchApplyPlan,
    resolve_priority,
)
from app.domain.models.patch_manifest import FilePatchEntry


_SHA_A = hashlib.sha256(b"a").hexdigest()


class TestGroupOutcomeValues:
    def test_all_values(self) -> None:
        for v in (
            "success", "failed", "cancelled", "timed_out",
            "needs_authorization", "conflict", "incomplete", "mixed",
        ):
            GroupOutcome(v)

    def test_str_enum_is_string(self) -> None:
        """GroupOutcome.SUCCESS == "success" — StrEnum semantics enable
        JSON serialization without explicit .value access."""
        assert GroupOutcome.SUCCESS == "success"
        assert isinstance(GroupOutcome.SUCCESS, str)


class TestPriorityResolution:
    def test_needs_auth_wins_over_success(self) -> None:
        assert resolve_priority(
            {GroupOutcome.SUCCESS, GroupOutcome.NEEDS_AUTHORIZATION},
        ) == GroupOutcome.NEEDS_AUTHORIZATION

    def test_conflict_wins_over_failed(self) -> None:
        assert resolve_priority(
            {GroupOutcome.CONFLICT, GroupOutcome.FAILED},
        ) == GroupOutcome.CONFLICT

    def test_failed_wins_over_cancelled(self) -> None:
        assert resolve_priority(
            {GroupOutcome.CANCELLED, GroupOutcome.FAILED},
        ) == GroupOutcome.FAILED

    def test_timed_out_wins_over_cancelled(self) -> None:
        assert resolve_priority(
            {GroupOutcome.TIMED_OUT, GroupOutcome.CANCELLED},
        ) == GroupOutcome.TIMED_OUT

    def test_cancelled_wins_over_incomplete(self) -> None:
        assert resolve_priority(
            {GroupOutcome.CANCELLED, GroupOutcome.INCOMPLETE},
        ) == GroupOutcome.CANCELLED

    def test_incomplete_wins_over_success(self) -> None:
        assert resolve_priority(
            {GroupOutcome.INCOMPLETE, GroupOutcome.SUCCESS},
        ) == GroupOutcome.INCOMPLETE

    def test_single_success(self) -> None:
        assert resolve_priority({GroupOutcome.SUCCESS}) == GroupOutcome.SUCCESS

    def test_empty_set_falls_back_to_success(self) -> None:
        """Pure-function contract: empty set is a degenerate caller input
        but must not crash the reducer. SUCCESS is the safe fallback —
        the upstream completeness check has already rejected the only
        realistic empty case (no workers returned)."""
        assert resolve_priority(set()) == GroupOutcome.SUCCESS

    def test_full_ladder_order(self) -> None:
        """Pin the complete §9.2 ladder: NEEDS_AUTHORIZATION > CONFLICT >
        FAILED > TIMED_OUT > CANCELLED > INCOMPLETE > SUCCESS. Any reorder
        is a wire-level break — touching this test means updating spec +
        orchestrator routing in lockstep."""
        all_outcomes = {
            GroupOutcome.NEEDS_AUTHORIZATION,
            GroupOutcome.CONFLICT,
            GroupOutcome.FAILED,
            GroupOutcome.TIMED_OUT,
            GroupOutcome.CANCELLED,
            GroupOutcome.INCOMPLETE,
            GroupOutcome.SUCCESS,
        }
        assert resolve_priority(all_outcomes) == GroupOutcome.NEEDS_AUTHORIZATION
        all_outcomes.remove(GroupOutcome.NEEDS_AUTHORIZATION)
        assert resolve_priority(all_outcomes) == GroupOutcome.CONFLICT
        all_outcomes.remove(GroupOutcome.CONFLICT)
        assert resolve_priority(all_outcomes) == GroupOutcome.FAILED
        all_outcomes.remove(GroupOutcome.FAILED)
        assert resolve_priority(all_outcomes) == GroupOutcome.TIMED_OUT
        all_outcomes.remove(GroupOutcome.TIMED_OUT)
        assert resolve_priority(all_outcomes) == GroupOutcome.CANCELLED
        all_outcomes.remove(GroupOutcome.CANCELLED)
        assert resolve_priority(all_outcomes) == GroupOutcome.INCOMPLETE

    def test_mixed_label_not_in_ladder(self) -> None:
        """MIXED is a reducer diagnostic label (PR-6 multi-cause routing),
        never participates in priority resolution. A set containing only
        MIXED falls through to the SUCCESS fallback — protects against a
        future bug that silently elects MIXED via the priority path."""
        assert resolve_priority({GroupOutcome.MIXED}) == GroupOutcome.SUCCESS


class TestPatchApplyPlan:
    def test_construction(self) -> None:
        e = FilePatchEntry(
            path="x/y.py", op="add",
            new_digest=_SHA_A, content_ref="r", content_size=10,
        )
        plan = PatchApplyPlan(
            coordinator_run_id="r1", files=(e,),
            total_size_bytes=10, file_count=1,
            source_work_unit_ids=("wu1",),
        )
        assert plan.file_count == 1
        assert plan.coordinator_run_id == "r1"
        assert plan.files[0] is e
        assert plan.source_work_unit_ids == ("wu1",)

    def test_frozen(self) -> None:
        """frozen=True so the orchestrator can pass plan across nodes
        without defensive copies — the reducer returns it once, the
        applier consumes it once, no in-flight mutation."""
        plan = PatchApplyPlan(
            coordinator_run_id="r1", files=(),
            total_size_bytes=0, file_count=0,
            source_work_unit_ids=(),
        )
        with pytest.raises((AttributeError, Exception)):
            plan.file_count = 99  # type: ignore[misc]

    def test_empty_plan_is_legal(self) -> None:
        """A reducer that resolves SUCCESS with zero file entries (all
        workers returned, none had a manifest — degenerate but possible
        during exploration-only steps) must still produce a valid plan."""
        plan = PatchApplyPlan(
            coordinator_run_id="r1", files=(),
            total_size_bytes=0, file_count=0,
            source_work_unit_ids=(),
        )
        assert plan.file_count == 0
        assert plan.files == ()

    def test_multiple_files_preserve_order(self) -> None:
        """Reducer sorts entries by path before constructing the plan
        (§9.3 step 6). PatchApplyPlan itself does NOT re-sort — it
        consumes whatever the reducer hands it."""
        e1 = FilePatchEntry(
            path="a.py", op="add",
            new_digest=_SHA_A, content_ref="r1", content_size=1,
        )
        e2 = FilePatchEntry(
            path="b.py", op="add",
            new_digest=_SHA_A, content_ref="r2", content_size=2,
        )
        plan = PatchApplyPlan(
            coordinator_run_id="r1", files=(e1, e2),
            total_size_bytes=3, file_count=2,
            source_work_unit_ids=("wu1", "wu2"),
        )
        assert [f.path for f in plan.files] == ["a.py", "b.py"]
