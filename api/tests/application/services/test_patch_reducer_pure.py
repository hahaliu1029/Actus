"""C2 PR-5 Task 5.3 — PatchReducerService pure-function tests.

Spec ref: §9 (whole), specifically §9.3 7-step algorithm + §9.4 templates
+ §9.5 pure-function invariant.

The reducer is the heart of the coordinator merge — pure async function,
no DB / mailbox / sandbox writes (only an optional read of
``parent_sandbox.compute_digest`` for drift detection). These tests pin
each step of the 7-step algorithm and the pure-function contract.
"""
from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from app.application.services.patch_reducer_service import (
    PatchReducerService,
    ReducerOutput,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest
from app.domain.services.graphs.parallel_execution_subgraph import WorkerResult

pytestmark = pytest.mark.anyio


_SHA_A = hashlib.sha256(b"a").hexdigest()
_SHA_B = hashlib.sha256(b"b").hexdigest()
_SHA_C = hashlib.sha256(b"c").hexdigest()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def service() -> PatchReducerService:
    return PatchReducerService()


def _wr(
    wu_id: str,
    outcome: ResultReadyOutcome,
    *,
    patch_manifest: PatchManifest | None = None,
    needs_authorization_details: object | None = None,
) -> WorkerResult:
    return WorkerResult(
        work_unit_id=wu_id,
        child_session_id=f"c_{wu_id}",
        outcome=outcome,
        cost_summary=CostAggregate(),
        patch_manifest=patch_manifest,
        needs_authorization_details=needs_authorization_details,
    )


def _pm(wu_id: str, files: list[FilePatchEntry]) -> PatchManifest:
    return PatchManifest(
        patch_id=f"r1:{wu_id}:p",
        coordinator_run_id="r1",
        work_unit_id=wu_id,
        files=tuple(files),
    )


# ─── Step 1: completeness ───────────────────────────────────────────────────


class TestCompleteness:
    async def test_missing_worker_returns_incomplete(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        assert out.group_outcome == GroupOutcome.INCOMPLETE
        assert out.apply_plan is None
        assert "wu2" in out.diagnostics.missing_ids

    async def test_all_workers_returning_passes_completeness(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        assert out.diagnostics.missing_ids == ()


# ─── Step 2: worker-status priority ──────────────────────────────────────────


class TestPriorityResolution:
    async def test_needs_auth_wins_over_success(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr("wu1", ResultReadyOutcome.SUCCESS),
                _wr("wu2", ResultReadyOutcome.NEEDS_AUTHORIZATION),
            ],
        )
        assert out.group_outcome == GroupOutcome.NEEDS_AUTHORIZATION
        assert out.apply_plan is None

    async def test_needs_auth_propagates_details_to_diagnostics(
        self, service: PatchReducerService,
    ) -> None:
        """[codex R4 P1] When any worker reaches NEEDS_AUTHORIZATION,
        the reducer must preserve its NeedsAuthorizationDetails in the
        diagnostics so PR-6's human-gate router can read the
        structured grievance (which path / tool / reason)."""
        from app.domain.models.needs_authorization_details import (
            NeedsAuthorizationDetails,
        )
        details = NeedsAuthorizationDetails(
            reason="hard_blocked",
            requested_tool="file_write",
            requested_paths=("forbidden.py",),
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.NEEDS_AUTHORIZATION,
                    needs_authorization_details=details,
                ),
            ],
        )
        assert out.group_outcome == GroupOutcome.NEEDS_AUTHORIZATION
        assert out.diagnostics.needs_authorization_details == (details,)

    async def test_needs_auth_omits_workers_without_details(
        self, service: PatchReducerService,
    ) -> None:
        """Workers that returned NEEDS_AUTHORIZATION but lack details
        (programmer error or PR-6 not-yet-wired) are silently skipped
        — diagnostics carries only the populated ones."""
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr("wu1", ResultReadyOutcome.NEEDS_AUTHORIZATION),
            ],
        )
        assert out.group_outcome == GroupOutcome.NEEDS_AUTHORIZATION
        assert out.diagnostics.needs_authorization_details == ()

    async def test_failed_over_cancelled(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr("wu1", ResultReadyOutcome.CANCELLED),
                _wr("wu2", ResultReadyOutcome.FAILED),
            ],
        )
        assert out.group_outcome == GroupOutcome.FAILED

    async def test_timed_out_over_cancelled(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr("wu1", ResultReadyOutcome.TIMED_OUT),
                _wr("wu2", ResultReadyOutcome.CANCELLED),
            ],
        )
        assert out.group_outcome == GroupOutcome.TIMED_OUT

    async def test_all_failed_returns_failed(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.FAILED)],
        )
        assert out.group_outcome == GroupOutcome.FAILED


# ─── Step 3-4: cross-worker conflict ────────────────────────────────────────


class TestConflict:
    async def test_same_path_two_workers_returns_conflict(
        self, service: PatchReducerService,
    ) -> None:
        """Two SUCCESS workers writing the same path → CONFLICT (§9.3
        step 3-4). Reducer must NOT silently elect one — the orchestrator
        needs the explicit signal to route to retry / split."""
        e1 = FilePatchEntry(
            path="shared/x.py", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="r1", content_size=1,
        )
        e2 = FilePatchEntry(
            path="shared/x.py", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_C,
            content_ref="r2", content_size=1,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e1]),
                ),
                _wr(
                    "wu2", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu2", [e2]),
                ),
            ],
        )
        assert out.group_outcome == GroupOutcome.CONFLICT
        assert out.apply_plan is None
        assert "shared/x.py" in out.diagnostics.conflict_paths

    async def test_disjoint_paths_no_conflict(
        self, service: PatchReducerService,
    ) -> None:
        e1 = FilePatchEntry(
            path="a.py", op="add",
            new_digest=_SHA_A, content_ref="r1", content_size=10,
        )
        e2 = FilePatchEntry(
            path="b.py", op="add",
            new_digest=_SHA_B, content_ref="r2", content_size=10,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e1]),
                ),
                _wr(
                    "wu2", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu2", [e2]),
                ),
            ],
        )
        assert out.group_outcome == GroupOutcome.SUCCESS
        assert out.apply_plan is not None
        assert out.apply_plan.file_count == 2


# ─── Step 5-6: digest drift + build plan ─────────────────────────────────────


class TestPlanConstruction:
    async def test_single_worker_success_builds_plan(
        self, service: PatchReducerService,
    ) -> None:
        e = FilePatchEntry(
            path="x.py", op="add",
            new_digest=_SHA_A, content_ref="r", content_size=10,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
        )
        assert out.group_outcome == GroupOutcome.SUCCESS
        assert out.apply_plan is not None
        assert out.apply_plan.file_count == 1
        assert out.apply_plan.total_size_bytes == 10
        assert out.apply_plan.source_work_unit_ids == ("wu1",)

    async def test_entries_sorted_by_path(
        self, service: PatchReducerService,
    ) -> None:
        """§9.3 step 6: PatchApplyPlan.files is sorted by path for
        deterministic apply order — orchestrator + audit + replay all
        depend on this. Two-worker out-of-order input must yield
        path-sorted output."""
        e1 = FilePatchEntry(
            path="zzz.py", op="add",
            new_digest=_SHA_A, content_ref="r1", content_size=1,
        )
        e2 = FilePatchEntry(
            path="aaa.py", op="add",
            new_digest=_SHA_B, content_ref="r2", content_size=1,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1", "wu2"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e1]),
                ),
                _wr(
                    "wu2", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu2", [e2]),
                ),
            ],
        )
        assert out.apply_plan is not None
        assert [f.path for f in out.apply_plan.files] == ["aaa.py", "zzz.py"]

    async def test_all_success_no_manifests_empty_plan(
        self, service: PatchReducerService,
    ) -> None:
        """Exploration-only step (every worker SUCCESS but none produced
        a manifest) → SUCCESS with empty PatchApplyPlan.

        Downstream behavior (NOT this test's responsibility, but for
        clarity): ``_run_parallel_backend`` skips the applier entirely
        when ``apply_plan.file_count == 0`` — no audit row is opened
        at all. This matches the exploration-phase contract: a
        no-files SUCCESS step is a pure compute step that doesn't
        need an audit trail."""
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        assert out.group_outcome == GroupOutcome.SUCCESS
        assert out.apply_plan is not None
        assert out.apply_plan.file_count == 0

    async def test_source_work_unit_ids_only_contributing(
        self, service: PatchReducerService,
    ) -> None:
        """[codex R8 P2] source_work_unit_ids tracks workers that
        contributed entries to the plan — workers without a manifest
        (legal in exploration phase) MUST NOT appear, else plan_hash
        semantics would change based on incidental sibling state."""
        e = FilePatchEntry(
            path="x.py", op="add",
            new_digest=_SHA_A, content_ref="r", content_size=1,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu_b", "wu_a"}),
            worker_results=[
                _wr(
                    "wu_b", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu_b", [e]),
                ),
                _wr("wu_a", ResultReadyOutcome.SUCCESS),
            ],
        )
        assert out.apply_plan is not None
        # Only wu_b contributed an entry; wu_a had no manifest.
        assert out.apply_plan.source_work_unit_ids == ("wu_b",)


# ─── Step 5: optional digest drift check ─────────────────────────────────────


class TestDigestDrift:
    async def test_drift_detected_when_base_mismatches(
        self, service: PatchReducerService,
    ) -> None:
        e = FilePatchEntry(
            path="x.py", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="r", content_size=10,
        )
        parent = AsyncMock()
        parent.compute_digest = AsyncMock(return_value=_SHA_C)
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
            parent_sandbox=parent,
        )
        assert out.group_outcome == GroupOutcome.SUCCESS
        assert out.apply_plan is not None
        assert any(
            "x.py" in w for w in out.diagnostics.digest_drift_warnings
        )

    async def test_no_drift_when_matching(
        self, service: PatchReducerService,
    ) -> None:
        e = FilePatchEntry(
            path="x.py", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="r", content_size=10,
        )
        parent = AsyncMock()
        parent.compute_digest = AsyncMock(return_value=_SHA_A)
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
            parent_sandbox=parent,
        )
        assert out.diagnostics.digest_drift_warnings == ()

    async def test_drift_check_skipped_for_add_op(
        self, service: PatchReducerService,
    ) -> None:
        """``add`` ops have no base_digest, so the drift check is N/A.
        Reducer must NOT call compute_digest for them — calling for add
        would either crash (no base_digest) or surface a misleading
        warning when the file legitimately doesn't yet exist."""
        e = FilePatchEntry(
            path="x.py", op="add",
            new_digest=_SHA_A, content_ref="r", content_size=10,
        )
        parent = AsyncMock()
        parent.compute_digest = AsyncMock(return_value=_SHA_C)
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
            parent_sandbox=parent,
        )
        parent.compute_digest.assert_not_called()
        assert out.diagnostics.digest_drift_warnings == ()

    async def test_no_parent_sandbox_no_drift_check(
        self, service: PatchReducerService,
    ) -> None:
        """When parent_sandbox is None (e.g. unit test of the reducer in
        isolation, or composition root chose to skip), reducer skips
        drift check entirely — proceeds to build plan."""
        e = FilePatchEntry(
            path="x.py", op="modify",
            base_digest=_SHA_A, new_digest=_SHA_B,
            content_ref="r", content_size=1,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
            parent_sandbox=None,
        )
        assert out.group_outcome == GroupOutcome.SUCCESS
        assert out.diagnostics.digest_drift_warnings == ()


# ─── Step 7: result text template ────────────────────────────────────────────


class TestStepResultCandidate:
    async def test_success_template_includes_file_count(
        self, service: PatchReducerService,
    ) -> None:
        e = FilePatchEntry(
            path="x.py", op="add",
            new_digest=_SHA_A, content_ref="r", content_size=1,
        )
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", [e]),
                ),
            ],
        )
        assert "1" in out.step_result_candidate

    async def test_failed_template_used(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.FAILED)],
        )
        assert out.step_result_candidate != ""
        assert out.group_outcome == GroupOutcome.FAILED

    async def test_candidate_truncated_to_2000_chars(
        self, service: PatchReducerService,
    ) -> None:
        """§9.3 step 7: candidate is clamped to 2000 chars so the
        downstream summarizer prompt doesn't blow context budget."""
        files = [
            FilePatchEntry(
                path=f"f{i}.py", op="add",
                new_digest=_SHA_A, content_ref=f"r{i}", content_size=1,
            )
            for i in range(50)
        ]
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[
                _wr(
                    "wu1", ResultReadyOutcome.SUCCESS,
                    patch_manifest=_pm("wu1", files),
                ),
            ],
        )
        assert len(out.step_result_candidate) <= 2000


# ─── §9.5 pure-function invariant ────────────────────────────────────────────


class TestPureFunctionInvariant:
    async def test_reducer_is_deterministic(
        self, service: PatchReducerService,
    ) -> None:
        out1 = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        out2 = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        assert out1.group_outcome == out2.group_outcome
        assert out1.step_result_candidate == out2.step_result_candidate

    async def test_reducer_does_not_mutate_worker_results(
        self, service: PatchReducerService,
    ) -> None:
        """[§9.5] Reducer treats worker_results as read-only. A mutation
        would corrupt the orchestrator's record + the SSE stream the
        frontend reads."""
        wr = _wr("wu1", ResultReadyOutcome.SUCCESS)
        snapshot_outcome = wr.outcome
        snapshot_summary = wr.summary
        await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[wr],
        )
        assert wr.outcome is snapshot_outcome
        assert wr.summary == snapshot_summary

    async def test_reducer_returns_reducer_output_dataclass(
        self, service: PatchReducerService,
    ) -> None:
        out = await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
        )
        assert isinstance(out, ReducerOutput)

    async def test_reducer_does_not_call_parent_sandbox_when_unused(
        self, service: PatchReducerService,
    ) -> None:
        """When parent_sandbox is provided but no modify/delete ops exist
        with a base_digest, reducer must NOT call compute_digest — keeps
        the pure-function side-effect surface minimal."""
        parent = AsyncMock()
        parent.compute_digest = AsyncMock(return_value=_SHA_A)
        await service.reduce(
            coordinator_run_id="r1",
            work_unit_ids_expected=frozenset({"wu1"}),
            worker_results=[_wr("wu1", ResultReadyOutcome.SUCCESS)],
            parent_sandbox=parent,
        )
        parent.compute_digest.assert_not_called()
