"""C2 v1 PatchReducerService — pure 7-step reducer (spec §9).

Pure async function. **No side effects** other than an optional read of
``parent_sandbox.compute_digest`` for §9.3 step 5 drift detection. No DB
writes, no mailbox publishes, no sandbox writes.

The 7-step algorithm (§9.3):

1. **Completeness**: any expected work_unit_id missing → INCOMPLETE.
2. **Worker priority**: if any worker is non-SUCCESS, fold all worker
   outcomes through ``resolve_priority`` and return early.
3-4. **Cross-worker conflict**: collect per-path writers across all
   manifests; any path written by ≥2 workers → CONFLICT.
5. **Optional drift check**: for each modify/delete entry with a
   ``base_digest``, if ``parent_sandbox.compute_digest`` returns a
   different digest, record a drift WARNING (not a hard fail — the
   applier preflight will block the actual write).
6. **Build plan**: concatenate all entries, sort by path, build
   ``PatchApplyPlan``.
7. **Step result candidate**: format §9.4 template, clamp to 2000 chars.

**Why pure**: the reducer feeds the orchestrator, the audit writer, AND
the SSE event stream; any mutation here would corrupt downstream views.
The single optional side effect (digest probe) is read-only and explicit
in the signature.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, Optional

from app.domain.models.mailbox_envelope import ResultReadyOutcome
from app.domain.models.patch_apply_plan import (
    GroupOutcome,
    PatchApplyPlan,
    resolve_priority,
)

if TYPE_CHECKING:
    from app.domain.external.parent_sandbox import ParentSandboxPort
    from app.domain.services.graphs.parallel_execution_subgraph import (
        WorkerResult,
    )

logger = logging.getLogger(__name__)


# ── output dataclasses ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReducerDiagnostics:
    """Side-channel diagnostic info — not part of the apply contract.

    The orchestrator surfaces these in the SSE event for the frontend
    (e.g. show which paths conflicted, which workers never returned).

    PR-5 scope: produced by the reducer and returned in
    ``ReducerOutput`` only. PR-7 (crash recovery + rehydrate) will
    widen the ``reducer_node → _run_parallel_backend → applier →
    update_terminal`` pipe to persist these into
    ``coordinator_apply_audit.diagnostics`` so operators can
    post-hoc-debug without replaying envelope logs; PR-5 leaves the
    audit ``diagnostics`` column unwritten by design.

    ``needs_authorization_details`` [codex R4 P1] preserves the per-
    worker ``NeedsAuthorizationDetails`` payloads when at least one
    worker reached ``ResultReadyOutcome.NEEDS_AUTHORIZATION``. The
    reducer's non-SUCCESS branch returns ``apply_plan=None``, so
    without this carrier PR-6's human-gate router would not see the
    structured grievance (which paths / tools / reasons triggered the
    request). Tuple of ``Any`` rather than the concrete type to keep
    domain/application boundary clean — consumers can isinstance-check
    or duck-type as needed.
    """

    missing_ids: tuple[str, ...] = ()
    conflict_paths: tuple[str, ...] = ()
    digest_drift_warnings: tuple[str, ...] = ()
    needs_authorization_details: tuple[Any, ...] = ()
    # [codex R5 P0/P1] Worker-level lineage / manifest hygiene warnings:
    # SUCCESS workers with no manifest (exploration-phase legal, write-
    # phase suspicious) or with a manifest whose coordinator_run_id /
    # work_unit_id didn't match the parent — those manifests are
    # dropped from the apply plan but the diagnostic surfaces here.
    empty_success_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReducerOutput:
    """The reducer → orchestrator handoff value.

    ``apply_plan`` is ``None`` whenever ``group_outcome`` is non-SUCCESS.
    On SUCCESS, ``apply_plan`` is always non-None (possibly empty when
    every worker returned SUCCESS without a manifest, e.g. exploration-
    only step). The orchestrator must branch on ``group_outcome`` and
    only invoke the applier when ``group_outcome == SUCCESS and
    apply_plan.file_count > 0``.
    """

    apply_plan: Optional[PatchApplyPlan]
    group_outcome: GroupOutcome
    step_result_candidate: str
    diagnostics: ReducerDiagnostics


# ── lookup tables ───────────────────────────────────────────────────────────


# Map per-worker terminal outcome → GroupOutcome bucket. CONFLICT /
# INCOMPLETE / MIXED have no per-worker source — they're reducer-derived.
_OUTCOME_TO_GROUP_OUTCOME: dict[ResultReadyOutcome, GroupOutcome] = {
    ResultReadyOutcome.SUCCESS: GroupOutcome.SUCCESS,
    ResultReadyOutcome.FAILED: GroupOutcome.FAILED,
    ResultReadyOutcome.CANCELLED: GroupOutcome.CANCELLED,
    ResultReadyOutcome.TIMED_OUT: GroupOutcome.TIMED_OUT,
    ResultReadyOutcome.NEEDS_AUTHORIZATION: GroupOutcome.NEEDS_AUTHORIZATION,
}


# §9.4 step_result_candidate templates. ``{n}`` = total expected workers,
# ``{k}`` = count of qualifying workers (failed/timed_out/etc.), ``{m}`` =
# total files written on SUCCESS. Templates are Chinese per project style
# (the planner + summarizer both write Chinese).
# [codex R4 P2] All templates Chinese-only — operator-facing text
# stays consistent with project convention; PR-6/PR-8 route via the
# enum/status surface, never on the human text.
_TEMPLATES: dict[GroupOutcome, str] = {
    GroupOutcome.SUCCESS:
        "并行执行完成：{n} 个 worker 完成，合并写入 {m} 个文件。",
    GroupOutcome.FAILED:
        "并行执行失败：{n} 个 worker 中 {k} 个失败。",
    GroupOutcome.TIMED_OUT:
        "并行执行超时：{k}/{n} 个 worker 超时。",
    GroupOutcome.CANCELLED:
        "并行执行被取消：{k}/{n} 个 worker 被取消。",
    GroupOutcome.NEEDS_AUTHORIZATION:
        "并行执行需要授权：{k} 个 worker 提出新权限请求。",
    GroupOutcome.CONFLICT:
        "并行执行冲突：{k} 个文件被多个 worker 同时写入。",
    GroupOutcome.INCOMPLETE:
        "并行执行不完整：{k}/{n} 个 worker 未返回结果。",
    GroupOutcome.MIXED:
        "并行执行结果混合：详见诊断信息。",
}


_MAX_CANDIDATE_LEN = 2000


# ── reducer ─────────────────────────────────────────────────────────────────


class PatchReducerService:
    """Pure 7-step reducer (spec §9.3).

    Constructorless by design — no state, no DI. The optional
    ``parent_sandbox`` is passed per-call so test code can omit it and
    composition root can inject the live adapter.
    """

    async def reduce(
        self,
        *,
        coordinator_run_id: str,
        work_unit_ids_expected: frozenset[str],
        worker_results: list["WorkerResult"],
        parent_sandbox: Optional["ParentSandboxPort"] = None,
    ) -> ReducerOutput:
        n = len(work_unit_ids_expected)

        # ── Step 1: completeness ─────────────────────────────────────────
        received_ids = {wr.work_unit_id for wr in worker_results}
        missing = work_unit_ids_expected - received_ids
        if missing:
            return ReducerOutput(
                apply_plan=None,
                group_outcome=GroupOutcome.INCOMPLETE,
                step_result_candidate=self._fmt(
                    GroupOutcome.INCOMPLETE, n=n, k=len(missing),
                ),
                diagnostics=ReducerDiagnostics(
                    missing_ids=tuple(sorted(missing)),
                ),
            )

        # ── Step 2: worker priority ──────────────────────────────────────
        outcomes_set: set[GroupOutcome] = {
            _OUTCOME_TO_GROUP_OUTCOME[wr.outcome] for wr in worker_results
        }
        if outcomes_set != {GroupOutcome.SUCCESS}:
            group_outcome = resolve_priority(outcomes_set)
            fail_count = sum(
                1 for wr in worker_results
                if wr.outcome != ResultReadyOutcome.SUCCESS
            )
            # [codex R4 P1] Preserve per-worker NeedsAuthorizationDetails
            # so PR-6's human-gate router can read the structured
            # grievance (requested_tool, requested_paths, reason).
            # Reducer's non-SUCCESS branch otherwise returns
            # apply_plan=None with empty diagnostics, losing the very
            # information the orchestrator needs to escalate.
            needs_auth_details = tuple(
                wr.needs_authorization_details
                for wr in worker_results
                if wr.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION
                and wr.needs_authorization_details is not None
            )
            return ReducerOutput(
                apply_plan=None,
                group_outcome=group_outcome,
                step_result_candidate=self._fmt(
                    group_outcome, n=n, k=fail_count, m=0,
                ),
                diagnostics=ReducerDiagnostics(
                    needs_authorization_details=needs_auth_details,
                ),
            )

        # ── Step 3-4: cross-worker conflict ──────────────────────────────
        # Collect (path → list of work_unit_ids that wrote that path) and
        # the flat entry list. ``patch_manifest`` is optional on
        # WorkerResult — a SUCCESS worker with no manifest is a legal
        # exploration-only outcome.
        #
        # [codex R5 P0] Cross-check each manifest's
        # ``coordinator_run_id`` + ``work_unit_id`` against the
        # parent-issued values BEFORE merging entries. A malicious or
        # replayed child could ship a manifest whose paths slip past
        # the per-entry strict validator (canonical relative path)
        # but whose lineage fields don't match the run/work-unit we
        # asked for — that would inject writes from a different run /
        # work-unit into the parent's apply plan. Mismatched manifests
        # are dropped here; the orchestrator sees a worker that
        # appeared to succeed but contributed no files (the
        # ``empty_success_warnings`` carries the signal for PR-7
        # rehydrate + operator debug).
        #
        # NB: this is the minimum-scope lineage cross-check. Full
        # lease-vs-manifest path enforcement (each entry's path must
        # be inside ``WorkUnit.write_lease``) is the Phase 2
        # PermissionEngine landing point — see PatchApplier module
        # docstring "Reducer trusts child manifest paths" deferral.
        path_to_workers: dict[str, list[str]] = defaultdict(list)
        all_entries: list = []
        empty_success_warnings: list[str] = []
        for wr in worker_results:
            pm = wr.patch_manifest
            if pm is None:
                # SUCCESS without manifest — legal for exploration
                # phase, but a write-phase SUCCESS with no manifest is
                # suspicious. PR-5 doesn't know per-worker phase here
                # (the dispatch layer carries phase but the reducer's
                # input has been collapsed); we record a warning so
                # the orchestrator's audit/diagnostics path can flag.
                if wr.outcome == ResultReadyOutcome.SUCCESS:
                    empty_success_warnings.append(
                        f"{wr.work_unit_id}: SUCCESS with no patch_manifest "
                        "(legal for exploration phase; investigate if write)"
                    )
                continue
            # Lineage cross-check — reject manifests whose run_id /
            # work_unit_id don't match the parent-issued ones.
            if pm.coordinator_run_id != coordinator_run_id:
                logger.warning(
                    "reducer: dropping manifest from wu=%s: "
                    "manifest coordinator_run_id=%r != parent %r",
                    wr.work_unit_id, pm.coordinator_run_id,
                    coordinator_run_id,
                )
                empty_success_warnings.append(
                    f"{wr.work_unit_id}: manifest run_id mismatch — dropped"
                )
                continue
            if pm.work_unit_id != wr.work_unit_id:
                logger.warning(
                    "reducer: dropping manifest from wu=%s: "
                    "manifest work_unit_id=%r != worker_result %r",
                    wr.work_unit_id, pm.work_unit_id, wr.work_unit_id,
                )
                empty_success_warnings.append(
                    f"{wr.work_unit_id}: manifest work_unit_id mismatch — dropped"
                )
                continue
            for e in pm.files:
                path_to_workers[e.path].append(wr.work_unit_id)
                all_entries.append(e)
        # NB: ``path_to_workers[path]`` may contain the same work_unit_id
        # multiple times if a single child wrote the same path twice;
        # ``len(set(workers)) > 1`` captures only *cross-worker* conflict,
        # which is the spec-defined CONFLICT signal.
        conflicts = sorted(
            p for p, ws in path_to_workers.items() if len(set(ws)) > 1
        )
        if conflicts:
            return ReducerOutput(
                apply_plan=None,
                group_outcome=GroupOutcome.CONFLICT,
                step_result_candidate=self._fmt(
                    GroupOutcome.CONFLICT, n=n, k=len(conflicts),
                ),
                diagnostics=ReducerDiagnostics(
                    conflict_paths=tuple(conflicts),
                    # [codex R7 P2] Preserve lineage / empty-success
                    # warnings even when conflict aborts plan-build.
                    # Otherwise operators see "CONFLICT" but lose
                    # visibility into earlier dropped manifests.
                    empty_success_warnings=tuple(empty_success_warnings),
                ),
            )

        # ── Step 5: optional digest drift check ──────────────────────────
        warnings: list[str] = []
        if parent_sandbox is not None:
            for e in all_entries:
                # Skip add ops (no base_digest) and entries that for any
                # reason lack base_digest (shouldn't happen for
                # modify/delete given FilePatchEntry validators, but be
                # defensive — wire schemas can drift).
                if e.op in ("modify", "delete") and e.base_digest:
                    current = await parent_sandbox.compute_digest(e.path)
                    # ``None`` from compute_digest = file doesn't exist;
                    # that's a hard-fail signal the applier preflight will
                    # catch as FILE_MISSING — we don't double-warn here.
                    if current is not None and current != e.base_digest:
                        warnings.append(
                            f"{e.path}: digest drift "
                            f"(observed={current[:8]} expected={e.base_digest[:8]})"
                        )

        # ── Step 6: build PatchApplyPlan ─────────────────────────────────
        sorted_entries = sorted(all_entries, key=lambda e: e.path)
        # [codex R8 P2] source_work_unit_ids tracks workers that
        # actually contributed entries to this plan (per
        # ``PatchApplyPlan.source_work_unit_ids`` docstring). Workers
        # whose manifest was dropped (lineage mismatch) OR who had no
        # manifest (exploration-phase success) MUST NOT pollute the
        # lineage — that would corrupt plan_hash semantics for PR-7
        # rehydrate's "is this the same plan?" check. Flatten the
        # ``path_to_workers`` map to derive the contributing set.
        contributing_workers = {
            wu for workers in path_to_workers.values() for wu in workers
        }
        plan = PatchApplyPlan(
            coordinator_run_id=coordinator_run_id,
            files=tuple(sorted_entries),
            total_size_bytes=sum(e.content_size or 0 for e in sorted_entries),
            file_count=len(sorted_entries),
            source_work_unit_ids=tuple(sorted(contributing_workers)),
        )

        # ── Step 7: candidate text ───────────────────────────────────────
        return ReducerOutput(
            apply_plan=plan,
            group_outcome=GroupOutcome.SUCCESS,
            step_result_candidate=self._fmt(
                GroupOutcome.SUCCESS, n=n, m=plan.file_count,
            ),
            diagnostics=ReducerDiagnostics(
                digest_drift_warnings=tuple(warnings),
                empty_success_warnings=tuple(empty_success_warnings),
            ),
        )

    @staticmethod
    def _fmt(
        outcome: GroupOutcome,
        *,
        n: int = 0,
        k: int = 0,
        m: int = 0,
    ) -> str:
        """Format a §9.4 template and clamp to 2000 chars (§9.3 step 7).

        Single-source-of-truth helper so the clamp can't be forgotten on
        a new branch — downstream summarizer + audit table both rely on
        the cap.
        """
        text = _TEMPLATES[outcome].format(n=n, k=k, m=m)
        return text[:_MAX_CANDIDATE_LEN]
