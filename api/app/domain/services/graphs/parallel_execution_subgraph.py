"""C2 PR-3 §7.3 + §7.5 — parallel_execution_subgraph.

LangGraph StateGraph compiled with ``checkpointer=False``.

Topology::

    START -> dispatch_node -> Send x N -> worker_node -> reducer_node -> END

PR-3 ships:
  - ``dispatch_node`` -- peek/bump coordinator attempt, build runtime WorkUnits,
    create N child sessions, upload SpawnManifest x N, start N runner tasks,
    publish SPAWN_REQUEST x N, launch orchestrator, fan-out via ``Send`` x N
  - ``worker_node`` -- await terminal envelope per child via
    ``CoordinatorTerminalEnvelopeWaiter``, normalize ``CANCEL_ACK`` final_state
    to ``ResultReadyOutcome``
  - ``reducer_node`` -- C2 PR-5 §9.6 wraps PatchReducerService

Cold code: the subgraph is constructed but is NOT reached at runtime unless
``ACTUS_C2_COORDINATOR_ENABLED=true`` AND ``Step.parallel_work_units != None``.

PR-7 DEFERRALS (concurrent dispatch + preflight ordering):
-----------------------------------------------------------
- **r5 P1-1**: two concurrent ``dispatch_node`` invocations for the same
  ``(parent_session_id, step_id)`` would both peek=None, both bump, ending
  with two separate ``coordinator_run_id``s (``:a1`` and ``:a2``). The
  partial unique index ``ux_sessions_coordinator_wu`` includes
  ``coordinator_run_id``, so different attempts do NOT collide at the DB
  level. The application would double-spawn both attempt cohorts.
  In PR-3 cold code (env flag off) this cannot fire; PR-7 will wrap the
  peek→rehydrate-decision→bump→reservation flow in a Postgres advisory
  lock (e.g. ``pg_advisory_xact_lock(hashtext(parent_session_id||':'||step_id))``)
  or a parent row ``SELECT FOR UPDATE`` held across the dispatch UoW so
  concurrent dispatches serialize on a parent-step key.

- **r5 P2-1**: ``Phase 1 max_subagent_depth=1`` "parent must be a root"
  preflight currently runs INSIDE ``SessionService.create_session_with_parent``
  (after bump_attempt has already committed and after seed artifacts may have
  been uploaded to MinIO). PR-7 will fold the preflight INTO the same UoW
  as the reservation/lock above so the bump is contingent on a valid
  parent ownership + root check, not just a defensive last-line guard.

- **r2 P2-1**: bump-commit-then-child-INSERT crash window leaves an orphaned
  attempt counter increment with no child rows. PR-7 rehydrate will
  distinguish "attempt reserved but no children" from "no attempt reserved"
  and reuse the reserved attempt instead of bumping again.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import operator
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, Optional, TypedDict

if TYPE_CHECKING:
    # [codex R5 P2] Narrow the loose ``Any`` annotation on
    # ``_rehydrate_dispatch(existing: Any)`` and
    # ``_build_pre_results_from_terminal(terminal: dict[str, Any])`` so
    # static analysis catches contract drift between this domain-graph
    # consumer and the application-layer producer
    # (``CoordinatorRehydrateService.detect_existing_run``).
    # TYPE_CHECKING-only: no runtime dependency on application/.
    from app.application.services.coordinator_rehydrate_service import (
        RehydrateResult,
        TerminalEnvelopeRecord,
    )

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from app.domain.models.mailbox_envelope import (
    CostAggregate,
    MailboxEnvelopeType,
    ResultReadyOutcome,
)
from app.domain.models.path_validation import (
    CoordinatorPathContractError,
    tree_contains,
    validate_coordinator_path,
    validate_coordinator_tree_prefix,
)
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.models.work_unit import PathLease, TreeLease, WorkUnit
from app.domain.services.coordinator_shell_mode_flag import (
    is_coordinator_shell_mode_enabled,
)

logger = logging.getLogger(__name__)


class WorkerResult:
    """[C2 PR-4 r4 P1] Carrier for the child's terminal envelope contents.

    Worker_node propagates patch_manifest / needs_authorization_details /
    cost_summary / summary from the validated ResultReadyPayload (PR-4 wire
    schema). PR-5 reducer reads ``patch_manifest`` for apply; PR-6 cost/quota
    aggregates ``cost_summary``; PR-5 reducer routes NEEDS_AUTHORIZATION via
    ``needs_authorization_details``.

    A dataclass would be nicer, but matching the existing dataclass-less PR-3
    minimal shape keeps the diff focused. PR-5 can promote to a frozen
    Pydantic schema during the reducer integration.
    """

    def __init__(
        self,
        *,
        work_unit_id: str,
        child_session_id: str,
        outcome: ResultReadyOutcome,
        cost_summary: Optional[CostAggregate] = None,
        error_summary: Optional[str] = None,
        summary: Optional[str] = None,
        patch_manifest: Optional[Any] = None,
        needs_authorization_details: Optional[Any] = None,
        # [C2-full S2 §3.2 R3-F1] When True, a SUCCESS worker with no resolved
        # patch_manifest is demoted to FAILED before the reducer (write-phase
        # SUCCESS must carry a manifest; a dropped/truncated/unresolvable one
        # must never silently zero-apply). Threaded onto the Send payload so
        # worker_node's demotion decision is local (no per-worker phase lookup).
        manifest_required: bool = False,
    ) -> None:
        self.work_unit_id = work_unit_id
        self.child_session_id = child_session_id
        self.outcome = outcome
        self.cost_summary = cost_summary or CostAggregate()
        self.error_summary = error_summary
        self.summary = summary
        self.patch_manifest = patch_manifest
        self.needs_authorization_details = needs_authorization_details
        self.manifest_required = manifest_required


class ParallelSubgraphState(TypedDict, total=False):
    """Subgraph state. ``coordinator_run_id`` is filled by ``dispatch_node``
    (planner does NOT pre-compute it)."""

    coordinator_run_id: Optional[str]
    step_id: str
    work_unit_requests: list[Any]  # list[WorkUnitRequest]
    work_units: list[WorkUnit]
    parent_session_id: str
    user_id: str
    root_session_id: str
    child_session_ids: dict[str, str]
    orchestrator_task: Optional[Any]  # asyncio.Task
    worker_results: Annotated[list[WorkerResult], operator.add]
    apply_plan: Optional[Any]       # PR-5
    group_outcome: Optional[Any]    # PR-5
    step_result_candidate: Optional[str]
    # [codex R6 P1] ReducerDiagnostics carrier — reducer_node writes
    # it so main_graph / PR-7 audit-diagnostics persistence have
    # something to read. ``Any`` keeps the application-layer type
    # (ReducerDiagnostics) out of the domain graph import surface.
    reducer_diagnostics: Optional[Any]  # PR-5
    # [codex R2 P1-4] True iff ``_first_time_dispatch`` successfully
    # acquired a per-user coordinator concurrency slot via
    # ``probe_quota.acquire_coordinator_concurrency`` and that slot is
    # still held when entering reducer_node. The rehydrate dispatch
    # path (``_rehydrate_dispatch``) does NOT acquire — the slot is
    # owned by the prior dispatch incarnation — so its Command leaves
    # this False (default). reducer_node gates its quota release on
    # this flag so the rehydrate path does not DECR below zero.
    quota_acquired: bool
    # [C2b rollout WS1b §3.3] monotonic clock captured at the START of
    # _first_time_dispatch (before any I/O). Carried to the reducer to derive
    # run duration AND to gate run-level metrics so a crash+rehydrate (which
    # routes _rehydrate_dispatch, never setting this) cannot double-count
    # run_cost_usd. total=False → absent on the rehydrate path (§3.4).
    dispatch_started_monotonic: float


# ── helpers ──────────────────────────────────────────────────────────────────


def _append_diag(existing: str, addition: str) -> str:
    """[PR-9b-B Task B4] Append a diagnostic clause to a diagnostics_summary
    string in a stable, observable way.

    The reducer's ``diagnostics_summary`` is a free-form string field on
    ``CoordinatorReduceEvent`` consumed by observability tooling. When the
    cost-pull path (INV-B1) reports ``cost_unavailable``, we want the
    addition to be visible without clobbering the prior summary (e.g.
    ``"ok"`` from a clean reducer run).
    """
    if not existing:
        return addition
    return f"{existing}; {addition}"


def _build_work_units_from_requests(
    work_unit_requests: list[Any],
    step_id_hash16: str,
    attempt_ix: int,
) -> list[WorkUnit]:
    """Convert planner WorkUnitRequest list to runtime WorkUnit list.

    [single-path contract — lease boundary] Each planner-proposed path is
    validated AND canonicalized via ``validate_coordinator_path`` so a bare /
    workspace-root path is rejected with a ``CoordinatorPathContractError``
    HERE — before any child spawns — instead of producing a bare manifest path
    that fails late at apply (§14 live-repro). The lease stores the CANONICAL
    directory-qualified workspace-relative form (e.g. ``/home/ubuntu/sub/b.py``
    and ``./sub/b.py`` both -> ``sub/b.py``), i.e. the ONE form the strict
    manifest validator + ChildScopeGate exact-match + the child prompt all
    agree on, so a lease can never be accepted in a shape that strands the run
    mid-flight at strict manifest validation.
    """
    units: list[WorkUnit] = []
    for i, req in enumerate(work_unit_requests):
        tree_leases = [
            TreeLease(
                prefix=validate_coordinator_tree_prefix(t.prefix),
                ops=frozenset(t.ops),
            )
            for t in getattr(req, "proposed_trees", []) or []
        ]
        # [S2 §3.3/§3.5] shell_mode is the OR of TWO sufficient signals:
        #   (1) the request's OWN positive shell_mode (§3.5 — a unit may request
        #       shell mode with ONLY exact file leases, NO tree leases), AND
        #   (2) the "tree lease IMPLIES shell_mode" rule (§3.3 — a non-empty tree
        #       lease is sufficient on its own).
        # Reading ONLY bool(tree_leases) would (a) strand a legit shell-mode unit
        # that has only exact file leases (shell_mode stays False forever) and
        # (b) silently drop the request's positive shell_mode signal. A
        # path-only / exploration unit with shell_mode unset stays typed-only.
        shell_mode = bool(getattr(req, "shell_mode", False)) or bool(tree_leases)
        units.append(
            WorkUnit(
                work_unit_id=f"{step_id_hash16}.a{attempt_ix}.{i}",
                objective=req.objective,
                phase=req.phase,
                allowed_tools=list(req.allowed_tools),
                write_lease=[
                    PathLease(path=validate_coordinator_path(p.path), op=p.op)
                    for p in req.proposed_paths
                ],
                write_tree_lease=tree_leases,
                shell_mode=shell_mode,
                expected_result_schema=req.expected_result_schema,
            )
        )
    return units


def _reject_cross_unit_tree_overlap(units: list[WorkUnit]) -> None:
    """[S2 §3.3 F24] Reject any cross-unit lease overlap: one unit's tree prefix
    must not contain another unit's file lease or tree prefix (component-aware
    via ``tree_contains``). Same-unit nesting (a unit's own file under its own
    tree) is fine — only CROSS-unit overlap strands the parallel apply plan
    (two children both authorized to create under the same dir). Raises
    ``CoordinatorPathContractError`` BEFORE any child spawns."""
    for i, unit_a in enumerate(units):
        for tl in unit_a.write_tree_lease:
            for j, unit_b in enumerate(units):
                if i == j:
                    continue
                for pl in unit_b.write_lease:
                    if tree_contains(tl.prefix, pl.path):
                        raise CoordinatorPathContractError(
                            f"tree/file lease overlap: unit {unit_a.work_unit_id} "
                            f"leases tree {tl.prefix!r} which contains unit "
                            f"{unit_b.work_unit_id}'s file lease {pl.path!r}"
                        )
                for tl_b in unit_b.write_tree_lease:
                    # [S2 §3.3 F24] tree/tree overlap is symmetric AND includes
                    # the IDENTICAL-prefix case. ``tree_contains`` returns False
                    # for EQUAL paths (a prefix is not "inside" itself), so two
                    # units each leasing ``workspace`` would BOTH pass a bare
                    # ``tree_contains`` check — the exact double-grant §3.3/§159
                    # dispatch-time rejection must catch. Compare both directions
                    # plus equality.
                    if (
                        tl.prefix == tl_b.prefix
                        or tree_contains(tl.prefix, tl_b.prefix)
                        or tree_contains(tl_b.prefix, tl.prefix)
                    ):
                        raise CoordinatorPathContractError(
                            f"tree/tree lease overlap: unit {unit_a.work_unit_id} "
                            f"leases tree {tl.prefix!r} which overlaps unit "
                            f"{unit_b.work_unit_id}'s tree {tl_b.prefix!r}"
                        )


def _coerce_units_typed_only_if_flag_off(units: list[WorkUnit]) -> list[WorkUnit]:
    """[S2 §3.6 F27] FLAG-OFF ACTIVE FAIL-CLOSED. While the master shell-mode
    flag is OFF, the spec wants any unit carrying ``shell_mode=True`` / a
    non-empty ``write_tree_lease`` "hard-rejected (or coerced typed-only)". The
    distinction turns on whether a TYPED write survives the strip:

    - **MIXED unit** (has a ``write_lease`` AND a tree lease / shell_mode): the
      typed write is still authorized, so the tree lease + ``shell_mode`` are
      stripped and the unit runs typed-only (``shell_mode=False`` +
      ``write_tree_lease=[]``). The path lease survives unchanged.
    - **TREE-ONLY unit** (no ``write_lease``, only a tree lease / shell_mode):
      there is NO typed write to fall back to. "Coercing" it to exploration
      would still SPAWN a child for a stale / hand-crafted shell payload, which
      is precisely the attack F27 closes. So it is **HARD REJECTED** —
      ``CoordinatorPathContractError`` raised BEFORE any child spawns.

    Flag ON -> identity (no coercion).
    """
    if is_coordinator_shell_mode_enabled():
        return units
    coerced: list[WorkUnit] = []
    for u in units:
        if not u.shell_mode and not u.write_tree_lease:
            coerced.append(u)
            continue
        if u.write_lease:
            # MIXED: the typed write survives; strip the dormant shell signals.
            coerced.append(
                u.model_copy(update={"shell_mode": False, "write_tree_lease": []})
            )
        else:
            # TREE-ONLY: nothing typed survives the strip -> hard-reject, do NOT
            # spawn a child for a flag-off shell payload (F27 active fail-closed).
            raise CoordinatorPathContractError(
                f"shell-mode unit {u.work_unit_id} carries a tree lease / "
                f"shell_mode but no file lease while "
                f"ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED is OFF; refusing to "
                f"dispatch a shell-capable child (F27 active fail-closed)."
            )
    return coerced


def _log_orchestrator_task_done(task: asyncio.Task) -> None:
    """r6 P1-1 — done-callback for the orchestrator background task.

    Surfaces unhandled exceptions (e.g., r5 P1-2 all-CANCEL-publish-failed
    raise from ``CoordinatorRunOrchestrator.run``) so they don't get silently
    swallowed when no one awaits the task.
    """
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(
            "Coordinator orchestrator task died with unhandled exception: %r",
            exc, exc_info=exc,
        )


def _serialize_spawn_manifest(wu: WorkUnit) -> bytes:
    """Minimal JSON serialization of WorkUnit for MinIO manifest.

    [S2 §3.3/§3.5 round-trip B5] write_tree_lease + shell_mode are serialized
    here and decoded symmetrically in the starter. TreeLease.ops is a frozenset
    -> sorted list for stable JSON. ``wu.shell_mode`` is the value built in
    ``_build_work_units_from_requests`` as ``req.shell_mode or
    bool(write_tree_lease)`` (and preserved through the Step-4 enrichment
    rebuild), so the request's positive shell_mode reaches the child manifest.
    """
    return json.dumps({
        "work_unit_id": wu.work_unit_id,
        "objective": wu.objective,
        "phase": wu.phase,
        "allowed_tools": list(wu.allowed_tools),
        "write_lease": [lease.model_dump() for lease in wu.write_lease],
        "write_tree_lease": [
            {"prefix": tl.prefix, "ops": sorted(tl.ops)}
            for tl in wu.write_tree_lease
        ],
        "shell_mode": wu.shell_mode,
        "expected_result_schema": wu.expected_result_schema,
    }, sort_keys=True).encode("utf-8")


# ── dispatch_node ────────────────────────────────────────────────────────────


async def dispatch_node(state: ParallelSubgraphState, config: RunnableConfig) -> Command:
    """C2 PR-3 §7.5 P0-3 — ordered dispatch with crash-recovery PEEK before BUMP.

    1. PEEK ``coordinator_attempts[step_id]`` (read-only)
    2. If peek returned ``>= 1`` -> try ``rehydrate_service.detect_existing_run``
       at the peeked attempt; if found, reuse (crash recovery, no bump).
       Otherwise fall through to fresh attempt.
    3. Otherwise BUMP atomically to get a fresh attempt_ix and run first-time
       dispatch.

    Invariant: NEVER bump on crash retry -- that would orphan prior child rows.
    """
    cfg = config["configurable"]
    # C2 PR-3 §7.5 P0-3 [r1 P0-2 fix] — peek/bump go through SessionService so
    # the JSONB UPDATE commits via UoW BEFORE child sessions are created.
    # Without this, the bump and child-create live in disjoint UoWs and a crash
    # between them violates the PEEK-before-BUMP crash recovery contract.
    session_service = cfg["session_service"]
    rehydrate_service = cfg["rehydrate_service"]
    parent_session_id = state["parent_session_id"]
    step_id = state["step_id"]
    step_id_hash16 = hashlib.sha256(step_id.encode("utf-8")).hexdigest()[:16]

    current_attempt_ix = await session_service.peek_coordinator_attempt(
        session_id=parent_session_id, step_id=step_id,
    )

    if current_attempt_ix is not None and current_attempt_ix >= 1:
        candidate_run_id = (
            f"{parent_session_id}:{step_id_hash16}:a{current_attempt_ix}"
        )
        rehydrate_emit = None
        _rq = cfg.get("event_queue")
        if _rq is not None:
            async def _rehydrate_emit(event: object) -> None:
                _rq.put_nowait(event)  # put_nowait — sync, cancellation-safe (matches _orch_emit_into_queue)
            rehydrate_emit = _rehydrate_emit
        existing = await rehydrate_service.detect_existing_run(
            coordinator_run_id=candidate_run_id,
            parent_session_id=parent_session_id,
            emit_event=rehydrate_emit,
        )
        if existing is not None:
            work_units = _build_work_units_from_requests(
                state["work_unit_requests"], step_id_hash16, current_attempt_ix,
            )
            # [S2 §3.6 F27] flag-off active fail-closed: coerce shell-mode units
            # to typed-only while the master flag is OFF (PR-3 state). MUST run
            # BEFORE overlap rejection [codex PR-3 R1 P1]: under flag OFF the
            # tree leases are stripped, so two units that coerce to DISJOINT
            # typed-only leases must not be spuriously rejected for a (moot)
            # tree overlap that no longer exists post-coercion.
            work_units = _coerce_units_typed_only_if_flag_off(work_units)
            # [S2 §3.3 F24] reject cross-unit lease overlap BEFORE spawning, on
            # the POST-coercion units: flag ON ⇒ coercion is identity ⇒ tree
            # overlaps still fail loud; flag OFF ⇒ only surviving typed leases
            # are overlap-checked.
            _reject_cross_unit_tree_overlap(work_units)
            return await _rehydrate_dispatch(
                state, config, existing, candidate_run_id, work_units,
            )

    # r2 P2-1 known limitation: if a crash hits BETWEEN bump-commit and the
    # first child-row INSERT (in session_service's own UoW), the next retry
    # will see ``peek == attempt_ix`` but ``rehydrate.detect_existing_run``
    # returns None (no children exist for that attempt). Per the rehydrate
    # contract, we then BUMP AGAIN to a fresh attempt_ix, leaving the prior
    # attempt as a counter-inflation orphan (no child rows, no data loss).
    # PR-7's full ``CoordinatorRunRepository`` rehydrate is the durable fix:
    # it will distinguish "attempt reserved but no children" from
    # "no attempt reserved" and reuse the reserved attempt instead of
    # bumping. PR-3 accepts counter inflation as a v1 limitation.
    attempt_ix = await session_service.bump_coordinator_attempt(
        session_id=parent_session_id, step_id=step_id,
    )
    coordinator_run_id = (
        f"{parent_session_id}:{step_id_hash16}:a{attempt_ix}"
    )
    work_units = _build_work_units_from_requests(
        state["work_unit_requests"], step_id_hash16, attempt_ix,
    )
    # [S2 §3.6 F27] flag-off active fail-closed: coerce shell-mode units to
    # typed-only while the master flag is OFF (PR-3 state). MUST run BEFORE
    # overlap rejection [codex PR-3 R1 P1]: under flag OFF the tree leases are
    # stripped, so two units that coerce to DISJOINT typed-only leases must not
    # be spuriously rejected for a (moot) tree overlap that no longer exists
    # post-coercion.
    work_units = _coerce_units_typed_only_if_flag_off(work_units)
    # [S2 §3.3 F24] reject cross-unit lease overlap BEFORE spawning, on the
    # POST-coercion units: flag ON ⇒ coercion is identity ⇒ tree overlaps still
    # fail loud; flag OFF ⇒ only surviving typed leases are overlap-checked.
    _reject_cross_unit_tree_overlap(work_units)
    return await _first_time_dispatch(state, config, coordinator_run_id, work_units)


async def _first_time_dispatch(
    state: ParallelSubgraphState,
    config: dict,
    coordinator_run_id: str,
    work_units: list[WorkUnit],
) -> Command:
    # [C2b rollout WS1b §3.3/§3.4] Stamp the dispatch start BEFORE any I/O
    # (session creation / manifest upload / child start). Carried in the
    # Command(update) below; the reducer derives duration + gates run-level
    # metrics on its presence.
    dispatch_started_monotonic = time.monotonic()
    cfg = config["configurable"]
    session_service = cfg["session_service"]
    runner_starter = cfg["child_runner_starter"]
    publisher = cfg["mailbox_publisher"]
    artifact = cfg["artifact_storage"]
    parent_sandbox = cfg["parent_sandbox"]
    orchestrator_factory = cfg["orchestrator_factory"]
    cancel_event = cfg["cancel_event"]

    parent_session_id = state["parent_session_id"]
    user_id = state["user_id"]
    root_session_id = state["root_session_id"]

    # ── C2 PR-6 §14.3 #1 — dispatch preflight caps ───────────────────────────
    # Fail-fast BEFORE any side effects (enrich leases / create sessions /
    # upload manifests / start runners / publish SPAWN_REQUEST). Order:
    #   1. work_unit_count cap (cheap in-memory check)
    #   2. user concurrency (atomic Redis acquire)
    #   3. descendants cap (DB count + projected delta)
    #
    # All three deps are looked up via ``cfg.get(...)`` so that legacy tests
    # / partial DI wiring don't regress while PR-6 lands. The composition
    # root (interfaces/service_dependencies.py) is responsible for wiring
    # these keys in production; absent keys cause the corresponding cap to
    # be skipped (backward compatible).
    coordinator_limits = cfg.get("coordinator_limits")
    probe_quota = cfg.get("probe_quota")
    # ``session_repository`` is the explicit DI key for descendants count.
    # We deliberately do NOT fall back to ``session_service`` here: the
    # production ``SessionService.count_descendants`` is not a public API
    # (the count lives behind ``uow.session.count_descendants``), and any
    # bare ``AsyncMock`` would silently auto-create the attribute and
    # spuriously fire the cap. Tests that exercise the descendants branch
    # MUST wire ``session_repository`` explicitly.
    session_repo = cfg.get("session_repository")

    # 1) work_unit count cap — cheap predicate, raise immediately.
    if coordinator_limits is not None:
        if len(work_units) > coordinator_limits.max_work_units_per_run:
            raise ValueError(
                f"coordinator preflight: work_units={len(work_units)} exceeds "
                f"cap {coordinator_limits.max_work_units_per_run}"
            )

    # 2) Per-user concurrency — atomic Redis acquire. Pair with rollback on
    # the descendants-cap rejection branch below to avoid quota leak.
    concurrency_acquired = False
    if probe_quota is not None and coordinator_limits is not None:
        concurrency_acquired = await probe_quota.acquire_coordinator_concurrency(
            user_id=user_id,
            cap=coordinator_limits.max_concurrent_coordinator_runs_per_user,
        )
        if not concurrency_acquired:
            raise ValueError(
                f"coordinator preflight: user {user_id} concurrency cap "
                f"{coordinator_limits.max_concurrent_coordinator_runs_per_user} "
                f"reached"
            )

    # TODO(PR-7+ daily cost cap): §14.3 #2 per-user daily cost cap -- call
    # probe_quota.acquire_coordinator_daily_cost(user_id=user_id,
    # cost_usd=coordinator_limits.max_total_token_cost_usd_per_run,
    # cap_usd=coordinator_limits.max_coordinator_token_cost_usd_per_user_per_day)
    # as a projected worst-case dispatch-time gate. On reject, raise +
    # release the concurrency slot acquired above (same rollback pattern
    # used by the descendants cap branch and the codex P1-3 try/except
    # below). Skipped in PR-6 because the call site needs more design
    # discussion re: actual vs projected cost (preflight projects worst-
    # case, or post-run records actual?).

    # [codex R2 P1-3] DISPATCH-BODY ROLLBACK GUARD
    # ----------------------------------------------------------------
    # Everything below this point can raise (digest computation,
    # MinIO uploads, session creates, manifest uploads, subscribe,
    # runner starts, SPAWN_REQUEST publishes, orchestrator launch).
    # Without this try/except, an exception AFTER the acquire above
    # but BEFORE reducer_node runs would leak the concurrency slot
    # forever (reducer_node is the sole release path on the happy
    # path; if dispatch raises the subgraph state is discarded and
    # reducer is never reached). Release on any exception so a
    # downstream failure does not pin the user's slot until TTL.
    #
    # [codex R8 P2-1] We also track pre-created waiter consumer
    # groups in ``created_waiter_groups`` so the except-clause can
    # destroy them on rollback. Initialized inside the try block —
    # Python scope-wise it is accessible from the except clause
    # because both share the enclosing function frame.
    created_waiter_groups: list[tuple[str, str]] = []
    # [C2b budget §3-9 R3#1] Children THIS dispatch successfully started —
    # appended ONLY after each runner_starter.start returns, so when the Nth
    # start raises the list holds exactly the N-1 started ones. The except
    # clause stops them via starter.request_stop_started (scoped: the starter
    # is a lifespan singleton shared by concurrent runs — R4#2).
    started_child_session_ids: list[str] = []
    # [finish-core §5.4 G4-min] Init BEFORE the try so the except clause can
    # reference both unconditionally. Without this, a failure before they are
    # assigned would raise UnboundLocalError and MASK the original dispatch
    # error. ``orchestrator_task`` is also returned on the success path (state
    # key), so None is the correct pre-launch sentinel for the INV-F4.3
    # ownership check.
    orchestrator_task = None
    orchestrator_group_created: Optional[tuple[str, str]] = None
    try:
        # 3) Descendants cap — DB count + projected delta. If concurrency was
        # acquired in step 2 but this rejects, release the concurrency slot to
        # avoid a permanent quota leak.
        if session_repo is not None:
            from app.domain.services.subagent_limits import MAX_DESCENDANTS_PER_ROOT
            existing = await session_repo.count_descendants(
                root_session_id,
                user_id=user_id,
                cap=MAX_DESCENDANTS_PER_ROOT,
            )
            if existing + len(work_units) > MAX_DESCENDANTS_PER_ROOT:
                # NB: release happens in the outer except below — keep this
                # branch a plain ValueError so the surrounding try/except
                # owns the single release path.
                raise ValueError(
                    f"coordinator preflight: descendants cap "
                    f"{MAX_DESCENDANTS_PER_ROOT} would be exceeded "
                    f"(existing={existing}, requested={len(work_units)})"
                )

        # Step 4 -- enrich PathLeases with base_digest + seed_content_ref for
        # modify/delete ops (add ops keep both None per WorkUnit invariants).
        enriched_units: list[WorkUnit] = []
        for wu in work_units:
            new_leases: list[PathLease] = []
            for lease in wu.write_lease:
                base_digest = lease.base_digest
                seed_ref = lease.seed_content_ref
                if lease.op in ("modify", "delete") and base_digest is None:
                    base_digest = await parent_sandbox.compute_digest(lease.path)
                    if base_digest is None:
                        raise ValueError(
                            f"lease {lease.path!r} (op={lease.op}) not found in parent sandbox"
                        )
                if lease.op in ("modify", "delete") and seed_ref is None:
                    content = await parent_sandbox.read_file(lease.path)
                    seed_ref = await artifact.put_content_addressed_bytes(
                        prefix=f"coordinator/{coordinator_run_id}/{wu.work_unit_id}/seed/",
                        content=content,
                    )
                new_leases.append(PathLease(
                    path=lease.path, op=lease.op,
                    base_digest=base_digest, seed_content_ref=seed_ref,
                ))
            enriched_units.append(WorkUnit(
                work_unit_id=wu.work_unit_id,
                objective=wu.objective,
                phase=wu.phase,
                allowed_tools=wu.allowed_tools,
                write_lease=new_leases,
                # [S2 §3.3/§3.5 — P0-1] preserve the shell-mode signals through
                # enrichment; enrichment only fills write_lease digests, it must
                # not strip the tree lease / shell_mode (else they never reach
                # _serialize_spawn_manifest -> the child -> PR-4/PR-5). ``wu``
                # here is the pre-enrichment unit produced by
                # ``_build_work_units_from_requests``, so ``wu.shell_mode``
                # already carries ``req.shell_mode or bool(write_tree_lease)`` —
                # the request's positive shell_mode flows through unchanged.
                write_tree_lease=wu.write_tree_lease,
                shell_mode=wu.shell_mode,
                expected_result_schema=wu.expected_result_schema,
            ))

        # Step 5 -- create N child sessions (live returns Session domain object).
        #
        # TODO(orphan-cleanup PR-7+): This loop is NOT atomic — each
        # ``session_service.create_session_with_parent`` runs in its own
        # UoW. If session N succeeds and session N+1 fails (DB
        # constraint, FK violation, transient asyncpg error), the first
        # N rows are committed as ORPHANS and the outer except below
        # only releases the concurrency slot — it does NOT cascade-
        # delete the orphan ``sessions`` rows. The mailbox supervisor's
        # orphan reaper (``mailbox_supervisor.py`` orphan-reap path)
        # only sees children with ``_last_seen_mono`` entries, which the
        # orphan rows lack because they never received their
        # SPAWN_REQUEST.
        #
        # Pre-existing PR-3 issue, NOT introduced by PR-6. PR-6's
        # preflight caps (descendants / daily cost / concurrency) sit
        # in front of this loop but the loop itself is unchanged.
        #
        # Real fix candidates (post-PR-6):
        #   (a) Batch the N child-session creates + manifest uploads
        #       into a single UoW with a per-step cleanup callback so
        #       partial failures roll the whole batch.
        #   (b) Periodic GC sweep for sessions stuck in PENDING status
        #       > N seconds with no SpawnManifest row reference.
        #   (c) Pre-allocate child_session_id deterministically from
        #       ``(coordinator_run_id, work_unit_id)`` so a retry can
        #       reconcile by primary-key conflict instead of producing
        #       new orphans.
        child_session_ids: dict[str, str] = {}
        for wu in enriched_units:
            child = await session_service.create_session_with_parent(
                user_id=user_id,
                parent_session_id=parent_session_id,
                tool_filter_preset=COORDINATOR_STEP_PRESET,
                coordinator_run_id=coordinator_run_id,
                work_unit_id=wu.work_unit_id,
            )
            child_session_ids[wu.work_unit_id] = child.id

        # Step 6 -- upload SpawnManifest x N (content-addressed).
        spawn_manifest_refs: dict[str, str] = {}
        spawn_manifest_shas: dict[str, str] = {}
        for wu in enriched_units:
            manifest_bytes = _serialize_spawn_manifest(wu)
            ref = await artifact.put_content_addressed_bytes(
                prefix=f"coordinator/{coordinator_run_id}/{wu.work_unit_id}/",
                content=manifest_bytes,
                filename="manifest.json",
            )
            spawn_manifest_refs[wu.work_unit_id] = ref
            spawn_manifest_shas[wu.work_unit_id] = hashlib.sha256(manifest_bytes).hexdigest()

        # C2 PR-3 §7.5 [r1 P0-1 + r2 P1-2 fix] — pre-create the terminal-waiter
        # consumer group BEFORE runners start (and BEFORE SPAWN_REQUEST is
        # published).
        #
        # XREADGROUP delivers messages added AFTER the group is created (id="$"),
        # so any child that publishes RESULT_READY/CANCEL_ACK very fast would be
        # missed by a lazy waiter.subscribe inside worker_node.await_terminal.
        # Pre-creating the group here guarantees the waiter sees every terminal
        # envelope produced after this point.
        #
        # r2 P1-2: ``mailbox_subscriber`` is REQUIRED (fail-fast on missing DI).
        # An earlier ``cfg.get`` silently skipped pre-creation, re-opening the
        # fast-publish race when DI forgot to inject the subscriber.
        subscriber = cfg["mailbox_subscriber"]
        for wu in enriched_units:
            child_sid = child_session_ids[wu.work_unit_id]
            stream_key = f"actus:child:{root_session_id}:mailbox"
            consumer_group = f"coordinator:waiter:{child_sid}"
            await subscriber.subscribe(
                stream_key=stream_key,
                consumer_group=consumer_group,
                consumer_name=f"waiter-{child_sid}",
            )
            # [codex R8 P2-1] Track for rollback destroy so a SUBSEQUENT
            # failure (runner_starter.start / publisher.publish /
            # orchestrator launch) does not leak the consumer group in
            # Redis. The except-clause below tears these down before
            # re-raising.
            created_waiter_groups.append((stream_key, consumer_group))
            # [C2b budget §3-9 R4#1] Pre-create the child cancel-LISTENER
            # group too (same start_id="$" retention property): a
            # CANCEL_REQUEST published before the child's own listener
            # subscribes is retained as group backlog and consumed once the
            # listener starts — this closes the pre-subscribe race the
            # listener docstring used to accept. The listener's own
            # subscribe is BUSYGROUP-idempotent; its shutdown() destroys
            # the group on the normal path, and the SAME rollback list
            # tears it down here on dispatch failure (double-destroy is
            # idempotent at the adapter — R5#6).
            listener_group = f"coordinator:child:{child_sid}"
            await subscriber.subscribe(
                stream_key=stream_key,
                consumer_group=listener_group,
                consumer_name=f"{child_sid}-listener",
            )
            created_waiter_groups.append((stream_key, listener_group))

        # [finish-core §5.4 G4-min, INV-F4.1] Pre-create the orchestrator's
        # observer group SYNCHRONOUSLY before ANY child task launches, so a
        # fast child's terminal envelope published between runner_starter.start
        # and the orchestrator's consume() is still captured (Redis consumer
        # groups with start_id="$" capture every message appended AFTER the
        # group is created). dispatch owns this group + its
        # pre-orchestrator-start rollback (INV-F4.3); the orchestrator owns the
        # normal-completion destroy (its finally fires; ``subscribed`` stays
        # True because run() skips its own subscribe via
        # observer_group_precreated=True).
        orchestrator_stream_key = f"actus:child:{root_session_id}:mailbox"
        orchestrator_group = f"coordinator:{coordinator_run_id}"
        await subscriber.subscribe(
            stream_key=orchestrator_stream_key,
            consumer_group=orchestrator_group,
            consumer_name=f"orch-{coordinator_run_id}",
            start_id="$",
        )
        orchestrator_group_created = (orchestrator_stream_key, orchestrator_group)

        # Step 7 -- start N runner tasks.
        # r4 P1-2: pass ``parent_session_id`` so PR-4's runner finalizers
        # (``_publish_result_ready`` / ``_publish_cancel_ack``) can build envelopes
        # without parsing the coordinator_run_id string. The runner ctor already
        # accepts ``parent_session_id`` at the skeleton level (Task 3.4).
        #
        # PR-9b-A audit round-1 P1: ``parent_sandbox`` MUST be threaded through
        # ``runner_starter.start(...)`` — A2's
        # ``DefaultCoordinatorChildRunnerStarter.start`` ctor (api/app/application/
        # services/coordinator_child_runner_starter.py:124-135) declares it as a
        # required kwarg (per-run; comes from cfg["parent_sandbox"]). Without
        # this kwarg the first flag-on dispatch would crash with TypeError
        # before any SPAWN_REQUEST publishes. The value is injected into cfg by
        # ``PlannerReActFlow._build_config`` at planner_react.py:1472
        # (``"parent_sandbox": self._sandbox``).
        parent_sandbox = cfg.get("parent_sandbox")
        for wu in enriched_units:
            # [C2b budget §3-9 D9] PER-CHILD cancel event. The run-level
            # cfg["cancel_event"] stays with the orchestrator (parent-cancel
            # observation) + worker_node waiter; user/parent cancel reaches a
            # child ONLY via the orchestrator's CANCEL_REQUEST envelope
            # fan-out → the child's cancel listener → request_stop(
            # PARENT_CANCEL) → this child-local event. A budget/watchdog trip
            # sets ONLY this child's event — sibling policy is the
            # orchestrator's RESULT_READY decision, never raw-event
            # contagion (INV-B7/INV-B8).
            child_cancel_event = asyncio.Event()
            await runner_starter.start(
                coordinator_run_id=coordinator_run_id,
                work_unit=wu,
                child_session_id=child_session_ids[wu.work_unit_id],
                spawn_manifest_ref=spawn_manifest_refs[wu.work_unit_id],
                cancel_event=child_cancel_event,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
                parent_sandbox=parent_sandbox,
                user_id=user_id,
            )
            started_child_session_ids.append(child_session_ids[wu.work_unit_id])

        # Step 8 -- publish SPAWN_REQUEST x N.
        from app.application.services.coordinator_envelope_factory import (
            CoordinatorEnvelopeFactory,
        )
        envelope_factory = cfg.get("envelope_factory") or CoordinatorEnvelopeFactory()
        for wu in enriched_units:
            envelope = envelope_factory.make_spawn_request(
                parent_session_id=parent_session_id,
                child_session_id=child_session_ids[wu.work_unit_id],
                correlation_id=coordinator_run_id,
                coordinator_run_id=coordinator_run_id,
                work_unit_id=wu.work_unit_id,
                spawn_manifest_ref=spawn_manifest_refs[wu.work_unit_id],
                spawn_manifest_sha256=spawn_manifest_shas[wu.work_unit_id],
            )
            await publisher.publish(envelope)

        # Step 9 -- orchestrator launch.
        # r3 P1-2: pass ``parent_session_id`` to the factory so the orchestrator's
        # ``make_cancel_request(parent_session_id=...)`` resolves to the real
        # parent (not the ``""`` default). Live publisher derives the Redis
        # stream key from ``envelope.parent_session_id``; an empty string would
        # publish CANCEL_REQUEST to ``actus:child::mailbox`` and never reach
        # the child.
        #
        # PR-9b-A6: bind ``emit_event`` to the per-stream ``event_queue`` so
        # the orchestrator's CoordinatorSiblingCancelEvent is captured by the
        # SSE timeline. The closure mirrors the pattern in
        # ``main_graph._run_parallel_backend`` (INV-A3): async signature
        # using synchronous ``put_nowait`` to avoid the cancellation window
        # from awaiting a full queue. When event_queue is absent (legacy
        # tests / non-coordinator paths) ``emit_event`` is left as None and
        # the orchestrator silently no-ops (matches PR-8 behaviour).
        orch_event_queue = cfg.get("event_queue")
        orch_emit_event = None
        if orch_event_queue is not None:
            async def _orch_emit_into_queue(event: object) -> None:
                # [INV-A3] put_nowait — sync, cancellation-safe.
                orch_event_queue.put_nowait(event)

            orch_emit_event = _orch_emit_into_queue

        orchestrator = orchestrator_factory.build(
            coordinator_run_id=coordinator_run_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
            emit_event=orch_emit_event,
        )
        orchestrator_task = asyncio.create_task(orchestrator.run(
            coordinator_run_id=coordinator_run_id,
            root_session_id=root_session_id,
            work_units_pending=[wu.work_unit_id for wu in enriched_units],
            child_session_ids=child_session_ids,
            cancel_event=cancel_event,
            # [finish-core §5.4 G4-min] dispatch pre-created the observer group
            # above; tell run() to skip its own subscribe but KEEP its
            # finally: destroy_group (it now owns the normal-completion teardown).
            observer_group_precreated=True,
        ))
        # r6 P1-1: attach done_callback to surface unhandled orchestrator
        # exceptions (e.g., r5 P1-2 "all CANCEL_REQUEST publishes failed" raise).
        # The task is fire-and-forget — without this callback its exception would
        # be silently swallowed by asyncio (no one awaits it; subgraph state is
        # discarded by _run_parallel_backend). PR-6 will replace this with a
        # supervised orchestrator lifecycle.
        orchestrator_task.add_done_callback(_log_orchestrator_task_done)
    except BaseException:
        # [C2b budget §3-9 R3#1] FIRST: stop the children this dispatch
        # already started. After the D9 event split the run-level event no
        # longer reaches them, so an explicit scoped stop is the only thing
        # standing between a failed dispatch and N orphaned RUNNING children
        # burning budget until their watchdogs trip. Scoped to
        # started_child_session_ids (R4#2: the starter is shared by
        # concurrent runs). Runs BEFORE quota release (R6#5) and is
        # best-effort — never masks the original dispatch exception.
        if started_child_session_ids:
            try:
                runner_starter.request_stop_started(started_child_session_ids)
            except BaseException:  # noqa: BLE001 — preserve original raise
                logger.warning(
                    "dispatch rollback: request_stop_started failed for %s "
                    "— children fall back to their wallclock watchdogs; "
                    "original dispatch exception preserved via outer raise",
                    started_child_session_ids,
                    exc_info=True,
                )

        # [codex R2 P1-3] Release the slot we acquired so a downstream
        # failure (digest computation / MinIO upload / session create /
        # manifest upload / subscribe / runner start / SPAWN_REQUEST
        # publish / orchestrator launch) does not pin the user's
        # concurrency slot until TTL. The release itself is best-effort
        # — Redis errors are logged but do NOT mask the dispatch
        # exception.
        #
        # ``BaseException`` (rather than ``Exception``) so asyncio
        # ``CancelledError`` from a parent task also releases the slot.
        # The release is followed by ``raise`` (re-raise the original)
        # so CancelledError still aborts the task and ValueError still
        # reaches the caller.
        #
        # NB: the Command(update={"quota_acquired": True, ...}) at the
        # tail of this function never executes when we hit this except,
        # so reducer_node.state["quota_acquired"] stays False and the
        # reducer's finally won't double-release.
        if concurrency_acquired and probe_quota is not None:
            try:
                await probe_quota.release_coordinator_quotas(user_id=user_id)
            except BaseException:  # noqa: BLE001 — see comment below
                # [codex R3 P1-4] MUST catch ``BaseException``, not
                # ``Exception``. On Py3.12 ``asyncio.CancelledError``
                # subclasses ``BaseException`` (not ``Exception``); if
                # release is itself cancelled by an upstream task-group
                # tear-down, an ``except Exception`` lets the
                # ``CancelledError`` escape and MASKS the original
                # dispatch exception (Python raises the new one and
                # chains the original via ``__context__``, which most
                # callers don't inspect). Log + swallow on ANY exit so
                # the trailing ``raise`` re-raises the ORIGINAL dispatch
                # exception unaltered — release failure is observability,
                # not a control-flow signal.
                logger.exception(
                    "dispatch rollback: release_coordinator_quotas failed "
                    "user=%s — manual cleanup may be needed (counter "
                    "will leak until TTL); original dispatch exception "
                    "preserved via outer raise",
                    user_id,
                )

        # [codex R8 P2-1] Destroy any pre-created waiter consumer groups
        # so they don't accumulate in Redis. If the dispatch failed
        # AFTER the pre-create loop (subscribe succeeded for N groups,
        # then runner_starter.start / publisher.publish / orchestrator
        # launch raised), each leaked group would otherwise stay in
        # Redis forever — XGROUP groups are persistent and there is
        # no orphan reaper for them.
        #
        # Best-effort: log + swallow on any error so the trailing
        # ``raise`` re-raises the ORIGINAL dispatch exception. We catch
        # ``BaseException`` for the same reason as the quota-release
        # block above (CancelledError on Py3.12 subclasses
        # BaseException, not Exception).
        for stream_key, consumer_group in created_waiter_groups:
            try:
                await subscriber.destroy_group(
                    stream_key=stream_key,
                    consumer_group=consumer_group,
                )
            except BaseException:  # noqa: BLE001 — see comment above
                logger.warning(
                    "dispatch rollback: destroy_group failed group=%s "
                    "— dead group may accumulate in Redis until manual "
                    "cleanup; original dispatch exception preserved "
                    "via outer raise",
                    consumer_group,
                    exc_info=True,
                )

        # [finish-core §5.4 G4-min, INV-F4.3] Destroy the pre-created
        # orchestrator group ONLY if the orchestrator task was never launched
        # (dispatch failed between pre-create and create_task). If the task
        # exists, the orchestrator's own finally owns the destroy — do NOT
        # double-destroy here.
        if orchestrator_group_created is not None and orchestrator_task is None:
            sk, cg = orchestrator_group_created
            try:
                await subscriber.destroy_group(stream_key=sk, consumer_group=cg)
            except BaseException:  # noqa: BLE001 — best-effort; preserve original raise
                logger.warning(
                    "dispatch rollback: orchestrator destroy_group failed group=%s",
                    cg, exc_info=True,
                )

        raise

    # C2 PR-8 §13 Task 8.4 — emit CoordinatorDispatchEvent + N
    # CoordinatorWorkerSpawnedEvent so the parent SSE timeline can render
    # the fan-out. BEST-EFFORT: try/except so a queue/serialization failure
    # never aborts dispatch (state has already committed side effects). The
    # except clause MUST stay above the Command(...) return because the
    # rollback path at line ~564 deliberately bypasses emit (no events for
    # a dispatch that never completed).
    event_queue: Optional[asyncio.Queue] = (
        config.get("configurable", {}).get("event_queue")
    )
    if event_queue is not None:
        # [codex R4 P1] Use put_nowait (synchronous, non-cancellable) instead of
        # await put. The dispatch has already created N child sessions, N runners,
        # N SPAWN_REQUEST publishes, and started the orchestrator task on the
        # happy-path side of the BaseException rollback above. An await between
        # those committed side effects and the Command return is a CancelledError
        # window (CancelledError subclasses BaseException, NOT Exception, on
        # Py3.12) — a cancel would suppress the state handoff and leave the
        # resources orphaned. Production event_queue is asyncio.Queue with
        # default maxsize=0 (unbounded — see event_bridge.py); put_nowait never
        # raises QueueFull there. The try/except still catches a defensive bounded
        # queue (raises QueueFull) or other unexpected errors.
        try:
            from app.domain.models.event import (
                CoordinatorDispatchEvent,
                CoordinatorWorkerSpawnedEvent,
            )
            event_queue.put_nowait(CoordinatorDispatchEvent(
                step_id=state.get("step_id", "unknown"),
                work_unit_count=len(enriched_units),
                work_unit_ids=[wu.work_unit_id for wu in enriched_units],
                phases=[wu.phase for wu in enriched_units],
                coordinator_run_id=coordinator_run_id,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
            ))
            for wu in enriched_units:
                event_queue.put_nowait(CoordinatorWorkerSpawnedEvent(
                    objective=wu.objective,
                    phase=wu.phase,
                    allowed_tools=list(wu.allowed_tools),
                    write_lease_count=len(wu.write_lease),
                    coordinator_run_id=coordinator_run_id,
                    work_unit_id=wu.work_unit_id,
                    child_session_id=child_session_ids[wu.work_unit_id],
                    root_session_id=root_session_id,
                    parent_session_id=parent_session_id,
                ))
        except Exception as exc:
            logger.warning(
                "_first_time_dispatch: emit coordinator events failed "
                "run=%s: %s", coordinator_run_id, exc,
            )

    # Step 10 -- fan out via Send x N.
    return Command(
        update={
            "coordinator_run_id": coordinator_run_id,
            "work_units": enriched_units,
            "child_session_ids": child_session_ids,
            "orchestrator_task": orchestrator_task,
            # [codex R2 P1-4] Mark the slot as held so reducer_node knows
            # to release it. Rehydrate dispatch (``_rehydrate_dispatch``)
            # does NOT acquire and therefore does NOT set this flag, so
            # reducer_node correctly skips release on the rehydrate path
            # (which would otherwise DECR below zero / steal another
            # incarnation's slot).
            "quota_acquired": concurrency_acquired,
            "dispatch_started_monotonic": dispatch_started_monotonic,
        },
        goto=[
            Send("worker_node", {
                "work_unit_id": wu.work_unit_id,
                "child_session_id": child_session_ids[wu.work_unit_id],
                "coordinator_run_id": coordinator_run_id,
                "root_session_id": root_session_id,
                # [S2 §3.2 R3-F1] write-phase ⇒ a resolvable manifest is
                # required; worker_node demotes SUCCESS-no-manifest to FAILED.
                "manifest_required": wu.phase == "write",
            })
            for wu in enriched_units
        ],
    )


async def _build_pre_results_from_terminal(
    terminal: "dict[str, TerminalEnvelopeRecord]",
    *,
    artifact_storage: Any,
    work_units_by_id: "dict[str, WorkUnit]",
) -> list[WorkerResult]:
    """[C2 PR-7 §12.3 + S2 §3.2 C1] Convert persisted terminal envelopes back
    into WorkerResult.

    The reducer consumes ``state["worker_results"]`` -- a list[WorkerResult]
    accumulated by ``worker_node`` Send fan-in. After a crash, we restore
    the same shape from ``coordinator_result_envelope_store`` rows so the
    reducer sees a complete worker_results set for already-terminated wu's
    without having to await them again.

    Per [r3 P1-3] the TerminalEnvelopeRecord carries ``envelope_type`` so
    we can correctly map RESULT_READY -> outcome from the payload and
    CANCEL_ACK -> outcome from final_state, mirroring ``worker_node``'s live
    decode.

    [S2 §3.2 C1] Now async: resolves ``patch_manifest_ref`` from the persisted
    payload via ``artifact_storage`` (shared ``_resolve_manifest`` helper), and
    demotes a write-phase (``work_units_by_id[wu_id].phase == "write"``) SUCCESS
    with no resolvable manifest to FAILED — the same fail-close worker_node
    applies live, so a crash-recovery replay can never silently zero-apply a
    dropped/truncated manifest.
    """
    pre_results: list[WorkerResult] = []
    for wu_id, record in terminal.items():
        payload = record.payload if isinstance(record.payload, dict) else {}
        if record.envelope_type == "RESULT_READY":
            outcome_raw = payload.get("outcome", "failed")
            try:
                outcome = ResultReadyOutcome(outcome_raw)
            except ValueError:
                outcome = ResultReadyOutcome.FAILED
            cost_raw = payload.get("cost_summary")
            cost_summary: Optional[CostAggregate] = None
            if isinstance(cost_raw, dict):
                try:
                    cost_summary = CostAggregate.model_validate(cost_raw)
                except Exception:  # noqa: BLE001 -- degrade to default
                    cost_summary = None
            summary = payload.get("summary")
            wu = work_units_by_id.get(wu_id)
            manifest_required = wu is not None and wu.phase == "write"
            patch_manifest: Optional[Any] = None
            if outcome == ResultReadyOutcome.SUCCESS:
                # [codex R2 P1] JSONB round-trip lands ``patch_manifest`` as a
                # plain ``dict`` after psycopg decode. Downstream consumers
                # (``patch_reducer_service.py``) do attribute access like
                # ``pm.coordinator_run_id`` on the manifest, which fails on
                # ``dict``. Build a thin shim exposing the two fields
                # ``_resolve_manifest`` reads. The persisted payload is the
                # minimum-rehydrate dict (it may carry an inline
                # ``patch_manifest`` dict OR a ``patch_manifest_ref`` string).
                # Coerce an inline dict to a typed PatchManifest first so
                # ``_resolve_manifest``'s inline branch returns the model, not a
                # dict.
                pm_raw = payload.get("patch_manifest")
                inline: Optional[Any] = None
                if isinstance(pm_raw, dict):
                    from app.domain.models.patch_manifest import PatchManifest
                    try:
                        inline = PatchManifest.model_validate(pm_raw)
                    except Exception:  # noqa: BLE001 -- degrade to None
                        inline = None
                elif pm_raw is not None:
                    inline = pm_raw  # typed model (mid-process replay)
                shim = _ManifestShim(
                    patch_manifest=inline,
                    patch_manifest_ref=payload.get("patch_manifest_ref"),
                )
                patch_manifest = await _resolve_manifest(
                    shim, artifact_storage=artifact_storage,
                )
                if manifest_required and patch_manifest is None:
                    logger.warning(
                        "rehydrate: demoting SUCCESS→FAILED for wu=%s — "
                        "write-phase manifest required but None/unresolvable "
                        "(ref=%r)",
                        wu_id, payload.get("patch_manifest_ref"),
                    )
                    outcome = ResultReadyOutcome.FAILED
                    summary = (
                        f"{summary or ''} [demoted: write-phase SUCCESS with "
                        "no resolvable patch_manifest]"
                    ).strip()
                    patch_manifest = None
            pre_results.append(WorkerResult(
                work_unit_id=wu_id,
                child_session_id=record.child_session_id,
                outcome=outcome,
                cost_summary=cost_summary,
                summary=summary,
                patch_manifest=patch_manifest,
                needs_authorization_details=payload.get(
                    "needs_authorization_details"
                ),
                manifest_required=manifest_required,
            ))
        elif record.envelope_type == "CANCEL_ACK":
            final_state = payload.get("final_state")
            if final_state == "cancelled":
                outcome = ResultReadyOutcome.CANCELLED
            elif final_state == "force_terminated":
                outcome = ResultReadyOutcome.TIMED_OUT
            else:
                outcome = ResultReadyOutcome.FAILED
            pre_results.append(WorkerResult(
                work_unit_id=wu_id,
                child_session_id=record.child_session_id,
                outcome=outcome,
                summary=payload.get("summary"),
            ))
        # Unknown envelope_type -> log + skip (orphan persisted row from a
        # future envelope_type that this code-version doesn't understand).
        else:
            logger.warning(
                "rehydrate: unknown envelope_type=%r for wu_id=%s; "
                "skipping pre_result construction",
                record.envelope_type, wu_id,
            )
    return pre_results


async def _rehydrate_dispatch(
    state: ParallelSubgraphState,
    config: dict,
    existing: "RehydrateResult",
    coordinator_run_id: str,
    work_units: list[WorkUnit],
) -> Command:
    """[C2 PR-7 §12] Resume dispatch from a prior coordinator run.

    Called when ``dispatch_node`` PEEK saw a non-None attempt_ix AND
    ``rehydrate_service.detect_existing_run`` returned a non-None
    ``RehydrateResult``. Implements §12.3 steps 4-8:

      * Step 4 (already_applied SHORT-CIRCUIT): if the apply audit row
        is success/rollback_partial/crash_mid_apply/in_progress_recent,
        skip worker_node + reducer entirely; main_graph reads the
        ``step_result_candidate`` and short-circuits its apply call
        too. Avoids re-running a successful apply / re-attempting an
        operator-blocked crash recovery.

      * Step 5 (UNEXPECTED CHILD): if a child row exists for a wu_id
        that's NOT in the current ``work_units`` (e.g. plan changed
        between attempts), best-effort publish a CANCEL_REQUEST so the
        orphan child terminates cleanly. NOT load-bearing (orphan reaper
        also catches it).

      * Step 6 (MISSING CHILD): if a wu_id is in ``work_units`` but
        has NO corresponding child row, RAISE a RuntimeError ([codex R1
        P1] fail loudly) so the orchestrator surfaces an
        operator-actionable error rather than routing an incomplete
        worker_results set to the reducer. TODO(PR-7+ partial-INSERT-
        unique idempotent spawn) will instead re-spawn the missing
        child idempotently.

      * Step 7 (TERMINAL -> pre-populate): inject the terminal-envelope-
        derived WorkerResult objects into ``state["worker_results"]``
        so the reducer sees them as if worker_node had completed.

      * Step 8 (PENDING -> Send): for the remaining truly-pending wu_ids,
        keep the live r3 P1-1 pre-subscribe-then-Send pattern so the
        waiter group exists before the worker_node loop dequeues.

    r4 P1-1: rehydrate path uses ``start_id="0"`` so the new consumer
    group sees terminal envelopes already buffered in the stream BEFORE
    this subscribe. Live ``_first_time_dispatch`` stays on default ``"$"``
    because the child hasn't published anything yet.
    """
    cfg = config["configurable"]
    parent_session_id = state["parent_session_id"]
    root_session_id = state["root_session_id"]

    # Step 4: already_applied short-circuit (BEFORE any waiter subscribe /
    # publish work -- minimize side effects when the apply was already
    # committed successfully or is in a manual-recovery state).
    if existing.already_applied is not None:
        applied = existing.already_applied
        candidate = f"ALREADY_APPLIED:{applied.status}:{applied.audit_id}"
        logger.info(
            "rehydrate: already_applied short-circuit run=%s status=%s audit=%d",
            coordinator_run_id, applied.status, applied.audit_id,
        )
        return Command(
            update={
                "coordinator_run_id": coordinator_run_id,
                "work_units": work_units,
                "child_session_ids": existing.child_session_ids,
                "step_result_candidate": candidate,
                "group_outcome": None,
            },
            goto=END,
        )

    # Step 5: unexpected-child CANCEL_REQUEST (best-effort).
    #
    # [codex R5 P2 -- deferred to PR-7+] Cross-pod idempotency gap:
    # ``envelope_factory.make_cancel_request`` generates a fresh UUID per
    # call; the orchestrator's ``_published`` dedup set is instance-local
    # and not used on the rehydrate path. Two pods concurrently
    # rehydrating the same run can each fire CANCEL_REQUEST for the same
    # unexpected child, producing two callback invocations on the child
    # side (no timer-reset, but redundant downstream work). PR-7+ should
    # derive the cancel envelope_id deterministically from
    # ``(coordinator_run_id, work_unit_id, child_session_id, reason)`` so
    # the supervisor's audit_repo dedup catches duplicates.
    expected_wu_ids = {wu.work_unit_id for wu in work_units}
    publisher = cfg.get("mailbox_publisher")
    envelope_factory = cfg.get("envelope_factory")
    for wu_id, child_sid in existing.child_session_ids.items():
        if wu_id not in expected_wu_ids:
            if publisher is None or envelope_factory is None:
                logger.warning(
                    "rehydrate: unexpected child wu_id=%s (sid=%s) but "
                    "publisher/envelope_factory missing -- skipping CANCEL_REQUEST",
                    wu_id, child_sid,
                )
                continue
            try:
                env = envelope_factory.make_cancel_request(
                    parent_session_id=parent_session_id,
                    child_session_id=child_sid,
                    correlation_id=coordinator_run_id,
                    reason="unexpected_child_after_rehydrate",
                )
                await publisher.publish(env)
                logger.info(
                    "rehydrate: published CANCEL_REQUEST for unexpected wu_id=%s",
                    wu_id,
                )
            except Exception:  # noqa: BLE001 -- best-effort cancel
                logger.exception(
                    "rehydrate: failed to publish CANCEL_REQUEST for wu_id=%s",
                    wu_id,
                )

    # Step 6: missing-child gap. v1 contract: rehydrate is only reached
    # when at least one child row exists for the peeked attempt, so the
    # work_units shape SHOULD match what was originally dispatched. A
    # non-empty ``missing_wu_ids`` means the planner produced a different
    # work_unit set on retry (plan changed between attempts) or the
    # original create_session_with_parent loop crashed PARTWAY through
    # (some children committed, some did not). Either case is a
    # rehydrate-recovery hazard: silently routing to reducer with N-K
    # worker_results would let the reducer build an incomplete
    # PatchApplyPlan and the applier would commit a subset of the
    # planned changes.
    #
    # [codex R1 P1] Fail loudly here so the orchestrator surfaces a
    # human-actionable error rather than data-eating fall-through.
    # PR-7+ will implement idempotent ``_spawn_one(wu)`` per missing
    # wu_id (depends on a ``sessions`` partial-unique on
    # ``(coordinator_run_id, work_unit_id)`` so the INSERT is safe under
    # concurrent re-dispatch).
    missing_wu_ids = [
        wu.work_unit_id for wu in work_units
        if wu.work_unit_id not in existing.child_session_ids
    ]
    if missing_wu_ids:
        raise RuntimeError(
            f"rehydrate: cannot resume run {coordinator_run_id!r} -- "
            f"work_units include {missing_wu_ids!r} but no child sessions "
            f"exist for those ids. Either the planner produced a different "
            f"work_unit set on retry, or the original dispatch crashed "
            f"part-way through child creation. PR-7+ will idempotently "
            f"re-spawn; v1 surfaces this as a hard failure so the "
            f"orchestrator can replan or operator can intervene."
        )

    # [codex R4 P1] Limbo-child guard. A wu_id is "limbo" if its child row
    # exists (so not missing) but the child is NEITHER classified as
    # pending/running by the rehydrate service NOR has a persisted
    # terminal envelope. Most likely cause: PR-7's best-effort
    # ``persist_terminal`` PROLOGUE in MailboxSupervisor swallowed an
    # exception (warning-logged), the child was destroyed + status moved
    # to a terminal value, but the envelope row was never written. On
    # retry, rehydrate sees the child in a terminal status with no
    # envelope record → would otherwise fall through to reducer with an
    # incomplete worker_results set, producing INCOMPLETE silently.
    # Fail loudly so operator can replan / clean up. PR-7+ should harden
    # ``persist_terminal`` with a transactional pattern so this case
    # becomes unreachable.
    expected_wu_ids = {wu.work_unit_id for wu in work_units}
    limbo_wu_ids = sorted(
        expected_wu_ids
        - set(existing.terminal.keys())
        - set(existing.pending)
        - set(missing_wu_ids)
    )
    if limbo_wu_ids:
        raise RuntimeError(
            f"rehydrate: cannot resume run {coordinator_run_id!r} -- "
            f"work_units {limbo_wu_ids!r} have child rows in a non-pending "
            f"non-running status but NO persisted terminal envelope. "
            f"Likely cause: PR-7 ``persist_terminal`` PROLOGUE swallowed "
            f"an exception on the original run, leaving the envelope_store "
            f"row missing. Operator must inspect the affected children's "
            f"status + supervisor logs and either reset coordinator_attempt "
            f"or backfill the envelope row before retrying."
        )

    # [codex PR-2 R1 P1 + opus-review follow-up] Out-of-plan guard (terminal AND
    # pending). ``existing.terminal`` / ``existing.pending`` are built by
    # ``CoordinatorRehydrateService.detect_existing_run`` (Steps 4-5) from EVERY
    # persisted envelope / child row for this ``coordinator_run_id`` with no
    # filtering against the rebuilt ``work_units``. Because ``coordinator_run_id``
    # embeds the attempt (``…:a{attempt_ix}``), prior-attempt rows never return —
    # so a wu_id in ``terminal`` OR ``pending`` but NOT in ``expected_wu_ids``
    # means the persisted run shape no longer matches the current plan (state
    # ``work_unit_requests`` changed, or the unit count shrank, between the
    # original dispatch and this rehydrate). That is the same drift class the
    # missing / limbo guards reject in the other direction. Left unguarded BOTH
    # routes inject out-of-plan writes into the apply plan:
    #   • terminal → ``_build_pre_results_from_terminal`` computes
    #     ``manifest_required = work_units_by_id.get(wu_id) is not None and …`` →
    #     ``False`` for the unknown id → NO demotion;
    #   • pending  → Step 8 ``Send``s it to ``worker_node`` with the same
    #     ``_phase_by_wu_id.get(wu_id) == "write"`` → ``False`` → NO demotion.
    # In either case a self-consistent SUCCESS manifest passes the reducer's
    # lineage cross-check (matched against the worker_result's OWN wu_id, not the
    # expected set) and folds into the apply plan. Fail loudly — symmetric with
    # the missing / limbo guards — so the orchestrator surfaces an
    # operator-actionable error instead of a silent fail-open. (Step 5 still
    # best-effort CANCELs any out-of-plan child first; this guard then refuses
    # the inconsistent resume.)
    unexpected_wu_ids = sorted(
        (set(existing.terminal.keys()) | set(existing.pending)) - expected_wu_ids
    )
    if unexpected_wu_ids:
        raise RuntimeError(
            f"rehydrate: cannot resume run {coordinator_run_id!r} -- "
            f"unexpected terminal/pending work units exist for "
            f"{unexpected_wu_ids!r} which are NOT in the current attempt's "
            f"work_units. The persisted run shape no longer matches the rebuilt "
            f"plan (work_unit_requests changed or the unit count shrank between "
            f"dispatch and rehydrate). v1 fails closed so a stale child SUCCESS "
            f"can never inject out-of-plan writes into the apply plan; operator "
            f"must replan or reset coordinator_attempt."
        )

    # [codex PR-2 R2 P1] Stale-child guard. Terminal records are keyed by
    # work_unit_id only (CoordinatorRehydrateService Step 4), and
    # (coordinator_run_id, work_unit_id) is NOT yet unique on the sessions table
    # (the partial-unique is deferred to PR-7 — see the missing-child guard
    # above). A duplicate / stale child row for an EXPECTED wu_id can therefore
    # surface a terminal envelope whose child_session_id differs from the
    # current child (``existing.child_session_ids[wu_id]`` = the newest child by
    # created_at). Left unguarded, the terminal-record path turns that stale
    # envelope into the WorkerResult for the expected wu_id (and ``pending``
    # skips the REAL current child), so a stale-child SUCCESS manifest folds
    # into the apply plan while the live child's work is dropped — the reducer's
    # lineage cross-check validates run_id + work_unit_id but NOT
    # child_session_id. Fail closed. (Runs after the union guard, so every
    # terminal wu_id here is in-plan and — after the missing-child guard — has a
    # current child entry.)
    for _wu_id, _rec in existing.terminal.items():
        _current_child = existing.child_session_ids.get(_wu_id)
        if _rec.child_session_id != _current_child:
            raise RuntimeError(
                f"rehydrate: cannot resume run {coordinator_run_id!r} -- "
                f"terminal envelope for wu_id={_wu_id!r} was produced by child "
                f"{_rec.child_session_id!r} but the current child for that work "
                f"unit is {_current_child!r} (stale / duplicate child row; "
                f"(coordinator_run_id, work_unit_id) is not yet unique pre-PR-7). "
                f"v1 fails closed so a stale child's result cannot masquerade as "
                f"the current child's; operator must reconcile the child rows or "
                f"reset coordinator_attempt."
            )

    # Step 7: pre-populate worker_results from persisted terminal envelopes.
    # [S2 §3.2 C1] async now: resolve manifest-by-ref + demote write-phase
    # SUCCESS-no-manifest. ``cfg`` is config["configurable"] (bound at the top
    # of _rehydrate_dispatch); ``work_units`` is the function parameter.
    pre_results = await _build_pre_results_from_terminal(
        existing.terminal,
        artifact_storage=cfg.get("artifact_storage"),
        work_units_by_id={wu.work_unit_id: wu for wu in work_units},
    )

    # Step 8: pre-subscribe waiter group + Send only for truly-pending.
    subscriber = cfg["mailbox_subscriber"]  # fail-fast on missing DI [r2 P1-2]
    for wu_id in existing.pending:
        child_sid = existing.child_session_ids[wu_id]
        await subscriber.subscribe(
            stream_key=f"actus:child:{root_session_id}:mailbox",
            consumer_group=f"coordinator:waiter:{child_sid}",
            consumer_name=f"waiter-{child_sid}",
            start_id="0",
        )

    _phase_by_wu_id = {wu.work_unit_id: wu.phase for wu in work_units}
    pending_sends = [
        Send("worker_node", {
            "work_unit_id": wu_id,
            "child_session_id": existing.child_session_ids[wu_id],
            "coordinator_run_id": coordinator_run_id,
            "root_session_id": root_session_id,
            # [S2 §3.2 R3-F1] mirror the first-time payload so a replayed
            # pending worker is held to the same manifest-required contract.
            "manifest_required": _phase_by_wu_id.get(wu_id) == "write",
        })
        for wu_id in existing.pending
    ]

    return Command(
        update={
            "coordinator_run_id": coordinator_run_id,
            "work_units": work_units,
            "child_session_ids": existing.child_session_ids,
            "worker_results": pre_results,
        },
        # If no pending (all wu_ids had terminal envelopes already), skip
        # the worker_node fan-out and go straight to reducer_node.
        goto=pending_sends if pending_sends else "reducer_node",
    )


# ── worker_node ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ManifestShim:
    """[S2 §3.2 C1] Minimal attribute carrier so the shared _resolve_manifest
    helper can read inline-or-ref from a rehydrate payload dict the same way it
    reads from a live ResultReadyPayload."""

    patch_manifest: Optional[Any]
    patch_manifest_ref: Optional[str]


async def _resolve_manifest(
    rr_payload: Any,
    *,
    artifact_storage: Any,
) -> Optional[Any]:
    """[C2-full S2 §3.2 C1] Shared inline/ref manifest resolver.

    Returns the inline ``patch_manifest`` if present; else resolves
    ``patch_manifest_ref`` via ``artifact_storage.get_bytes`` +
    ``PatchManifest.model_validate``. Returns ``None`` on absent ref OR on any
    resolution failure (missing key / corrupt JSON / validation error) — the
    caller decides whether None is fatal (write-phase) or legal (exploration).
    The two fields are mutually exclusive by the ResultReadyPayload validator,
    so at most one branch fires.
    """
    inline = getattr(rr_payload, "patch_manifest", None)
    if inline is not None:
        return inline
    ref = getattr(rr_payload, "patch_manifest_ref", None)
    if ref is None:
        return None
    if artifact_storage is None:
        logger.warning(
            "manifest-by-ref present (%r) but no artifact_storage in config; "
            "treating as unresolved",
            ref,
        )
        return None
    try:
        raw = await artifact_storage.get_bytes(ref)
        from app.domain.models.patch_manifest import PatchManifest
        return PatchManifest.model_validate(json.loads(raw))
    except Exception as exc:  # noqa: BLE001 — get_bytes / json / validation
        logger.warning(
            "manifest-by-ref resolution failed for ref=%r: %s", ref, exc
        )
        return None


async def worker_node(state_per_send: dict, config: RunnableConfig) -> dict:
    """Thin await of the terminal envelope; normalize CANCEL_ACK final_state."""
    cfg = config["configurable"]
    waiter = cfg["terminal_waiter"]
    cancel_event = cfg["cancel_event"]

    # [codex R5 P1] Pass coordinator_run_id so the waiter additionally
    # filters by ``env.correlation_id``. PR-5 applies patch_manifests
    # from these envelopes directly to the parent sandbox; without
    # this filter a stale RESULT_READY from a different attempt
    # sharing the same child_session_id could cross-contaminate the
    # current apply plan.
    manifest_required = bool(state_per_send.get("manifest_required", False))

    # [codex PR-2 R4 P0] Fail-closed terminal decode. The waiter matches the
    # terminal on envelope-level fields (type/child/correlation) over a RAW
    # dict, and ``RedisMailboxSubscriber.consume`` XACKs the matched entry
    # BEFORE the waiter deep-validates it via ``MailboxEnvelope.model_validate``
    # (which runs the typed ``ResultReadyPayload`` schema). A malformed child
    # payload therefore surfaces here as a pydantic ``ValidationError`` AFTER
    # the entry is already acked/lost. Catch it and synthesize a FAILED
    # WorkerResult: the reducer's completeness invariant (every Send yields a
    # worker_result) must hold, and a single malformed child envelope must not
    # crash the whole parallel superstep or strand the run. (Only ValidationError
    # is caught — transient infra errors from the subscriber stay retryable.)
    from pydantic import ValidationError
    try:
        envelope = await waiter.await_terminal(
            child_session_id=state_per_send["child_session_id"],
            root_session_id=state_per_send["root_session_id"],
            cancel_event=cancel_event,
            coordinator_run_id=state_per_send.get("coordinator_run_id"),
        )
    except ValidationError as exc:
        logger.warning(
            "worker_node: malformed terminal envelope for wu=%s child=%s — "
            "failing closed (FAILED): %s",
            state_per_send["work_unit_id"],
            state_per_send["child_session_id"],
            exc,
        )
        return {"worker_results": [WorkerResult(
            work_unit_id=state_per_send["work_unit_id"],
            child_session_id=state_per_send["child_session_id"],
            outcome=ResultReadyOutcome.FAILED,
            summary="malformed terminal envelope (undecodable payload)",
            manifest_required=manifest_required,
        )]}
    except asyncio.TimeoutError:
        # [codex PR-2 R5 P0] The child never emitted a terminal within the
        # waiter's window. A raise here would crash the whole ``Send`` fan-out
        # superstep (only CoordinatorPathContractError is caught upstream in
        # ``_run_parallel_backend``; executor_node has no retry policy), losing
        # every sibling worker's result and stranding the run. Fail closed to a
        # TIMED_OUT WorkerResult so the reducer's completeness invariant holds
        # and one slow child (more likely for S2 shell-mode children) cannot
        # kill the batch. (TIMED_OUT → reducer non-SUCCESS → no apply.)
        logger.warning(
            "worker_node: terminal wait timed out for wu=%s child=%s — "
            "failing closed (TIMED_OUT)",
            state_per_send["work_unit_id"],
            state_per_send["child_session_id"],
        )
        return {"worker_results": [WorkerResult(
            work_unit_id=state_per_send["work_unit_id"],
            child_session_id=state_per_send["child_session_id"],
            outcome=ResultReadyOutcome.TIMED_OUT,
            summary="terminal envelope wait timed out",
            manifest_required=manifest_required,
        )]}

    # [r4 P1 fix] Propagate PR-4 wire-schema fields (patch_manifest /
    # needs_authorization_details / cost_summary / summary) into the
    # WorkerResult so PR-5 reducer + PR-6 cost aggregator have something to
    # consume. Validating envelope.payload through ResultReadyPayload gives
    # us strongly-typed access to the new optional fields.
    summary: Optional[str] = None
    cost_summary: Optional[CostAggregate] = None
    patch_manifest: Optional[Any] = None
    needs_authorization_details: Optional[Any] = None

    if envelope.type == MailboxEnvelopeType.RESULT_READY:
        from app.domain.models.mailbox_envelope import ResultReadyPayload
        if isinstance(envelope.payload, ResultReadyPayload):
            rr_payload = envelope.payload
        else:
            # Redis JSON round-trip lands payload as dict; revalidate to
            # typed model so the new optional fields land on WorkerResult.
            rr_payload = ResultReadyPayload.model_validate(envelope.payload)
        outcome = rr_payload.outcome
        summary = rr_payload.summary
        cost_summary = rr_payload.cost_summary
        needs_authorization_details = rr_payload.needs_authorization_details
        # [C2-full S2 §3.2 C1] Resolve inline-or-ref manifest, then fail-close:
        # a write-phase SUCCESS (manifest_required) with no resolved manifest
        # MUST NOT reach the reducer as SUCCESS (it would zero-apply silently).
        if outcome == ResultReadyOutcome.SUCCESS:
            patch_manifest = await _resolve_manifest(
                rr_payload, artifact_storage=cfg.get("artifact_storage"),
            )
            if manifest_required and patch_manifest is None:
                logger.warning(
                    "worker_node: demoting SUCCESS→FAILED for wu=%s — "
                    "write-phase manifest required but None/unresolvable "
                    "(ref=%r)",
                    state_per_send["work_unit_id"],
                    getattr(rr_payload, "patch_manifest_ref", None),
                )
                outcome = ResultReadyOutcome.FAILED
                summary = (
                    f"{summary or ''} [demoted: write-phase SUCCESS with no "
                    "resolvable patch_manifest]"
                ).strip()
                patch_manifest = None
        else:
            patch_manifest = None
    elif envelope.type == MailboxEnvelopeType.CANCEL_ACK:
        final_state = (
            envelope.payload.get("final_state")
            if isinstance(envelope.payload, dict)
            else getattr(envelope.payload, "final_state", None)
        )
        if final_state == "cancelled":
            outcome = ResultReadyOutcome.CANCELLED
        elif final_state == "force_terminated":
            outcome = ResultReadyOutcome.TIMED_OUT
        else:
            outcome = ResultReadyOutcome.FAILED
        # CancelAckPayload has its own ``summary`` field — surface it for
        # SSE/audit downstream (PR-8).
        cancel_summary = (
            envelope.payload.get("summary")
            if isinstance(envelope.payload, dict)
            else getattr(envelope.payload, "summary", None)
        )
        summary = cancel_summary
    else:
        outcome = ResultReadyOutcome.FAILED

    result = WorkerResult(
        work_unit_id=state_per_send["work_unit_id"],
        child_session_id=state_per_send["child_session_id"],
        outcome=outcome,
        cost_summary=cost_summary,
        summary=summary,
        patch_manifest=patch_manifest,
        needs_authorization_details=needs_authorization_details,
        manifest_required=manifest_required,
    )
    return {"worker_results": [result]}


# ── reducer_node — wired to PatchReducerService (C2 PR-5) ───────────────────


async def reducer_node(
    state: ParallelSubgraphState, config: RunnableConfig,
) -> Command:
    """[C2 PR-5 §9.6] Thin wrapper around ``PatchReducerService.reduce``.

    Reads:
    - ``config["configurable"]["patch_reducer_service"]`` — required.
      Composition root binds the singleton service instance.
    - ``config["configurable"]["parent_sandbox"]`` — optional. When
      present the reducer's §9.3 step 5 drift check fires; when absent
      drift detection is skipped (the applier's preflight still
      catches stale base_digest at apply time).
    - ``config["configurable"]["probe_quota"]`` — optional. When present
      and ``state["user_id"]`` is populated, the per-user coordinator
      concurrency slot acquired by ``_first_time_dispatch`` is released
      on reducer exit (try/finally — fires even on reducer exception).

    Writes ``apply_plan`` / ``group_outcome`` / ``step_result_candidate``
    into the subgraph state via ``Command(update=...)`` and routes to
    ``END``. The outer ``_run_parallel_backend`` reads these to decide
    whether to invoke ``PatchApplier``.
    """
    cfg = config["configurable"]
    reducer = cfg["patch_reducer_service"]
    parent_sandbox = cfg.get("parent_sandbox")
    # C2 PR-6 §14.3 #1 — release the user-concurrency slot acquired by
    # ``_first_time_dispatch``. We deliberately release here (reducer
    # exit) rather than dispatch tail so the slot stays reserved across
    # the worker/reducer span, matching the spec's "active coordinator
    # run" semantics. Best-effort: release failures are logged and
    # swallowed so they don't mask the reducer return / exception.
    #
    # [codex R2 P1-4] Gate release on ``state["quota_acquired"]`` so the
    # rehydrate dispatch path (which does NOT acquire — the slot is owned
    # by the prior incarnation that crashed) does not DECR below zero.
    # ``_first_time_dispatch`` sets the flag True via Command(update=...)
    # when concurrency was acquired AND the dispatch body completed;
    # ``_rehydrate_dispatch`` and the dispatch-body rollback path both
    # leave it False (default per ``total=False`` TypedDict).
    probe_quota = cfg.get("probe_quota")
    user_id = state.get("user_id")
    quota_acquired = state.get("quota_acquired", False)

    try:
        coordinator_run_id = state.get("coordinator_run_id")
        if coordinator_run_id is None:
            # ``dispatch_node`` is responsible for filling this. A missing
            # value here means a topology bug — fail loudly via the step
            # result candidate text rather than calling reduce() with None
            # which would yield a confusing downstream error.
            return Command(
                update={
                    "apply_plan": None,
                    "group_outcome": None,
                    "step_result_candidate": (
                        "[reducer] missing coordinator_run_id; dispatch_node "
                        "did not initialize subgraph state correctly"
                    ),
                },
                goto=END,
            )

        output = await reducer.reduce(
            coordinator_run_id=coordinator_run_id,
            work_unit_ids_expected=frozenset(
                wu.work_unit_id for wu in state["work_units"]
            ),
            worker_results=state["worker_results"],
            parent_sandbox=parent_sandbox,
        )

        # [PR-9b-B codex F1 — HIGH] Build the LOAD-BEARING handoff Command from
        # ``output`` BEFORE the best-effort telemetry block below. ``reduce()``
        # has already produced apply_plan/group_outcome/step_result_candidate;
        # those drive the downstream apply branch and MUST reach the graph. By
        # materialising ``command`` here (independent of telemetry) we guarantee
        # the handoff value exists no matter what the cost-aggregate pull or the
        # CoordinatorReduceEvent emit does — see the cancellation rationale on
        # the telemetry block.
        command = Command(
            update={
                "apply_plan": output.apply_plan,
                "group_outcome": output.group_outcome,
                "step_result_candidate": output.step_result_candidate,
                # [codex R6 P1] Surface the reducer's diagnostics into the
                # subgraph state so main_graph / PR-7 audit persistence
                # have something to read (R5 lineage warnings, R4
                # needs_authorization_details, etc.).
                "reducer_diagnostics": output.diagnostics,
            },
            goto=END,
        )

        # C2 PR-8 §13 Task 8.4 + PR-9b-B Task B4 — emit CoordinatorReduceEvent.
        # BEST-EFFORT: try/except so a queue/serialization failure does not
        # mask the reducer output.
        #
        # [PR-9b-B Task B4] Cost authority is the durable cost_records ledger
        # (INV-B1), pulled via ``cost_rollup_service.aggregate(...)`` rather
        # than the worker envelope's per-child cost_summary attribute. The
        # previous silent-zero fallback (INV-B2 violation — coalescing a
        # missing/None reducer attribute into a fresh ``CostAggregate()``)
        # is removed. Both INV-B3 failure modes are LOUD (never a silent
        # zero), but they differ in what cost_total carries:
        #   • aggregate() RAISES → cost_total = CostAggregate() (zero) +
        #     diagnostics_summary 'cost_unavailable: aggregate raised'.
        #   • aggregate() SUCCEEDS with non-empty missing_children → the
        #     partial ledger cost is emitted AS-IS (the present children's
        #     cost IS authoritative) AND diagnostics_summary carries
        #     'cost_unavailable: missing_children=[...]'. Plan INV-B3 requires
        #     SURFACING the gap, not zeroing the partial (see the B4 test
        #     test_reducer_node_emit_missing_children_surfaces_diagnostic).
        event_queue: Optional[asyncio.Queue] = cfg.get("event_queue")
        if event_queue is not None:
            # [codex R4 P2] put_nowait (synchronous) — see _first_time_dispatch's
            # explanatory comment. Reducer has already completed reduce() before
            # this point; an awaited put_nowait could be cancelled before the
            # Command update propagates apply_plan/group_outcome/diagnostics.
            try:
                from app.domain.models.event import CoordinatorReduceEvent
                diagnostics_summary = "ok" if output.diagnostics else ""

                # INV-B1 cost-pull: derive cost_total from the ledger via
                # cost_rollup_service.aggregate(...) — NOT from the worker
                # envelope's cost_summary.
                cost_rollup_service = cfg.get("cost_rollup_service")
                # [PR-9b-B codex F2 — HIGH] Source the aggregate's child id
                # list from the FULL dispatched set (state["child_session_ids"],
                # work_unit_id -> child_session_id for every dispatched child),
                # NOT just the children that produced a WorkerResult. In an
                # INCOMPLETE-group / rehydrate-skipped-envelope scenario a child
                # that was dispatched but has no WorkerResult would otherwise be
                # excluded from BOTH the cost SUM and ``missing_children`` —
                # cost_total would undercount yet look authoritative with NO
                # cost_unavailable diagnostic (fail-open vs INV-B3). Driving the
                # id list off the dispatched set means such a child surfaces in
                # ``missing_children`` (it has no cost rows) → the
                # cost_unavailable diagnostic fires (INV-B3 satisfied).
                child_session_ids = list(
                    state.get("child_session_ids", {}).values()
                )
                cost_total: CostAggregate
                missing_children: tuple[str, ...] = ()
                if cost_rollup_service is None:
                    # No service wired → can't authoritatively compute cost.
                    # Surface as cost_unavailable rather than silently zeroing.
                    cost_total = CostAggregate()
                    diagnostics_summary = _append_diag(
                        diagnostics_summary,
                        "cost_unavailable: cost_rollup_service unavailable",
                    )
                else:
                    try:
                        agg_result = await cost_rollup_service.aggregate(
                            coordinator_run_id=coordinator_run_id,
                            child_session_ids=child_session_ids,
                        )
                        cost_total = agg_result.cost
                        missing_children = tuple(agg_result.missing_children)
                    except Exception as agg_exc:
                        logger.warning(
                            "reducer_node: cost aggregate(...) failed "
                            "run=%s: %s — surfacing cost_unavailable",
                            coordinator_run_id, agg_exc,
                        )
                        cost_total = CostAggregate()
                        missing_children = tuple(child_session_ids)
                        # [PR-9b-B codex F2 — MEDIUM/SEC] Emit a STABLE,
                        # non-sensitive code into the client-visible diagnostic.
                        # ``diagnostics_summary`` rides CoordinatorReduceEvent →
                        # EventMapper → SSE, so interpolating the raw exception
                        # (``{agg_exc}``) would leak SQLAlchemy text + bound
                        # params (session IDs). The detailed exception stays in
                        # the server-side logger.warning above ONLY.
                        diagnostics_summary = _append_diag(
                            diagnostics_summary,
                            "cost_unavailable: aggregate_error",
                        )

                # INV-B3: missing_children non-empty MUST surface even on
                # the aggregate-success path.
                if (
                    missing_children
                    and "cost_unavailable" not in diagnostics_summary
                ):
                    diagnostics_summary = _append_diag(
                        diagnostics_summary,
                        f"cost_unavailable: missing_children="
                        f"{list(missing_children)}",
                    )

                event_queue.put_nowait(CoordinatorReduceEvent(
                    group_outcome=output.group_outcome,
                    per_worker_outcomes={
                        wr.work_unit_id: wr.outcome
                        for wr in state["worker_results"]
                    },
                    diagnostics_summary=diagnostics_summary,
                    cost_total=cost_total,
                    conflict_paths=[],
                    coordinator_run_id=coordinator_run_id,
                    root_session_id=state.get("root_session_id"),
                    parent_session_id=state.get("parent_session_id"),
                ))
            except Exception as exc:
                logger.warning(
                    "reducer_node: emit CoordinatorReduceEvent failed "
                    "run=%s: %s", coordinator_run_id, exc,
                )

            # [C2b rollout WS1b §3.2/§3.3/§3.4] Run-level metrics — SEPARATE
            # try AFTER the reduce-event emit so a recorder exception can't
            # drop the load-bearing event. Gated on dispatch_started_monotonic:
            # set ONLY by _first_time_dispatch, so a crash+rehydrate (routes
            # _rehydrate_dispatch, no stamp) records nothing → no double-count
            # of the monotonic run_cost_usd counter. Reuse the cost_total +
            # diagnostics_summary computed above; do NOT re-aggregate.
            recorder = cfg.get("coordinator_metrics_recorder")
            start = state.get("dispatch_started_monotonic")
            if recorder is not None and start is not None:
                try:
                    cost_authoritative = "cost_unavailable" not in diagnostics_summary
                    recorder.record_run_terminal(
                        coordinator_run_id=coordinator_run_id,
                        user_id=state.get("user_id"),
                        cost_usd=cost_total.total_usd,
                        outcome=output.group_outcome.value,
                        cost_authoritative=cost_authoritative,
                        duration_s=time.monotonic() - start,
                    )
                except Exception:  # noqa: BLE001 — best-effort; never break reduce
                    logger.warning(
                        "reducer_node: record_run_terminal failed (best-effort) "
                        "run=%s", coordinator_run_id, exc_info=True,
                    )
            elif start is not None and recorder is None:
                # [R3 P3 dead-埋点 guard] A first-time run reached terminal
                # (metrics SHOULD fire) but the recorder is missing from cfg →
                # the _build_config threading was dropped, silently re-creating
                # the dead-instrument bug. Make it LOUD.
                logger.warning(
                    "reducer_node: dispatch_started_monotonic set but "
                    "coordinator_metrics_recorder absent from cfg — run-level "
                    "metrics NOT recorded (wiring regression?) run=%s",
                    coordinator_run_id,
                )

        return command
    finally:
        if (
            probe_quota is not None
            and user_id is not None
            and quota_acquired
        ):
            try:
                await probe_quota.release_coordinator_quotas(user_id=user_id)
            except BaseException:  # noqa: BLE001 — see comment below
                # [codex R3 P1-4] Same invariant as the dispatch
                # rollback path above: release failure inside ``finally``
                # MUST NOT mask the original control-flow signal — be it
                # a domain exception from ``reducer.reduce`` or a
                # ``CancelledError`` from an upstream task-group cancel.
                # An ``except Exception`` here would let
                # ``CancelledError`` from a cancelled
                # ``release_coordinator_quotas`` propagate, replacing the
                # original exception (or, on the happy path, hijacking
                # ``finally`` to abort an otherwise-returning node). Log
                # + swallow on ANY exit.
                logger.exception(
                    "reducer_node: probe_quota release failed user=%s "
                    "— continuing (original control flow preserved)",
                    user_id,
                )


# ── build ────────────────────────────────────────────────────────────────────


def build_parallel_execution_subgraph() -> Any:
    """Spec §7.3 -- compile with ``checkpointer=False`` (outer graph owns checkpoint)."""
    g: StateGraph = StateGraph(ParallelSubgraphState)
    g.add_node("dispatch_node", dispatch_node)
    g.add_node("worker_node", worker_node)
    g.add_node("reducer_node", reducer_node)
    g.add_edge(START, "dispatch_node")
    g.add_edge("worker_node", "reducer_node")
    return g.compile(checkpointer=False)
