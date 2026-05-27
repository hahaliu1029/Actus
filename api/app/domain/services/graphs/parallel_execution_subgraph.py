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
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from app.domain.models.mailbox_envelope import (
    CostAggregate,
    MailboxEnvelopeType,
    ResultReadyOutcome,
)
from app.domain.models.work_unit import PathLease, WorkUnit

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
    ) -> None:
        self.work_unit_id = work_unit_id
        self.child_session_id = child_session_id
        self.outcome = outcome
        self.cost_summary = cost_summary or CostAggregate()
        self.error_summary = error_summary
        self.summary = summary
        self.patch_manifest = patch_manifest
        self.needs_authorization_details = needs_authorization_details


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


# ── helpers ──────────────────────────────────────────────────────────────────


def _build_work_units_from_requests(
    work_unit_requests: list[Any],
    step_id_hash16: str,
    attempt_ix: int,
) -> list[WorkUnit]:
    """Convert planner WorkUnitRequest list to runtime WorkUnit list."""
    units: list[WorkUnit] = []
    for i, req in enumerate(work_unit_requests):
        units.append(
            WorkUnit(
                work_unit_id=f"{step_id_hash16}.a{attempt_ix}.{i}",
                objective=req.objective,
                phase=req.phase,
                allowed_tools=list(req.allowed_tools),
                write_lease=[
                    PathLease(path=p.path, op=p.op) for p in req.proposed_paths
                ],
                expected_result_schema=req.expected_result_schema,
            )
        )
    return units


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
    """Minimal JSON serialization of WorkUnit for MinIO manifest."""
    return json.dumps({
        "work_unit_id": wu.work_unit_id,
        "objective": wu.objective,
        "phase": wu.phase,
        "allowed_tools": list(wu.allowed_tools),
        "write_lease": [lease.model_dump() for lease in wu.write_lease],
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
        existing = await rehydrate_service.detect_existing_run(
            coordinator_run_id=candidate_run_id,
            parent_session_id=parent_session_id,
        )
        if existing is not None:
            work_units = _build_work_units_from_requests(
                state["work_unit_requests"], step_id_hash16, current_attempt_ix,
            )
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
    return await _first_time_dispatch(state, config, coordinator_run_id, work_units)


async def _first_time_dispatch(
    state: ParallelSubgraphState,
    config: dict,
    coordinator_run_id: str,
    work_units: list[WorkUnit],
) -> Command:
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

    # TODO(PR-7 wiring): §14.3 #2 per-user daily cost cap — call
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
                tool_filter_preset="coordinator_step",
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

        # Step 7 -- start N runner tasks.
        # r4 P1-2: pass ``parent_session_id`` so PR-4's runner finalizers
        # (``_publish_result_ready`` / ``_publish_cancel_ack``) can build envelopes
        # without parsing the coordinator_run_id string. The runner ctor already
        # accepts ``parent_session_id`` at the skeleton level (Task 3.4).
        for wu in enriched_units:
            await runner_starter.start(
                coordinator_run_id=coordinator_run_id,
                work_unit=wu,
                child_session_id=child_session_ids[wu.work_unit_id],
                spawn_manifest_ref=spawn_manifest_refs[wu.work_unit_id],
                cancel_event=cancel_event,
                root_session_id=root_session_id,
                parent_session_id=parent_session_id,
            )

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
        orchestrator = orchestrator_factory.build(
            coordinator_run_id=coordinator_run_id,
            root_session_id=root_session_id,
            parent_session_id=parent_session_id,
        )
        orchestrator_task = asyncio.create_task(orchestrator.run(
            coordinator_run_id=coordinator_run_id,
            root_session_id=root_session_id,
            work_units_pending=[wu.work_unit_id for wu in enriched_units],
            child_session_ids=child_session_ids,
            cancel_event=cancel_event,
        ))
        # r6 P1-1: attach done_callback to surface unhandled orchestrator
        # exceptions (e.g., r5 P1-2 "all CANCEL_REQUEST publishes failed" raise).
        # The task is fire-and-forget — without this callback its exception would
        # be silently swallowed by asyncio (no one awaits it; subgraph state is
        # discarded by _run_parallel_backend). PR-6 will replace this with a
        # supervised orchestrator lifecycle.
        orchestrator_task.add_done_callback(_log_orchestrator_task_done)
    except BaseException:
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

        raise

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
        },
        goto=[
            Send("worker_node", {
                "work_unit_id": wu.work_unit_id,
                "child_session_id": child_session_ids[wu.work_unit_id],
                "coordinator_run_id": coordinator_run_id,
                "root_session_id": root_session_id,
            })
            for wu in enriched_units
        ],
    )


async def _rehydrate_dispatch(
    state: ParallelSubgraphState,
    config: dict,
    existing: Any,
    coordinator_run_id: str,
    work_units: list[WorkUnit],
) -> Command:
    """Crash-recovery rehydrate -- reuse existing child sessions; PR-7 expands.

    r3 P1-1: must pre-create the waiter consumer group for each pending child
    BEFORE returning ``Send`` to worker_node. After a pod restart the waiter
    group from the previous incarnation is gone (consumer groups are NOT
    persisted by Redis Streams when the supervisor recreates the stream);
    relying on the lazy ``waiter.subscribe`` inside ``worker_node.await_terminal``
    re-opens the fast-publish race because the in-flight terminal envelope
    would be missed when XGROUP CREATE id=$ excludes already-buffered messages.
    """
    cfg = config["configurable"]
    subscriber = cfg["mailbox_subscriber"]  # fail-fast on missing DI [r2 P1-2]
    root_session_id = state["root_session_id"]
    for wu_id in existing.pending:
        child_sid = existing.child_session_ids[wu_id]
        # r4 P1-1: rehydrate path uses ``start_id="0"`` so the new consumer
        # group sees terminal envelopes already buffered in the stream BEFORE
        # this subscribe (the race window between
        # ``rehydrate_service.detect_existing_run`` and the new subscribe
        # call). first-time dispatch stays on the default ``"$"`` because
        # the child hasn't published anything yet.
        await subscriber.subscribe(
            stream_key=f"actus:child:{root_session_id}:mailbox",
            consumer_group=f"coordinator:waiter:{child_sid}",
            consumer_name=f"waiter-{child_sid}",
            start_id="0",
        )
    return Command(
        update={
            "coordinator_run_id": coordinator_run_id,
            "work_units": work_units,
            "child_session_ids": existing.child_session_ids,
        },
        goto=[
            Send("worker_node", {
                "work_unit_id": wu_id,
                "child_session_id": existing.child_session_ids[wu_id],
                "coordinator_run_id": coordinator_run_id,
                "root_session_id": root_session_id,
            })
            for wu_id in existing.pending
        ],
    )


# ── worker_node ──────────────────────────────────────────────────────────────


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
    envelope = await waiter.await_terminal(
        child_session_id=state_per_send["child_session_id"],
        root_session_id=state_per_send["root_session_id"],
        cancel_event=cancel_event,
        coordinator_run_id=state_per_send.get("coordinator_run_id"),
    )

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
        patch_manifest = rr_payload.patch_manifest
        needs_authorization_details = rr_payload.needs_authorization_details
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
        return Command(
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
