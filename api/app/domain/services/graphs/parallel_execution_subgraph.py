"""C2 PR-3 §7.3 + §7.5 — parallel_execution_subgraph.

LangGraph StateGraph compiled with ``checkpointer=False``.

Topology::

    START -> dispatch_node -> Send x N -> worker_node -> reducer_node (placeholder) -> END

PR-3 ships:
  - ``dispatch_node`` -- peek/bump coordinator attempt, build runtime WorkUnits,
    create N child sessions, upload SpawnManifest x N, start N runner tasks,
    publish SPAWN_REQUEST x N, launch orchestrator, fan-out via ``Send`` x N
  - ``worker_node`` -- await terminal envelope per child via
    ``CoordinatorTerminalEnvelopeWaiter``, normalize ``CANCEL_ACK`` final_state
    to ``ResultReadyOutcome``
  - ``reducer_node_placeholder`` -- PR-5 wires real PatchReducerService

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
    """PR-3 minimal carrier; PR-4 promotes to Pydantic BaseModel + patch_manifest."""

    def __init__(
        self,
        *,
        work_unit_id: str,
        child_session_id: str,
        outcome: ResultReadyOutcome,
        cost_summary: Optional[CostAggregate] = None,
        error_summary: Optional[str] = None,
    ) -> None:
        self.work_unit_id = work_unit_id
        self.child_session_id = child_session_id
        self.outcome = outcome
        self.cost_summary = cost_summary or CostAggregate()
        self.error_summary = error_summary
        self.patch_manifest = None  # PR-4
        self.needs_authorization_details = None  # PR-4


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
        await subscriber.subscribe(
            stream_key=f"actus:child:{root_session_id}:mailbox",
            consumer_group=f"coordinator:waiter:{child_sid}",
            consumer_name=f"waiter-{child_sid}",
        )

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

    # Step 10 -- fan out via Send x N.
    return Command(
        update={
            "coordinator_run_id": coordinator_run_id,
            "work_units": enriched_units,
            "child_session_ids": child_session_ids,
            "orchestrator_task": orchestrator_task,
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

    envelope = await waiter.await_terminal(
        child_session_id=state_per_send["child_session_id"],
        root_session_id=state_per_send["root_session_id"],
        cancel_event=cancel_event,
    )

    if envelope.type == MailboxEnvelopeType.RESULT_READY:
        raw_outcome = (
            envelope.payload.get("outcome")
            if isinstance(envelope.payload, dict)
            else getattr(envelope.payload, "outcome", None)
        )
        outcome = (
            ResultReadyOutcome(raw_outcome) if isinstance(raw_outcome, str)
            else (raw_outcome or ResultReadyOutcome.FAILED)
        )
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
    else:
        outcome = ResultReadyOutcome.FAILED

    result = WorkerResult(
        work_unit_id=state_per_send["work_unit_id"],
        child_session_id=state_per_send["child_session_id"],
        outcome=outcome,
    )
    return {"worker_results": [result]}


# ── reducer_node placeholder ─────────────────────────────────────────────────


async def reducer_node_placeholder(state: ParallelSubgraphState, config: RunnableConfig) -> Command:
    """PR-3 placeholder; PR-5 wires PatchReducerService."""
    return Command(
        update={
            "step_result_candidate": "[PR-3 placeholder] reducer not yet wired",
            "group_outcome": None,
        },
        goto=END,
    )


# ── build ────────────────────────────────────────────────────────────────────


def build_parallel_execution_subgraph() -> Any:
    """Spec §7.3 -- compile with ``checkpointer=False`` (outer graph owns checkpoint)."""
    g: StateGraph = StateGraph(ParallelSubgraphState)
    g.add_node("dispatch_node", dispatch_node)
    g.add_node("worker_node", worker_node)
    g.add_node("reducer_node", reducer_node_placeholder)
    g.add_edge(START, "dispatch_node")
    g.add_edge("worker_node", "reducer_node")
    return g.compile(checkpointer=False)
