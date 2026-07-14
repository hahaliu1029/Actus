"""C3 MailboxSupervisor (spec §6.1/§6.3/§6.4/§6.5).

Per-root supervisor task. Single owner of terminal sandbox transitions for
this root session (M1 invariant). PR-3a ships the skeleton + dispatch table +
main loop with stub handlers; PR-3b adds reliability (XAUTOCLAIM/poison/
crash-recovery); PR-3c integrates into agent_task_runner lifecycle; PR-4
swaps stub handlers for real destroy hooks.

Layer note: this module lives in ``application/`` (not ``domain/``) because
it directly depends on ``redis.asyncio`` and the infrastructure consumer
wrapper. Domain models (``MailboxEnvelope``, audit Protocol) are still imported
from ``domain/`` — only the orchestration is application-layer.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Protocol

from redis.asyncio import Redis

from app.application.services.coordinator_terminal_transition import (
    CoordinatorTerminalCommand,
    ExpectedCoordinatorLineage,
)

if TYPE_CHECKING:  # pragma: no cover — type-only to avoid runtime import cycle
    # [C2 PR-6 §14.4] CostRollupService is consumed by ``ResultReadyHandler``
    # via ``SupervisorContext.cost_rollup_service``. Kept under TYPE_CHECKING
    # so the supervisor module stays importable from the cost rollup module
    # if the rollup impl ever needs to grow supervisor-aware helpers.
    from app.application.services.cost_rollup_service import CostRollupService
    from app.application.services.coordinator_liveness_lease_service import (
        CoordinatorLivenessLeaseService,
    )
    # [C2 PR-7 §12.4] coordinator_envelope_store — Optional in SupervisorContext.
    # Late import here keeps the application-layer module-load path lean and
    # mirrors the cost_rollup_service guard pattern.
    from app.domain.repositories.coordinator_result_envelope_store_repository import (
        CoordinatorResultEnvelopeStoreRepository,
    )

from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxLifecycleError,
)
from app.domain.external.mailbox_publisher import MailboxPublisher
from app.domain.models.mailbox_envelope import (
    APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
    CANCEL_AUTO_ESCALATE_TO_TERMINATE,
    CHILD_CANCEL_ACK_TIMEOUT_MS,
    CHILD_TO_PARENT_TYPES,
    MAILBOX_PEL_IDLE_MS_FOR_CLAIM,
    MAILBOX_POISON_MAX_RECLAIM,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS,
    MAILBOX_XREADGROUP_BLOCK_MS,
    MAILBOX_XREADGROUP_COUNT,
    SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
    ApprovalDecidedBy,
    ApprovalResponsePayload,
    CancelAckPayload,
    CancelPolicy,
    CancelRequestPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ProgressKind,
    ResultReadyPayload,
)
from app.domain.models.session import DestroyReason
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.repositories.mailbox_envelope_audit_repository import (
    MailboxEnvelopeAuditRepository,
)
from app.domain.repositories.session_repository import SessionRepository
from app.domain.repositories.subagent_run_repository import SubagentRunRepository
from app.infrastructure.external.mailbox.redis_mailbox_consumer import (
    RedisMailboxConsumer,
)

# Re-export for tests that monkeypatch via the supervisor module path
# (e.g. ``monkeypatch.setattr(ms, "CANCEL_AUTO_ESCALATE_TO_TERMINATE", False)``).
__all__ = [
    "APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS",
    "ApprovalRequestHandler",
    "CANCEL_AUTO_ESCALATE_TO_TERMINATE",
    "CancelAckHandler",
    "CancelRequestHandler",
    "HandlerOutcome",
    "HandoffRequestHandler",
    "MailboxSupervisor",
    "ResultReadyHandler",
    "SupervisorContext",
    "build_default_dispatch_table",
]


logger = logging.getLogger(__name__)


# C3 PR-4 — audit table caps ``envelope_id`` at ``String(64)``
# (``infrastructure/models/mailbox_envelope_audit.py:47``). Every supervisor-
# synthesised envelope_id (synthetic CANCEL_ACK echo, APPROVAL_RESPONSE deny,
# cascade re-publishes) MUST stay inside this bound. Helper hashes the
# composition key so the result is stable for repeated inputs (e.g.
# redelivery of the same incoming envelope produces the same synthetic id —
# the audit-repo's UNIQUE constraint catches dedup naturally).
#
# Format: ``{tag}:{sha256(key)[:32]}`` — 32-hex digest + tag + colon.
# - ``tag="ack"`` → 36 chars; ``tag="deny"`` → 37 chars; both ≤ 64.
# - sha256 truncated to 128 bits is collision-resistant for the supervisor's
#   single-instance synthetic-envelope workload (millions of envelopes is
#   safely below the birthday bound).
#
# codex r3 [R3-7, HIGH CONTRACT] — the previous format
# ``{envelope.envelope_id}:ack`` could overflow 64 when the incoming
# envelope_id was at or near the limit (e.g. a future producer using the
# full ULID + suffix).
_SYNTHETIC_DIGEST_BYTES: int = 32


def _synthetic_envelope_id(tag: str, key: str) -> str:
    """Compose a stable, length-bounded envelope_id for synthetic envelopes.

    ``tag`` is a short human-readable prefix (``"ack"``, ``"deny"``).
    ``key`` is the composition input (typically the originating envelope_id
    plus any disambiguator); the function hashes it so the result fits
    inside the audit table's ``envelope_id`` ``String(64)`` column even when
    the upstream envelope_id is at the 64-char limit.
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:_SYNTHETIC_DIGEST_BYTES]
    return f"{tag}:{digest}"


class _SandboxLifecycleProtocol(Protocol):
    async def destroy(self, session_id: str, reason) -> None: ...


class _TelemetryProtocol(Protocol):
    async def emit(self, name: str, data: dict) -> None: ...


class _CascadeFailedError(Exception):
    """codex r7 [R7-7, HIGH CONTRACT] — raised by
    ``_emit_cascade_terminate`` when the XADD publish failed *and* the
    direct-kill fallback also failed with a retryable
    ``SandboxLifecycleError``. Callers (orphan tick / cancel
    auto-escalate / poison drop) catch this to skip per-child tracking
    cleanup so a subsequent supervisor tick retries the cascade.

    Non-retryable terminal cases (``SandboxAlreadyDestroyed``,
    ``SandboxBindingMissing``) do NOT raise this — the child is already
    gone from the lifecycle service's POV, so retrying would burn
    cycles with no chance of success. They are "success-equivalent"
    from a cascade POV and let the caller proceed to clear tracking.
    """


@dataclass
class SupervisorContext:
    """Handler injection bag — supervisor passes this to every handler.

    Handlers MUST treat ``ctx`` as read-only **at handler invocation time** —
    there is no per-call ``ctx`` cloning. State that must be observed across
    handler invocations belongs on ``audit_repo`` (durable) or on the
    per-supervisor instance via a dedicated dependency.

    One narrow exception (PR-4): ``register_cancel_state`` is a hook bound
    once by :meth:`MailboxSupervisor.__init__` so the ``CancelRequestHandler``
    can stash REQUEST_CANCEL cascade state for the auto-escalate tick to
    promote later (spec §8). The supervisor mutates ``ctx`` *exactly once*
    at construction; handlers only *call* the hook.
    """

    root_session_id: str
    pod_id: str
    instance_id: str
    redis: Redis
    audit_repo: MailboxEnvelopeAuditRepository
    publisher: MailboxPublisher
    sandbox_lifecycle: _SandboxLifecycleProtocol
    agent_service_callback: Callable[[MailboxEnvelope], Awaitable[None]]
    telemetry: _TelemetryProtocol
    clock: Callable[[], float] = time.monotonic
    register_cancel_state: Optional[
        Callable[[str, CancelPolicy, float], Awaitable[None]]
    ] = None
    # Codex F7+F9 (HIGH) — synchronous hook so terminal handlers can drop
    # ``_cancel_states[child]`` + ``_last_seen_mono[child]`` after a child
    # is destroyed. Without this cleanup the auto-escalate tick and orphan
    # detector fire spurious CANCEL_REQUEST(TERMINATE) cascades for an
    # already-dead child. Bound once in
    # :meth:`MailboxSupervisor.__init__` (same pattern as
    # ``register_cancel_state``).
    clear_child_tracking: Optional[Callable[[str], None]] = None
    # Task 4 terminal-ownership port. The typed command carries only expected
    # envelope/context lineage; the composition adapter obtains authority via a
    # locked row read and performs validation + SSM CAS in one short-lived UoW.
    # Optional keeps legacy/direct-test contexts backward compatible; production
    # always wires it.
    terminalize_child: Optional[
        Callable[[CoordinatorTerminalCommand], Awaitable[bool]]
    ] = None
    # codex r6 [R6-2, HIGH CONTRACT] — supervisor-private side-table for
    # threading an explicit ``DestroyReason`` from orphan/poison cascades
    # to ``CancelRequestHandler._terminate_outcome``. Keyed by the
    # synthetic envelope_id of the cascade-emitted CANCEL_REQUEST; the
    # publisher populates the entry BEFORE ``publisher.publish`` so the
    # handler can read on dispatch. Replaces the previous
    # ``CancelRequestPayload.destroy_reason`` wire field (R2-6/R3-6),
    # which broke the frozen schema and required a producer_role guard
    # to defuse hostile overrides. The dict is shared by reference
    # between the supervisor and its context so the handler reads from
    # the same backing store the publisher writes to.
    #
    # codex r7 [R7-4, HIGH CONTRACT] — pop happens AFTER side_effect's
    # destroy + publish_ack succeed (not before). On PEL retry the
    # override is still present and the same ``DestroyReason`` is
    # re-applied. See ``CancelRequestHandler._terminate_outcome``.
    #
    # codex r7 [R7-5, HIGH CONTRACT — DOCUMENTED DRIFT] — this is an
    # in-process Python dict; it does NOT survive supervisor restart.
    # If the supervisor crashes between publishing the synthetic
    # CANCEL_REQUEST and the handler consuming it, the restarting
    # supervisor reads the durable envelope from Redis but its empty
    # side-table → the handler falls back to ``DestroyReason
    # .FORCE_TERMINATE`` instead of the original ``ORPHAN_TIMEOUT``
    # (or any other override). This is an **accepted degradation**:
    # ``ORPHAN_TIMEOUT`` vs ``FORCE_TERMINATE`` is a *reason annotation*
    # — both result in the same kill action against the sandbox. The
    # destroy still fires (the cascade envelope is durable in the
    # stream); only the audit reason field is approximate post-restart.
    # The alternative (encoding the override into the envelope_id /
    # adding a producer_role variant) re-introduces a wire-format
    # channel that the spec freeze explicitly forbids and that earlier
    # rounds (R6-2) intentionally removed. Test coverage for the
    # fallback path lives at
    # ``test_mailbox_supervisor.py::test_cascade_override_lost_after_supervisor_restart_falls_back_to_force_terminate``.
    cascade_destroy_overrides: dict = field(default_factory=dict)
    # C3 PR-5 (spec §11.6 rollback runbook + R1 P2.2) — optional plumbing
    # for ``MailboxSupervisor._check_should_stop_for_rollback``.
    #
    # ``session_repo`` is the read-only session repository the supervisor
    # uses to detect "all my subagent children are now legacy-plane" — that
    # condition fires after the operator runs the §11.6 rollback SQL
    # (UPDATE sessions SET subagent_control_plane='legacy' WHERE
    # root_session_id=... AND subagent_control_plane='mailbox'). Optional
    # because PR-3a/3b/3c-era unit tests construct contexts without DI;
    # the rollback-check method early-returns when None.
    #
    # ``stop_self_callback`` is the registry-injected hook for
    # ``_check_should_stop_for_rollback`` to pop this supervisor's slot from
    # ``SupervisorRegistry._slots`` AND cancel its run task in one shot.
    # ``MailboxSupervisor.stop()`` alone only signals the run loop to exit —
    # the slot stays in ``_slots`` (showing as "crashed" in ``health_check``)
    # until ``registry.stop(root_session_id)`` is invoked. The registry
    # binds this callback at ``spawn`` time (same pattern as
    # ``register_cancel_state``); pre-PR-5 paths that never bind leave the
    # supervisor to ``self.stop()`` only — backward-compatible.
    session_repo: Optional[SessionRepository] = None
    stop_self_callback: Optional[Callable[[], Awaitable[None]]] = None
    # [C2 PR-6 §14.4] Optional cost rollup hook. ``None`` on legacy /
    # non-coordinator supervisor instances; populated by service_dependencies
    # wiring when the coordinator feature is enabled. ``ResultReadyHandler``
    # fires ``rollup_to_parent`` as a best-effort PROLOGUE in its
    # ``_side_effect`` for children whose ``Session.tool_filter_preset ==
    # "coordinator_step"`` (gate also requires ``session_repo`` so the
    # handler can look the child session up). A rollup failure MUST NOT
    # abort destroy + audit safety operations; the handler swallows
    # exceptions after logging.
    cost_rollup_service: Optional["CostRollupService"] = None
    # [C2 PR-7 §12.4] Optional coordinator result envelope persistence hook.
    # ``None`` on legacy / non-coordinator supervisor instances; populated by
    # ``service_dependencies`` wiring when the coordinator feature is enabled.
    # Both ``ResultReadyHandler`` and ``CancelAckHandler`` fire
    # ``persist_terminal`` as a best-effort PROLOGUE for children whose
    # ``Session.tool_filter_preset == "coordinator_step"``. A persist failure
    # MUST NOT abort destroy + audit safety operations; the handlers swallow
    # exceptions after logging — the run is still recoverable from the live
    # session row even if the envelope row was lost (worst case: a duplicate
    # work-unit re-spawn on the next rehydrate).
    coordinator_envelope_store: Optional[
        "CoordinatorResultEnvelopeStoreRepository"
    ] = None
    # [C4.1a §5.1] Optional subagent-run observation sink. ``None`` on legacy /
    # flag-OFF contexts (repo-or-None at the composition root). ``ResultReadyHandler``
    # fires a THIRD independent best-effort PROLOGUE for ``coordinator_step``
    # children (own gate / own get_by_id fetch / own try-except — does NOT reuse
    # the cost-rollup or persist-terminal fetch, mirroring their independence).
    # A record failure MUST NOT abort destroy + audit; the handler swallows.
    subagent_run_repo: Optional[SubagentRunRepository] = None
    # Task 6 durable coordinator-child liveness. Optional for legacy/direct
    # tests; production composition always supplies the shared Redis service.
    liveness_service: Optional["CoordinatorLivenessLeaseService"] = None

    def now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    # ── C2 PR-8 §13.4 lineage helpers (used by CoordinatorProgressUpdateHandler) ──
    #
    # All three methods are instance methods on the dataclass (NOT injected
    # Callables) so legacy SupervisorContext instantiation sites — which
    # don't know about PR-8 — get the default behaviour for free without
    # widening the dataclass field count. Override callers (e.g. PR-9
    # service_dependencies once it wants to emit telemetry on the wrap path)
    # can rebind on the instance with ``ctx.relay_progress_with_lineage = ...``
    # because the dataclass isn't ``frozen``.

    # Bounded walk depth — guards against pathological cycles or
    # mis-rehydrated session rows that point parent → self. The C1a session
    # tree is structurally shallow (root → planner → coordinator children),
    # so 32 hops is two orders of magnitude over any legitimate depth.
    _COMPUTE_ROOT_MAX_HOPS: int = 32

    async def compute_root_session_id(self, child_session) -> Optional[str]:
        """Walk ``parent_session_id`` to the top of the session tree.

        Returns the root session id (the deepest ancestor with no parent).
        If ``child_session.parent_session_id`` is None, returns
        ``child_session.id`` (the child is its own root). If the walk breaks
        mid-chain (the next parent row is missing from ``session_repo`` —
        could be a deleted ancestor on a long-running orphan), returns the
        last known ancestor id as a best-effort lineage anchor; this is a
        stable identifier that downstream consumers can still group by.

        The depth cap (``_COMPUTE_ROOT_MAX_HOPS``) defuses any cycle by
        returning whatever the cursor landed on when the cap was hit. The
        result is ``Optional[str]`` to match the
        :class:`CoordinatorLineageMixin` field type — emitting events with
        ``root_session_id=None`` is acceptable when the supervisor
        genuinely cannot reconstruct lineage.
        """
        if child_session.parent_session_id is None:
            return child_session.id
        if self.session_repo is None:
            return child_session.id

        current_id: Optional[str] = child_session.id
        next_parent_id: Optional[str] = child_session.parent_session_id
        for _ in range(self._COMPUTE_ROOT_MAX_HOPS):
            if next_parent_id is None:
                return current_id
            parent = await self.session_repo.get_by_id(next_parent_id)
            if parent is None:
                # Walk broke — the next_parent_id is the last known anchor
                # in the lineage; return it rather than the deeper current_id
                # so downstream consumers see the actual chain endpoint.
                return next_parent_id
            current_id = next_parent_id
            next_parent_id = parent.parent_session_id
        # Cap exceeded — return whatever ancestor we last landed on. Logged
        # because hitting the cap indicates a pathological session tree
        # (cycle or extreme depth) and should be investigated.
        logger.warning(
            "compute_root_session_id hit walk cap (%d) starting from %s — "
            "returning best-effort cursor %s",
            self._COMPUTE_ROOT_MAX_HOPS,
            child_session.id,
            current_id,
        )
        return current_id

    async def relay_progress_default(self, envelope: "MailboxEnvelope") -> None:
        """Forward PROGRESS_UPDATE unchanged — the legacy stub path.

        Identical behaviour to ``_StubNonTerminalHandler.handle``'s body so
        non-coordinator children keep their pre-C2 fan-out shape.
        """
        await self.agent_service_callback(envelope)

    async def relay_progress_with_lineage(
        self,
        envelope: "MailboxEnvelope",
        lineage: dict,
    ) -> None:
        """Forward PROGRESS_UPDATE with coordinator lineage injected.

        :class:`MailboxEnvelope` is a frozen Pydantic model, so we cannot
        mutate ``envelope.payload`` in place. Build a new envelope with
        ``payload = {**old, "_lineage": lineage}`` and forward that. The
        downstream consumer (``agent_service_callback`` → SSE bridge → PR-8
        ``coordinator_progress_update`` event emit, wired in Task 8.4) reads
        ``payload["_lineage"]`` to populate the
        :class:`CoordinatorLineageMixin` fields on the emitted event.

        The ``_lineage`` prefix (underscore) marks this as an in-band
        supervisor-injected hint; the wire-form
        :class:`ProgressUpdatePayload` does NOT carry it on the producer
        side — only the supervisor adds it on the consume path. Validation
        is bypassed for the rebuilt envelope (the original validated the
        payload via ``model_validator`` already; we just extend the dict).
        """
        decorated_payload = {**envelope.payload, "_lineage": lineage}
        # The Pydantic model_validator on MailboxEnvelope rejects unknown
        # payload keys when re-validating against the PROGRESS_UPDATE schema
        # (ProgressUpdatePayload has ``extra="forbid"``). Use
        # ``model_construct`` to skip re-validation — the original envelope
        # already passed validation, and we are deliberately adding an
        # out-of-band supervisor hint that the downstream consumer reads.
        forwarded = MailboxEnvelope.model_construct(
            envelope_id=envelope.envelope_id,
            type=envelope.type,
            parent_session_id=envelope.parent_session_id,
            child_session_id=envelope.child_session_id,
            correlation_id=envelope.correlation_id,
            emitted_at=envelope.emitted_at,
            producer_role=envelope.producer_role,
            payload=decorated_payload,
            reclaim_count=envelope.reclaim_count,
        )
        await self.agent_service_callback(forwarded)


@dataclass
class HandlerOutcome:
    """Handler return shape — drives ACK / PEL retention + post-ACK hooks.

    Per spec §6.x (and codex r2 [P1] fix — prior wording inverted the spec):

    - ``ack=True`` + ``side_effect=None`` → ACK immediately. Canonical for
      non-terminal handlers (PROGRESS_UPDATE/SPAWN_ACK/...) that forward to
      ``ctx.agent_service_callback`` and have no destructive follow-up work.
    - ``ack=False`` + ``side_effect`` set → run side_effect first.
        - side_effect completes normally → ACK (terminal handlers in PR-4
          return this shape: destroy() runs, success → ACK).
        - side_effect raises → leave in PEL (no ACK); PR-3b XAUTOCLAIM
          retries, reclaim_count > MAILBOX_POISON_MAX_RECLAIM → poison drop.
    - ``ack=False`` + ``side_effect=None`` → leave in PEL deliberately
      (explicit defer; PR-3b reliability layer handles).
    - ``ack=True`` + ``side_effect`` set → ACK after side_effect normal
      completion; same PEL-retain on raise as the canonical ``ack=False``
      shape (functionally equivalent, but encourages callers to express
      intent via ``ack=False`` when destruction is the gating event).

    The ACK-or-not branch is in :meth:`MailboxSupervisor._handle_envelope`.
    """

    ack: bool
    side_effect: Optional[Callable[[], Awaitable[None]]] = None
    audit_payload: dict = field(default_factory=dict)


class EnvelopeHandler(Protocol):
    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome: ...


@dataclass
class _CancelState:
    """In-memory CancelRequest tracking — populated by PR-4 cancel handler,
    consumed by PR-4 auto-escalate tick (spec §7.5). PR-3b ships the store
    so the supervisor instance carries the per-child cancel state across
    XREADGROUP iterations; PR-4 wires the periodic tick that promotes
    REQUEST_CANCEL → TERMINATE after CHILD_CANCEL_ACK_TIMEOUT_MS.
    """

    child_session_id: str
    policy: CancelPolicy
    requested_at_mono: float


# ─── PR-3a non-terminal stub (PR-4 keeps this for SPAWN/PROGRESS/etc.) ────────


class _StubNonTerminalHandler:
    """PR-3a placeholder for non-terminal envelopes (PROGRESS_UPDATE,
    SPAWN_REQUEST, ...). Forwards to the in-process agent_service callback
    and ACKs. PR-4 will diverge per-type with real business logic."""

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        await ctx.agent_service_callback(envelope)
        return HandlerOutcome(ack=True, audit_payload={"stub": True})


# ─── PR-4 terminal + cascade handlers (spec §7.3-§7.6 + §10.2 + §6.6) ─────────


async def _terminalize_child_from_envelope(
    envelope: MailboxEnvelope,
    ctx: SupervisorContext,
) -> bool:
    """Apply the existing mapping only for an authoritative coordinator row.

    This is the first load-bearing operation in terminal handler side effects.
    Live legacy child terminals use the stable ``spawn:<child_id>`` correlation
    contract and already own their row transition; that negative cheap gate
    preserves their pre-Task4 cleanup/ACK path without adding a DB dependency.

    Every other terminal is a coordinator candidate. The terminalizer owns the
    locked row read, complete lineage revalidation, SSM CAS and commit in one
    transaction. Missing/mismatched rows refuse only the new terminal DB write
    and deliberately leave the pre-existing sandbox/tracking/callback cleanup
    trust surface unchanged. Lock/read/CAS/commit failures propagate to
    ``_handle_envelope``, retaining Redis PEL and tracking for retry.
    """
    terminalize = ctx.terminalize_child
    if terminalize is None:
        return False

    # AgentTaskRunner is the sole live producer of legacy research/general
    # terminal envelopes and fixes this correlation in its constructor. This is
    # a negative gate only: every candidate allowed past it still needs the
    # authoritative row checks below; payload/child-id alone never grant DB
    # terminal authority.
    if envelope.correlation_id == f"spawn:{envelope.child_session_id}":
        return False

    from app.application.services.child_terminal_reconciler import (
        row_terminal_from_envelope,
    )

    payload = envelope.payload if isinstance(envelope.payload, dict) else {}
    status, reason = row_terminal_from_envelope(envelope.type.value, payload)
    transitioned = await terminalize(
        CoordinatorTerminalCommand(
            lineage=ExpectedCoordinatorLineage(
                child_session_id=envelope.child_session_id,
                parent_session_id=envelope.parent_session_id,
                root_session_id=ctx.root_session_id,
                coordinator_run_id=envelope.correlation_id,
            ),
            status=status,
            reason=reason,
        )
    )
    if transitioned:
        return True

    # A PEL replay after the first transition legitimately loses the SSM CAS.
    # Verify the now-terminal row before allowing the replay to recreate the
    # Redis tombstone. A mismatched/forged envelope remains a safe no-op.
    if ctx.session_repo is None:
        return False
    row = await ctx.session_repo.get_by_id(envelope.child_session_id)
    return bool(
        row is not None
        and row.id == envelope.child_session_id
        and row.worker_type == "subagent"
        and row.subagent_control_plane == "mailbox"
        and row.tool_filter_preset == COORDINATOR_STEP_PRESET
        and row.parent_session_id == envelope.parent_session_id
        and envelope.parent_session_id == ctx.root_session_id
        and row.root_session_id == ctx.root_session_id
        and row.coordinator_run_id == envelope.correlation_id
        and bool(row.work_unit_id)
        and getattr(row.status, "value", row.status) in {"completed", "timed_out"}
    )


async def _mark_terminal_liveness(
    envelope: MailboxEnvelope,
    ctx: SupervisorContext,
) -> None:
    if await _terminalize_child_from_envelope(envelope, ctx):
        if ctx.liveness_service is None:
            return
        # Load-bearing: failure retains the envelope in PEL and deliberately
        # happens before sandbox/tracking cleanup. A replay re-verifies the
        # terminal DB row above and retries this idempotent tombstone write.
        await ctx.liveness_service.mark_terminal(envelope.child_session_id)


async def _resolve_terminate_echo_correlation_id(
    envelope: MailboxEnvelope,
    ctx: SupervisorContext,
) -> str:
    """Resolve authoritative run lineage for an internal coordinator cascade.

    Internal cascade requests deliberately keep their stable ``cascade:*``
    correlation for request audit/idempotency. Their synthetic CANCEL_ACK echo,
    however, is the terminal event consumed by Task 4 and must carry the current
    coordinator run id. Legacy/non-coordinator rows retain the old echo contract.
    """
    is_internal_cascade = (
        envelope.producer_role == ProducerRole.SUPERVISOR
        and envelope.correlation_id.startswith("cascade:")
    )
    if not is_internal_cascade or ctx.terminalize_child is None:
        return envelope.correlation_id

    session_repo = ctx.session_repo
    if session_repo is None:
        raise RuntimeError(
            "coordinator cascade echo requires authoritative session reader"
        )
    try:
        row = await session_repo.get_by_id(envelope.child_session_id)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "coordinator cascade echo authoritative session read failed"
        ) from exc

    if row is None:
        raise RuntimeError(
            "coordinator cascade echo authoritative session row missing"
        )

    # Coordinator lineage columns are never populated for legacy/research
    # children. Any surviving coordinator marker makes this a fail-closed
    # candidate even if a concurrent control-plane/preset flip has occurred.
    coordinator_candidate = (
        getattr(row, "tool_filter_preset", None) == COORDINATOR_STEP_PRESET
        or bool(getattr(row, "coordinator_run_id", None))
        or bool(getattr(row, "work_unit_id", None))
    )
    if not coordinator_candidate:
        return envelope.correlation_id

    authoritative = (
        getattr(row, "id", None) == envelope.child_session_id
        and getattr(row, "worker_type", None) == "subagent"
        and getattr(row, "subagent_control_plane", None) == "mailbox"
        and getattr(row, "tool_filter_preset", None) == COORDINATOR_STEP_PRESET
        and getattr(row, "parent_session_id", None) == envelope.parent_session_id
        and envelope.parent_session_id == ctx.root_session_id
        and getattr(row, "root_session_id", None) == ctx.root_session_id
        and bool(getattr(row, "coordinator_run_id", None))
        and bool(getattr(row, "work_unit_id", None))
    )
    if not authoritative:
        raise RuntimeError(
            "coordinator cascade echo authoritative lineage mismatch"
        )
    return str(row.coordinator_run_id)


class ResultReadyHandler:
    """Spec §7.3 — terminal envelope → destroy with full failure classification.

    Side-effect-first ordering (spec §5.8 hard rule):

    1. ``get_processed`` precheck → already processed? return ack=True
       without side_effect (idempotency).
    2. ``upsert_processing`` stages the audit row (belt-and-suspenders;
       supervisor's outer ``_handle_envelope`` also upserts).
    3. ``side_effect`` runs terminal-row CAS → destroy() → on success:
       agent_service_callback (ChildDoneEvent fanout) → mark_processed.

    Destroy outcome classification:
      - clean return                  → callback + mark_processed (ACK)
      - ``SandboxAlreadyDestroyed``    → terminal-success (idempotent no-op)
      - ``SandboxBindingMissing``      → terminal-success (nothing to destroy)
      - ``SandboxLifecycleError`` else → raise (no ACK, XAUTOCLAIM retry)

    codex r5 [R5-2, MEDIUM PERF] handler invariant — every terminal handler
    MUST verify its own audit state (``get_processed`` + ``upsert_processing``)
    even though the outer ``_handle_envelope`` already pre-staged both calls
    (see ``MailboxSupervisor._handle_envelope`` lines ~1396 and ~1493). This
    is deliberate belt-and-suspenders, NOT redundant work that should be
    deleted:
      * Handlers are invoked directly in unit tests (47 callsites in
        ``test_mailbox_supervisor.py``) that do NOT pre-call
        ``upsert_processing``; without the in-handler check the
        ``mark_processed`` calls inside the handler would raise on the
        DB-faithful ``_StrictAuditRepo`` stub.
      * Future code paths (e.g. PR-5 ``reconcile_orphans`` synthetic
        re-dispatch) may invoke the handler outside the standard
        ``_handle_envelope`` funnel — the duplicate guard keeps each handler
        self-contained.
    Cost: 2 extra DB roundtrips per terminal envelope (one ``get_processed``
    + one ``upsert_processing``); both are short, indexed reads on the
    audit table. The supervisor's terminal-envelope rate is bounded by
    child completion (not high frequency), so the cost is acceptable in
    exchange for the contract simplicity.
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # R5-2 belt-and-suspenders — see class docstring. The outer
        # ``_handle_envelope`` already ran ``get_processed`` at the entry
        # gate (line ~1396) and ``upsert_processing`` before dispatch (line
        # ~1493). Repeating both here keeps the handler usable outside the
        # supervisor's main loop (unit tests, future synthetic dispatchers).
        if await ctx.audit_repo.get_processed(
            envelope.parent_session_id, envelope.envelope_id
        ):
            return HandlerOutcome(ack=True, audit_payload={"dedup": True})

        await ctx.audit_repo.upsert_processing(envelope, processing_at=ctx.now())

        async def _side_effect() -> None:
            await _mark_terminal_liveness(envelope, ctx)
            # [C2 PR-6 §14.4] cost rollup PROLOGUE — fires only for
            # ``coordinator_step`` children with a parent_session_id set.
            # Best-effort: a rollup failure MUST NOT abort the load-bearing
            # destroy + callback + mark_processed body below (destroy is the
            # last point we hold the child session row, so cost data should
            # be rolled up first — but rollup is observability, destroy is
            # safety). All exceptions are logged + swallowed except
            # ``asyncio.CancelledError`` which always propagates.
            #
            # Gate: both ``cost_rollup_service`` AND ``session_repo`` must
            # be wired. Legacy / pre-PR-6 supervisor contexts leave both
            # None and the prologue silently no-ops.
            #
            # Implementations are required to be idempotent (see
            # ``CostRollupService.rollup_to_parent`` docstring) because
            # XAUTOCLAIM may redeliver after a transient destroy failure
            # and the prologue will re-fire each replay.
            if ctx.cost_rollup_service is not None and ctx.session_repo is not None:
                try:
                    child_session = await ctx.session_repo.get_by_id(
                        envelope.child_session_id
                    )
                    if (
                        child_session is not None
                        and child_session.tool_filter_preset == COORDINATOR_STEP_PRESET
                        and child_session.parent_session_id is not None
                    ):
                        cost_summary = (
                            envelope.payload.get("cost_summary", {})
                            if isinstance(envelope.payload, dict)
                            else {}
                        )
                        await ctx.cost_rollup_service.rollup_to_parent(
                            parent_session_id=child_session.parent_session_id,
                            cost=cost_summary,
                            source="coordinator_subagent",
                            # [codex R2 P1-5] Pass envelope_id as the
                            # idempotency key so concrete implementations
                            # can dedupe on supervisor PEL retry. The
                            # envelope_id is unique per RESULT_READY
                            # event (mint-once on the child side), so
                            # XAUTOCLAIM-driven replay of the same event
                            # cannot double-count cost.
                            idempotency_key=envelope.envelope_id,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — best-effort rollup
                    logger.exception(
                        "result_ready cost rollup failed envelope=%s "
                        "child_session=%s — destroy continues",
                        envelope.envelope_id,
                        envelope.child_session_id,
                    )

            # [C2 PR-7 §12.4] persist terminal envelope PROLOGUE — fires only
            # for ``coordinator_step`` children. Best-effort: a persist failure
            # MUST NOT abort the destroy + callback + mark_processed body
            # below; the rehydrate path is correct-by-construction (it always
            # re-queries DB on each pod-start) so missing an envelope row only
            # forces a re-spawn at recovery time, not data loss.
            #
            # Gate: BOTH ``coordinator_envelope_store`` AND ``session_repo``
            # must be wired (the latter so we can read the child's
            # tool_filter_preset / coordinator_run_id / work_unit_id).
            # Crucially, we re-fetch the child session HERE (not reuse the
            # rollup-side fetch) because the cost-rollup gate may have
            # short-circuited on missing session_repo. Independence keeps
            # one PROLOGUE's failure from poisoning the next.
            #
            # The (coordinator_run_id, work_unit_id) UNIQUE constraint on
            # ``coordinator_result_envelope_store`` means a retry replay of
            # the same envelope raises IntegrityError — we swallow that
            # specifically (it's the success-case of idempotent persistence).
            if (
                ctx.coordinator_envelope_store is not None
                and ctx.session_repo is not None
            ):
                try:
                    child_session = await ctx.session_repo.get_by_id(
                        envelope.child_session_id
                    )
                    if (
                        child_session is not None
                        and child_session.tool_filter_preset == COORDINATOR_STEP_PRESET
                        and child_session.coordinator_run_id is not None
                        and child_session.work_unit_id is not None
                    ):
                        payload = (
                            envelope.payload
                            if isinstance(envelope.payload, dict)
                            else {}
                        )
                        await ctx.coordinator_envelope_store.persist_terminal(
                            coordinator_run_id=child_session.coordinator_run_id,
                            work_unit_id=child_session.work_unit_id,
                            child_session_id=envelope.child_session_id,
                            envelope_type="RESULT_READY",
                            payload=payload,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — best-effort persist
                    logger.exception(
                        "result_ready persist_terminal failed envelope=%s "
                        "child_session=%s — destroy continues",
                        envelope.envelope_id,
                        envelope.child_session_id,
                    )

            # [C4.1a §5.1] subagent_run record PROLOGUE — THIRD independent,
            # best-effort. Own gate + own get_by_id fetch + own try/except
            # (does NOT reuse the cost-rollup / persist-terminal child_session —
            # independence keeps one prologue's failure from poisoning the next).
            # Gate: subagent_run_repo AND session_repo wired (repo-or-None means
            # None on flag-OFF → silent skip → byte-identical, INV-C4.1-1). Only
            # coordinator_step children with full lineage are recorded; research
            # children are recorded by the research seat (§5.0.1 preset-disjoint).
            if ctx.subagent_run_repo is not None and ctx.session_repo is not None:
                try:
                    child_session = await ctx.session_repo.get_by_id(
                        envelope.child_session_id
                    )
                    if (
                        child_session is not None
                        and child_session.tool_filter_preset == COORDINATOR_STEP_PRESET
                        and child_session.coordinator_run_id is not None
                        and child_session.work_unit_id is not None
                        and child_session.parent_session_id is not None
                    ):
                        from app.application.services.subagent_worker_projection import (
                            project_local_result,
                        )
                        payload = (
                            envelope.payload
                            if isinstance(envelope.payload, dict)
                            else {}
                        )
                        run = project_local_result(
                            ResultReadyPayload.model_validate(payload),
                            parent_session_id=child_session.parent_session_id,
                            child_session_id=envelope.child_session_id,
                            work_unit_id=child_session.work_unit_id,
                        )
                        await ctx.subagent_run_repo.record(run)
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — best-effort subagent-run record
                    logger.exception(
                        "subagent_run record failed envelope=%s child_session=%s "
                        "— destroy continues",
                        envelope.envelope_id,
                        envelope.child_session_id,
                    )

            # Codex F5 (HIGH) — every ``telemetry.emit`` inside this block
            # is wrapped so an OTel / sink fault never escapes the
            # side_effect. A raised emit would propagate up to
            # ``_handle_envelope``, prevent the ACK, and XAUTOCLAIM would
            # redeliver. Once destroyed, redelivery hits AlreadyDestroyed
            # (terminal-success) → emit raises again → infinite loop until
            # poison drop. **Fail-open via the observability path.**
            try:
                await ctx.sandbox_lifecycle.destroy(
                    envelope.child_session_id,
                    DestroyReason.SUBAGENT_TERMINAL_RESULT,
                )
            except SandboxAlreadyDestroyed:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.destroy_idempotent_noop",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "destroy_idempotent_noop telemetry raised envelope=%s "
                        "— terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxBindingMissing:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.destroy_binding_missing",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "destroy_binding_missing telemetry raised envelope=%s "
                        "— terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxLifecycleError as e:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.destroy_retryable_failed",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "error": str(e),
                            "reclaim_count": envelope.reclaim_count,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "destroy_retryable_failed telemetry raised envelope=%s "
                        "— re-raising SandboxLifecycleError to preserve PEL-"
                        "retain semantics",
                        envelope.envelope_id,
                    )
                # No mark_processed; supervisor sees the raise → no ACK →
                # XAUTOCLAIM retries; reclaim_count > MAX → poison drop.
                raise

            # Codex r4 [R4-4, MEDIUM CONTRACT] — clear per-child tracking
            # BEFORE the agent_service_callback. The previous ordering
            # placed ``clear_child_tracking`` after the callback so a
            # callback failure (agent_service down, SSE bridge raises,
            # ...) caused the in-memory tracking state to leak: the
            # destroyed child stayed in ``_last_seen_mono`` and
            # ``_cancel_states`` and the auto-escalate / orphan ticks
            # would fire spurious cascades against it.
            #
            # The child IS destroyed at this point; tracking should be
            # cleared regardless of whether the downstream notification
            # (callback fires SSE events to the frontend) succeeds. The
            # callback's job is to surface the terminal event to
            # observers — orthogonal to cleanup invariants.
            if ctx.clear_child_tracking is not None:
                ctx.clear_child_tracking(envelope.child_session_id)

            # Wrap the callback so its failure doesn't abort the rest
            # of the side_effect (mark_processed below). The callback
            # is a notification hook; a downstream observer fault must
            # not corrupt the supervisor's own state-machine progress.
            try:
                await ctx.agent_service_callback(envelope)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — best-effort hook
                logger.exception(
                    "result_ready agent_service_callback raised envelope=%s "
                    "— continuing to mark_processed; child tracking already "
                    "cleared (R4-4)",
                    envelope.envelope_id,
                )

            # Codex F6 (HIGH) — mark_processed here is belt-and-suspenders;
            # the outer ``_handle_envelope`` runs an idempotent mark_processed
            # post-side_effect. Wrap in try/except so a DB hiccup doesn't
            # cause the side_effect to raise → no ACK → XAUTOCLAIM redelivers
            # → destroy is now AlreadyDestroyed (terminal-success) → re-fire
            # callback + spurious work on every retry. The outer
            # mark_processed at line ~1109 is the load-bearing dedup write.
            try:
                await ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "result_ready inner mark_processed failed envelope=%s "
                    "— outer _handle_envelope will retry post-side_effect",
                    envelope.envelope_id,
                )

        return HandlerOutcome(
            ack=False,
            side_effect=_side_effect,
            audit_payload={"terminal": "RESULT_READY"},
        )


class CancelAckHandler:
    """Spec §7.4 — mirror of ResultReady with destroy reason CANCEL_ACK_OBSERVED.

    Triggered when the child cooperatively confirmed cancellation (its CANCEL_ACK
    envelope arrived). Same 4-way destroy classification + side-effect-first
    ordering as ResultReady; only the ``DestroyReason`` differs.

    codex r9b [R9b-2, HIGH CONTRACT] — supervisor-echo cleanup trust contract:
    the handler short-circuits duplicate sandbox destroy + callback on
    ``producer_role=SUPERVISOR_ECHO`` without verifying the envelope's origin,
    by design. This trust applies only to the pre-existing cleanup surface.
    Task 4's terminal DB write is a separate side effect and requires the
    authoritative session-row guard in ``_terminalize_child_from_envelope``.

      Publisher contract — ``ProducerRole.SUPERVISOR_ECHO`` is reserved
      for the supervisor's own re-published CANCEL_ACK envelope after a
      TERMINATE cascade. No other publisher (child agent, sibling
      supervisor, ops tool, external publisher) is permitted to set
      this producer_role. The supervisor still trusts the wire field for
      suppressing duplicate cleanup, but no longer grants terminal DB authority
      from that field: production ``session_repo`` verifies child identity,
      mailbox control plane, coordinator preset, and parent/root lineage first.

    A forged ``CHILD_AGENT``-published CANCEL_ACK with
    ``producer_role=SUPERVISOR_ECHO`` would short-circuit destroy +
    callback. The cross-root guard (parent_session_id check) catches
    misrouted envelopes from other supervisor scopes; within one root,
    we trust the publishers participating in that root's mailbox only for the
    duplicate-cleanup decision. The
    SUPERVISOR_ECHO short-circuit is load-bearing for the FORCE_TERMINATE
    flow: without it, every cascade's synthetic CANCEL_ACK re-fires
    destroy → wasted lifecycle work + duplicate telemetry on every
    cascade (the R2-4 regression this guard fixed).

    Locked behaviour:
    ``tests/domain/services/test_mailbox_supervisor.py::TestCancelAckHandlerEchoTrust::test_supervisor_echo_short_circuit_is_trust_based``
    asserts that cleanup short-circuit. It does not grant or test an
    unguarded terminal DB write.
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # R5-2 belt-and-suspenders — see ResultReadyHandler.__doc__. The
        # outer ``_handle_envelope`` already ran the duplicate audit calls,
        # but each handler must verify its own state to remain usable in
        # direct unit-test / synthetic-dispatch contexts.
        if await ctx.audit_repo.get_processed(
            envelope.parent_session_id, envelope.envelope_id
        ):
            return HandlerOutcome(ack=True, audit_payload={"dedup": True})

        # Codex r2 [R2-4, HIGH CONTRACT] — supervisor-echo short-circuit.
        # CancelRequestHandler._terminate_outcome publishes a synthetic
        # ``producer_role=SUPERVISOR_ECHO`` CANCEL_ACK so external observers
        # (frontend SSE bridge, parent audit trail) see the terminal
        # transition (spec §9.2). The same supervisor reads that envelope
        # back via XREADGROUP and re-dispatches here. Without this guard
        # every FORCE_TERMINATE produces a second destroy attempt that
        # raises ``SandboxAlreadyDestroyed`` (idempotent terminal-success)
        # → wasted lifecycle work + duplicate telemetry on every cascade.
        # The echo carries a synthetic envelope_id (``ack:{sha256...}``
        # after R3-7) but ``producer_role`` is the load-bearing
        # discriminator — NOT the envelope_id prefix. Mark
        # processed + ACK so the audit trail records the supervisor's own
        # echo without re-firing destroy / agent_service_callback.
        if envelope.producer_role == ProducerRole.SUPERVISOR_ECHO:
            await ctx.audit_repo.upsert_processing(
                envelope, processing_at=ctx.now()
            )
            # The echo still owns the session-row terminal CAS. It skips
            # destroy/callback because CancelRequestHandler already performed
            # those operations before publishing this envelope. Keeping the CAS
            # inside a side_effect preserves PEL retry + audit-before-XACK.
            async def _terminalize_echo() -> None:
                await _mark_terminal_liveness(envelope, ctx)

            return HandlerOutcome(
                ack=True,
                side_effect=_terminalize_echo,
                audit_payload={"supervisor_echo": True},
            )

        await ctx.audit_repo.upsert_processing(envelope, processing_at=ctx.now())

        async def _side_effect() -> None:
            await _mark_terminal_liveness(envelope, ctx)
            # [C2 PR-7 §12.4] persist terminal envelope PROLOGUE for CANCEL_ACK.
            # Same best-effort pattern as ResultReadyHandler; see that handler
            # for the full rationale. Unlike RESULT_READY this handler does
            # NOT roll up cost — CancelAck on a coordinator_step child is the
            # cooperative-cancel terminal and cost is already accruing into
            # the parent via earlier RESULT_READY (if any). The persisted
            # CANCEL_ACK envelope carries the final_state for rehydrate-driven
            # replay (e.g. cancelled-mid-step → resume needs to know that
            # work_unit was cancelled, not pending).
            if (
                ctx.coordinator_envelope_store is not None
                and ctx.session_repo is not None
            ):
                try:
                    child_session = await ctx.session_repo.get_by_id(
                        envelope.child_session_id
                    )
                    if (
                        child_session is not None
                        and child_session.tool_filter_preset == COORDINATOR_STEP_PRESET
                        and child_session.coordinator_run_id is not None
                        and child_session.work_unit_id is not None
                    ):
                        payload = (
                            envelope.payload
                            if isinstance(envelope.payload, dict)
                            else {}
                        )
                        await ctx.coordinator_envelope_store.persist_terminal(
                            coordinator_run_id=child_session.coordinator_run_id,
                            work_unit_id=child_session.work_unit_id,
                            child_session_id=envelope.child_session_id,
                            envelope_type="CANCEL_ACK",
                            payload=payload,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001 — best-effort persist
                    logger.exception(
                        "cancel_ack persist_terminal failed envelope=%s "
                        "child_session=%s — destroy continues",
                        envelope.envelope_id,
                        envelope.child_session_id,
                    )

            # Codex F5 (HIGH) — telemetry.emit isolation; see
            # ResultReadyHandler for the full rationale. Without these
            # wrappers, a sink fault on the AlreadyDestroyed branch
            # propagates → no ACK → XAUTOCLAIM redelivers → destroy
            # AlreadyDestroyed → loops on the observability path.
            try:
                await ctx.sandbox_lifecycle.destroy(
                    envelope.child_session_id,
                    DestroyReason.CANCEL_ACK_OBSERVED,
                )
            except SandboxAlreadyDestroyed:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.cancel_ack_destroy_idempotent",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "cancel_ack_destroy_idempotent telemetry raised "
                        "envelope=%s — terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxBindingMissing:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.cancel_ack_binding_missing",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "cancel_ack_binding_missing telemetry raised "
                        "envelope=%s — terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxLifecycleError as e:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.cancel_ack_destroy_retryable_failed",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "error": str(e),
                            "reclaim_count": envelope.reclaim_count,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "cancel_ack_destroy_retryable_failed telemetry raised "
                        "envelope=%s — re-raising SandboxLifecycleError to "
                        "preserve PEL-retain semantics",
                        envelope.envelope_id,
                    )
                raise

            # Codex r4 [R4-4, MEDIUM CONTRACT] — clear per-child tracking
            # BEFORE the agent_service_callback so a callback failure
            # doesn't leak the destroyed child's tracking state into
            # ``_last_seen_mono`` / ``_cancel_states``. See
            # ResultReadyHandler for the full rationale; mirrored here
            # because the bug shape is identical:
            #   prior order: destroy → callback → clear_child_tracking
            #                callback raises → clear never runs → stale
            #                state → auto-escalate cascades fire against
            #                an already-destroyed child.
            if ctx.clear_child_tracking is not None:
                ctx.clear_child_tracking(envelope.child_session_id)

            # Wrap the callback so its failure doesn't abort the rest
            # of the side_effect (mark_processed below). Notification
            # hook fault must not corrupt the supervisor's progress.
            try:
                await ctx.agent_service_callback(envelope)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — best-effort hook
                logger.exception(
                    "cancel_ack agent_service_callback raised envelope=%s "
                    "— continuing to mark_processed; child tracking already "
                    "cleared (R4-4)",
                    envelope.envelope_id,
                )

            # Codex F6 (HIGH) — best-effort mark_processed; outer
            # ``_handle_envelope`` provides the load-bearing dedup write.
            try:
                await ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cancel_ack inner mark_processed failed envelope=%s "
                    "— outer _handle_envelope will retry post-side_effect",
                    envelope.envelope_id,
                )

        return HandlerOutcome(
            ack=False,
            side_effect=_side_effect,
            audit_payload={"terminal": "CANCEL_ACK"},
        )


class CancelRequestHandler:
    """Spec §7.6 + §8 — TERMINATE drives hard kill; REQUEST_CANCEL records state
    for the auto-escalate tick (§8.4).

    TERMINATE branch ordering (spec §7.6 step 1-3):
      1. ``agent_service_callback`` first (callback may stop the agent task
         cooperatively before the sandbox dies).
      2. ``destroy(reason)`` — reason defaults to FORCE_TERMINATE. Internal
         cascades (orphan tick §7.5, poison drop §5.7, cancel auto-escalate
         §8.4) thread an explicit ``DestroyReason`` (e.g., ``ORPHAN_TIMEOUT``)
         via the supervisor-private ``cascade_destroy_overrides`` side-table
         on :class:`SupervisorContext` — see R6-2 and R7-4 below. Parent-
         originated cancels never appear in the side-table and always get
         the default ``FORCE_TERMINATE``.
      3. Synthetic CANCEL_ACK echo (``producer_role=SUPERVISOR_ECHO`` — must
         NOT be CHILD_AGENT or the last_seen heartbeat invariant breaks).
      4. ``mark_processed``.

    REQUEST_CANCEL branch:
      1. Forward to in-process callback so the child agent can cooperate.
      2. Record ``_cancel_state`` for ``_maybe_tick_cancel_check`` to promote
         to TERMINATE after ``CHILD_CANCEL_ACK_TIMEOUT_MS``.
      3. ``mark_processed``.

    Override-threading mechanism (codex r6 [R6-2] + r7 [R7-4]):
      Earlier rounds threaded the cascade destroy reason through the wire
      via ``CancelRequestPayload.destroy_reason`` and used a
      ``producer_role=SUPERVISOR`` guard to defuse hostile overrides. R6-2
      moved the field off the wire — the schema remains frozen at C3 ship
      at ``{reason, policy}`` — and into the supervisor-private
      ``ctx.cascade_destroy_overrides`` side-table keyed by the synthetic
      envelope_id of the cascade. The publisher
      (``_emit_cascade_terminate``) populates the entry BEFORE
      ``publisher.publish`` so the handler can read it on dispatch via
      ``ctx.cascade_destroy_overrides.get(env_id)``. External producers
      cannot reach the dict, which moots the previous producer_role guard.

      R7-4 defers the ``pop`` until AFTER the side_effect's
      ``destroy + publish_ack`` have both succeeded; on PEL retry the
      override is still present so the same ``DestroyReason`` is
      re-applied. See ``_terminate_outcome`` and
      :attr:`SupervisorContext.cascade_destroy_overrides` for details.
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # R5-2 belt-and-suspenders — see ResultReadyHandler.__doc__. Handler
        # repeats the supervisor's pre-stage so it stays self-contained
        # under direct invocation.
        if await ctx.audit_repo.get_processed(
            envelope.parent_session_id, envelope.envelope_id
        ):
            return HandlerOutcome(ack=True, audit_payload={"dedup": True})

        await ctx.audit_repo.upsert_processing(envelope, processing_at=ctx.now())

        policy_raw = envelope.payload.get("policy", CancelPolicy.REQUEST_CANCEL.value)
        # `policy` can already be coerced to enum by the envelope model_validator
        # round-trip (mode="python"); accept both raw string and enum value.
        try:
            policy = (
                policy_raw
                if isinstance(policy_raw, CancelPolicy)
                else CancelPolicy(policy_raw)
            )
        except ValueError:
            await ctx.telemetry.emit(
                "mailbox.cancel_invalid_policy",
                {
                    "envelope_id": envelope.envelope_id,
                    "policy": str(policy_raw),
                },
            )
            return HandlerOutcome(
                ack=True, audit_payload={"invalid_policy": True}
            )

        if policy == CancelPolicy.TERMINATE:
            return self._terminate_outcome(envelope, ctx)
        # REQUEST_CANCEL is the only remaining value — ABANDON is reserved/
        # rejected per spec §4.2.1, so CancelPolicy() above would have raised
        # ValueError and the invalid-policy ACK-drop would have triggered.
        return self._request_cancel_outcome(envelope, ctx)

    def _terminate_outcome(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # codex r6 [R6-2, HIGH CONTRACT] — read the supervisor-private
        # ``cascade_destroy_overrides`` side-table for the explicit
        # ``DestroyReason`` to thread into ``SandboxLifecycleService
        # .destroy``. Orphan tick (§7.5) and poison-drop fallback (§5.7)
        # stamp the entry (keyed by the synthetic envelope_id of the
        # cascade) BEFORE publishing the synthetic CANCEL_REQUEST; this
        # handler reads on dispatch. Parent-originated cancels
        # never appear in the side-table → default
        # ``DestroyReason.FORCE_TERMINATE``.
        #
        # Earlier rounds (R2-6 + R3-6) threaded the override via
        # ``CancelRequestPayload.destroy_reason`` on the wire and
        # restricted honoring to ``producer_role=SUPERVISOR`` to defuse a
        # hostile-override attack. R6-2 moves the field off the wire so
        # the schema stays frozen at C3 ship at ``{reason, policy}`` and
        # the side-table is supervisor-private (external producers
        # cannot reach it), which moots the R3-6 guard.
        #
        # codex r7 [R7-4, HIGH CONTRACT] — earlier R6-2 implementation
        # ``pop``ped the override BEFORE running the side_effect. If any
        # downstream step raised (e.g., destroy raised the retryable
        # ``SandboxLifecycleError``), side_effect re-raises → no ACK →
        # XAUTOCLAIM redelivers the envelope → the handler runs again
        # → the override has already been popped → falls back to
        # ``FORCE_TERMINATE`` instead of the original ``ORPHAN_TIMEOUT``.
        # Switch to ``get`` here; the pop happens *only* on the
        # all-steps-succeeded path inside the side_effect closure (right
        # before the outer ACK flow runs). On retry the override is still
        # present and the same destroy reason is re-applied.
        cascade_overrides = (
            ctx.cascade_destroy_overrides
            if ctx.cascade_destroy_overrides is not None
            else {}
        )
        override = cascade_overrides.get(envelope.envelope_id)
        destroy_reason_to_apply: DestroyReason = (
            override if override is not None else DestroyReason.FORCE_TERMINATE
        )

        async def _side_effect() -> None:
            # Step 1: stop_session (best-effort, isolated). The agent task
            # cooperating means the destroy below is less likely to land
            # mid-tool-call, but a callback failure must NOT prevent the
            # destroy from running. Codex F1 (HIGH) — the telemetry emit
            # itself is wrapped so a sink fault doesn't abort the cascade
            # (spec §7.6 invariant: stop failure must not prevent destroy).
            try:
                await ctx.agent_service_callback(envelope)
            except Exception as e:  # noqa: BLE001 — best-effort hook
                try:
                    await ctx.telemetry.emit(
                        "mailbox.force_terminate_stop_failed",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "error": str(e),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "force_terminate_stop_failed telemetry raised "
                        "envelope=%s — continuing to destroy",
                        envelope.envelope_id,
                    )

            # Step 2: destroy with the resolved ``DestroyReason``. Default
            # is FORCE_TERMINATE; orphan/poison cascades override via the
            # supervisor-private ``cascade_destroy_overrides`` side-table
            # (codex r6 [R6-2]; the R2-6 payload field is gone). Codex F5 (HIGH)
            # — every ``telemetry.emit`` in this block is isolated so an
            # OTel / sink hiccup doesn't escape the side_effect and
            # trigger XAUTOCLAIM redelivery (which would re-hit
            # AlreadyDestroyed → terminal-success → loop forever).
            try:
                await ctx.sandbox_lifecycle.destroy(
                    envelope.child_session_id,
                    destroy_reason_to_apply,
                )
            except SandboxAlreadyDestroyed:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.force_terminate_idempotent_noop",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "force_terminate_idempotent_noop telemetry raised "
                        "envelope=%s — terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxBindingMissing:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.force_terminate_binding_missing",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "force_terminate_binding_missing telemetry raised "
                        "envelope=%s — terminal-success path continues",
                        envelope.envelope_id,
                    )
            except SandboxLifecycleError as e:
                try:
                    await ctx.telemetry.emit(
                        "mailbox.force_terminate_destroy_retryable_failed",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "reclaim_count": envelope.reclaim_count,
                            "error": str(e),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "force_terminate_destroy_retryable_failed telemetry "
                        "raised envelope=%s — re-raising SandboxLifecycleError "
                        "to preserve PEL-retain semantics",
                        envelope.envelope_id,
                    )
                raise

            # Codex r4 [R4-4, MEDIUM CONTRACT] — clear per-child tracking
            # immediately after destroy success, BEFORE the synthetic
            # CANCEL_ACK echo publish. The publish call is the only
            # un-wrapped step in this side_effect; if the Redis client
            # raises here, ``clear_child_tracking`` would never fire,
            # leaving stale tracking state for an already-destroyed
            # child → auto-escalate / orphan cascades against a dead
            # child. Mirror of the ResultReady / CancelAck R4-4 fix:
            # cleanup invariants run before any potentially-raising
            # downstream step.
            if ctx.clear_child_tracking is not None:
                ctx.clear_child_tracking(envelope.child_session_id)

            # Step 3: synthetic CANCEL_ACK echo so any parent / SSE bridge
            # observers see the terminal transition. SUPERVISOR_ECHO is
            # critical here — using CHILD_AGENT would falsely refresh
            # ``_last_seen_mono`` and the orphan detector would think a
            # destroyed child is still alive (spec §9.2 invariant).
            # codex r3 [R3-7, HIGH CONTRACT] — hash the synthetic envelope_id
            # so a long upstream envelope_id doesn't overflow the audit
            # column's ``String(64)`` bound. See ``_synthetic_envelope_id``.
            ack_correlation_id = await _resolve_terminate_echo_correlation_id(
                envelope, ctx
            )
            ack_env = MailboxEnvelope(
                envelope_id=_synthetic_envelope_id("ack", envelope.envelope_id),
                type=MailboxEnvelopeType.CANCEL_ACK,
                parent_session_id=envelope.parent_session_id,
                child_session_id=envelope.child_session_id,
                correlation_id=ack_correlation_id,
                emitted_at=ctx.now(),
                producer_role=ProducerRole.SUPERVISOR_ECHO,
                payload=CancelAckPayload(
                    final_state="force_terminated"
                ).model_dump(mode="json"),
            )
            await ctx.publisher.publish(ack_env)

            # codex r7 [R7-4, HIGH CONTRACT] — pop the cascade override
            # ONLY after destroy + publish_ack have succeeded. If any
            # earlier step raised, side_effect re-raises → outer
            # _handle_envelope leaves the entry in PEL → XAUTOCLAIM
            # redelivers → this handler runs again → override is still
            # present and the same ``DestroyReason`` is re-applied. The
            # publish above is the last raise-capable step that gates
            # PEL retention; mark_processed below is wrapped to swallow
            # so the pop here is safe. Mirrors the “consume the override
            # only when the handler is about to ACK successfully”
            # pattern called out in the R7-4 finding.
            if ctx.cascade_destroy_overrides is not None:
                ctx.cascade_destroy_overrides.pop(envelope.envelope_id, None)

            # Codex F6 (HIGH) — mark_processed here is belt-and-suspenders;
            # the outer ``_handle_envelope`` runs an idempotent mark_processed
            # post-side_effect. Wrapping in try/except so a DB hiccup doesn't
            # raise from side_effect → trigger redelivery → burn a destroy
            # cycle on every retry.
            try:
                await ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "force_terminate inner mark_processed failed envelope=%s "
                    "— outer _handle_envelope will retry post-side_effect",
                    envelope.envelope_id,
                )

        return HandlerOutcome(
            ack=False,
            side_effect=_side_effect,
            audit_payload={"terminal": "CANCEL_TERMINATE"},
        )

    def _request_cancel_outcome(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        async def _side_effect() -> None:
            # codex r5 [R5-3, HIGH CONTRACT] — register cancel state FIRST so
            # the auto-escalate tick (`_maybe_tick_cancel_check` §8.4) is
            # armed BEFORE any callback that might hang or fail. The prior
            # ordering was `callback → register_cancel_state`; if the
            # in-process ``agent_service_callback`` hung (child task slow
            # to respond to cancel) or raised (callback bug), the
            # ``register_cancel_state`` step never ran → ``_cancel_states[child]``
            # stayed empty → the auto-escalate tick saw no state for this
            # child → REQUEST_CANCEL never escalated to TERMINATE → child
            # stuck in REQUEST_CANCEL limbo until the 90s orphan tick. The
            # invariant: cancel-state registration is the load-bearing
            # safety net; the callback is best-effort cooperation. Wrap the
            # callback in try/except so its failure doesn't abort
            # mark_processed below.
            if ctx.register_cancel_state is not None:
                await ctx.register_cancel_state(
                    envelope.child_session_id,
                    CancelPolicy.REQUEST_CANCEL,
                    ctx.clock(),
                )
            else:
                # Should never happen — supervisor binds the hook in __init__.
                try:
                    await ctx.telemetry.emit(
                        "mailbox.cancel_state_hook_missing",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "cancel_state_hook_missing telemetry raised "
                        "envelope=%s — continuing",
                        envelope.envelope_id,
                    )

            # Forward to child via in-process callback (best-effort). With
            # cancel state already registered, even a callback fault leaves
            # the auto-escalate tick armed — child cannot get stuck in
            # REQUEST_CANCEL limbo on a callback bug.
            try:
                await ctx.agent_service_callback(envelope)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001 — best-effort hook
                try:
                    await ctx.telemetry.emit(
                        "mailbox.request_cancel_callback_failed",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "error": str(e),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "request_cancel_callback_failed telemetry raised "
                        "envelope=%s — continuing to mark_processed",
                        envelope.envelope_id,
                    )
            await ctx.audit_repo.mark_processed(
                envelope.parent_session_id,
                envelope.envelope_id,
                processed_at=ctx.now(),
            )

        return HandlerOutcome(
            ack=False,
            side_effect=_side_effect,
            audit_payload={"cancel_state": "REQUEST_CANCEL"},
        )


class ApprovalRequestHandler:
    """Spec §10.2 — Permission-Engine HITL stub.

    PR-4 ships an *immediate deny* stub (NOT a 300s timeout wait) because
    PE-2 is not yet integrated. Publishes a paired APPROVAL_RESPONSE envelope
    with ``decided_by=AUTO_POLICY`` so the child's pending request unblocks
    immediately instead of timing out at ``APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS``.
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # ApprovalRequest carries its own correlation_id inside the payload
        # (spec §10.1) — use it to thread the response back to the same
        # in-flight tool call on the child side. The ``correlation_id`` key
        # is REQUIRED by ``ApprovalRequestPayload`` (see
        # ``api/app/domain/models/mailbox_envelope.py``) and that schema is
        # re-validated by ``MailboxEnvelope._validate_payload_matches_type``,
        # so any envelope reaching this handler is guaranteed to carry a
        # non-empty value (envelope-level locked guard:
        # ``tests/domain/services/test_mailbox_supervisor.py::TestApprovalRequestCorrelationIdMismatch::test_envelope_rejects_approval_request_missing_payload_correlation_id``).
        payload_correlation_id = envelope.payload["correlation_id"]

        # codex r4 [R4-2, HIGH CONTRACT] (refined by codex r6 [R6-4]) —
        # spec §13.1 T7 producer contract: ``payload.correlation_id`` MUST
        # equal ``envelope.correlation_id``. The two ids serve different
        # layers (envelope-level routing vs. tool-call-level correlation
        # on the child side), but APPROVAL_REQUEST is the one envelope
        # type where the spec mandates they match — producers are
        # required to wire the same string into both fields.
        #
        # codex r6 [R6-4, HIGH CONTRACT] — earlier R4-2 fix ACK+dropped
        # on mismatch WITHOUT publishing a paired APPROVAL_RESPONSE,
        # which forced the child agent to wait the full 300s
        # ``APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS`` before its
        # deny-by-default branch fired. PR-4's design intent (codex r5
        # veto: "立即 deny — 不假等 300s 制造假挂死") is the opposite:
        # children should NEVER block 300s on approval. Publish an
        # immediate deny keyed to ``envelope.correlation_id`` (the
        # supervisor's trusted source — the envelope router is what we
        # control; the payload field is producer-supplied). A child
        # written against the documented contract (payload mirrors
        # envelope) will unblock immediately on this deny. A buggy
        # child that keys only on the payload's id will still time out
        # at 300s — that's the producer's bug, not the supervisor's.
        # ``payload_correlation_id`` is guaranteed non-empty by the
        # envelope-level Pydantic validator (see comment at the top of
        # ``handle``); the mismatch branch is the only divergence case
        # because both ids are present.
        if payload_correlation_id != envelope.correlation_id:
            try:
                await ctx.telemetry.emit(
                    "mailbox.approval_correlation_id_mismatch",
                    {
                        "envelope_id": envelope.envelope_id,
                        "envelope_correlation_id": envelope.correlation_id,
                        "payload_correlation_id": payload_correlation_id,
                        "producer_role": envelope.producer_role.value,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "approval_correlation_id_mismatch telemetry raised "
                    "envelope=%s — continuing to publish deny keyed to "
                    "envelope.correlation_id (R6-4)",
                    envelope.envelope_id,
                )

            # R6-4 — publish a deny keyed to the trusted envelope-level
            # correlation_id so the child unblocks immediately. Use
            # ``_synthetic_envelope_id`` to keep the response envelope
            # id ≤ 64 chars even when the request id is at the limit.
            mismatch_response = MailboxEnvelope(
                envelope_id=_synthetic_envelope_id(
                    "mismatch_deny", envelope.envelope_id
                ),
                type=MailboxEnvelopeType.APPROVAL_RESPONSE,
                parent_session_id=envelope.parent_session_id,
                child_session_id=envelope.child_session_id,
                correlation_id=envelope.correlation_id,
                emitted_at=ctx.now(),
                producer_role=ProducerRole.SUPERVISOR,
                payload=ApprovalResponsePayload(
                    correlation_id=envelope.correlation_id,
                    approved=False,
                    decided_by=ApprovalDecidedBy.AUTO_POLICY,
                    reason=(
                        "correlation_id_mismatch — payload.correlation_id "
                        "differs from envelope.correlation_id; supervisor "
                        "denies and keys response off envelope.correlation_id "
                        "(R6-4)"
                    ),
                ).model_dump(mode="json"),
            )
            # codex r7 [R7-6, HIGH CONTRACT] — earlier R6-4 implementation
            # ACKed the envelope regardless of the deny publish outcome.
            # If ``publisher.publish`` raised here, the paired deny never
            # reached the stream so the child agent stayed blocked the
            # full ``APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS`` (300s) —
            # violating PR-4's "immediate deny" contract (codex r5 veto
            # rationale). Worse: the ACK + ``mark_processed`` made
            # redelivery impossible (no XAUTOCLAIM rescue). The
            # publisher uses SET NX dedupe (see RedisMailboxPublisher),
            # so retrying the same deny envelope_id is safe; defer ACK
            # so PEL retention triggers redelivery + retry until the
            # deny lands.
            try:
                await ctx.publisher.publish(mismatch_response)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "approval mismatch deny publish failed envelope=%s — "
                    "leaving envelope in PEL (R7-6) so XAUTOCLAIM redelivers "
                    "and the paired deny is retried; child unblocks on the "
                    "first successful publish via SET NX dedup",
                    envelope.envelope_id,
                )
                # Explicit defer — no ACK, no side_effect; PR-3b reliability
                # layer reclaims via XAUTOCLAIM and re-dispatches.
                return HandlerOutcome(
                    ack=False,
                    audit_payload={
                        "correlation_id_mismatch": True,
                        "mismatch_deny_publish_failed": True,
                    },
                )
            return HandlerOutcome(
                ack=True,
                audit_payload={"correlation_id_mismatch": True},
            )

        # codex r9b [R9b-3, MEDIUM TEST] — the prior fallback
        # ``payload_correlation_id or envelope.correlation_id`` was dead
        # code: ``ApprovalRequestPayload.correlation_id`` is required and
        # re-validated by ``MailboxEnvelope._validate_payload_matches_type``,
        # so reaching this point with a missing ``payload_correlation_id``
        # is impossible from real wire input. Use the payload value
        # directly; the equality check above guarantees it matches
        # ``envelope.correlation_id`` on this branch.
        correlation_id = payload_correlation_id
        # codex r3 [R3-7, HIGH CONTRACT] — hash the synthetic envelope_id
        # so a long upstream envelope_id doesn't overflow the audit column's
        # ``String(64)`` bound. See ``_synthetic_envelope_id``.
        response_env = MailboxEnvelope(
            envelope_id=_synthetic_envelope_id("deny", envelope.envelope_id),
            type=MailboxEnvelopeType.APPROVAL_RESPONSE,
            parent_session_id=envelope.parent_session_id,
            child_session_id=envelope.child_session_id,
            correlation_id=correlation_id,
            emitted_at=ctx.now(),
            producer_role=ProducerRole.SUPERVISOR,
            payload=ApprovalResponsePayload(
                correlation_id=correlation_id,
                approved=False,
                reason="approval handler unavailable (PE-2 not yet integrated)",
                decided_by=ApprovalDecidedBy.AUTO_POLICY,
            ).model_dump(mode="json"),
        )
        await ctx.publisher.publish(response_env)
        return HandlerOutcome(
            ack=True,
            audit_payload={"pe_stub_denied": True},
        )


class HandoffRequestHandler:
    """Spec §6.6 — handoff intentionally not implemented in C3 ship.

    The wire envelope is frozen so future PRs can light up real handoff
    behavior without breaking publishers / consumers; for now we just emit
    telemetry and ACK. No destroy, no agent_service_callback.
    """

    async def handle(
        self, envelope: MailboxEnvelope, ctx: SupervisorContext
    ) -> HandlerOutcome:
        # codex r2 [R2-7, MEDIUM CONTRACT] — telemetry.emit isolation;
        # same pattern as ResultReadyHandler / CancelAckHandler. If the
        # sink raises (OTel hiccup, JSONL disk full, ...), the
        # exception propagates → no ACK → XAUTOCLAIM redelivers → handler
        # raises again → infinite loop on the observability path. The
        # handoff envelope is unsupported by design (spec §6.6 — wire
        # reserved without lighting up handoff logic) so the only
        # action is the telemetry note; failing to emit is loggable but
        # MUST NOT block the ACK.
        try:
            await ctx.telemetry.emit(
                "mailbox.handoff_request_unsupported",
                {
                    "envelope_id": envelope.envelope_id,
                    "child_session_id": envelope.child_session_id,
                    "handoff_target": envelope.payload.get("handoff_target"),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "handoff_request_unsupported telemetry raised envelope=%s "
                "— ACKing anyway; the handoff envelope is unsupported and "
                "carries no side-effect work that retrying could unblock",
                envelope.envelope_id,
            )
        return HandlerOutcome(
            ack=True,
            audit_payload={
                "unsupported": True,
                "reason": "c3_ship_does_not_implement_handoff",
            },
        )


# ─── Dispatch table (spec §6.3) ───────────────────────────────────────────────


# Sentinel values for ``sig.bind()`` arity probing — never compared, only used
# to assert the handler signature can accept two positional args.
_HANDLER_VALIDATION_ENV_SENTINEL = object()
_HANDLER_VALIDATION_CTX_SENTINEL = object()


def _validate_dispatch_table(
    table: dict[MailboxEnvelopeType, EnvelopeHandler],
) -> None:
    """Hard-fail at construction if ``table`` would let the supervisor drop
    or crash on a legitimate envelope.

    Checks (in order):
    1. Every ``MailboxEnvelopeType`` value is a key — otherwise the
       unknown-type branch in :meth:`MailboxSupervisor._handle_envelope`
       would ACK-drop the missing type.
    2. Every value has a callable ``handle`` attribute — ``None`` slips
       past (1) but ``.get(type)`` would return ``None`` at dispatch time.
    3. ``handle`` is ``async def`` — a sync ``def handle`` is ``callable``
       but ``await handler.handle(...)`` would crash with ``TypeError``.
    4. ``handle``'s signature accepts ``(envelope, ctx)`` positionally —
       wrong arity (codex r5 [P2]) passes (3) but crashes at call site.

    Called from :meth:`MailboxSupervisor.__init__` for BOTH the user-supplied
    ``dispatch_table`` AND the default returned by
    :func:`build_default_dispatch_table` — codex r5 [P2] wanted the default
    path under the same guard so a future regression that drops a stub
    handler trips the validator instead of silently ACK-dropping envelopes.
    """
    missing = set(MailboxEnvelopeType) - set(table)
    if missing:
        raise ValueError(
            "MailboxSupervisor dispatch_table must cover every "
            f"MailboxEnvelopeType — missing: "
            f"{sorted(t.value for t in missing)}. Start from "
            "build_default_dispatch_table() and mutate entries "
            "rather than passing a partial table."
        )

    invalid: list[tuple[str, str]] = []
    for t, h in table.items():
        handle = getattr(h, "handle", None)
        if not callable(handle):
            invalid.append((t.value, "no callable .handle"))
            continue
        if not inspect.iscoroutinefunction(handle):
            invalid.append((t.value, "handle is not async def"))
            continue
        try:
            inspect.signature(handle).bind(
                _HANDLER_VALIDATION_ENV_SENTINEL,
                _HANDLER_VALIDATION_CTX_SENTINEL,
            )
        except TypeError as e:
            invalid.append((t.value, f"signature won't accept (envelope, ctx): {e}"))

    if invalid:
        raise ValueError(
            "MailboxSupervisor dispatch_table values must expose an "
            "``async def handle(envelope, ctx) -> HandlerOutcome`` method — "
            f"invalid: {invalid!r}"
        )


def build_default_dispatch_table() -> dict[MailboxEnvelopeType, EnvelopeHandler]:
    """PR-4 default — terminal + cascade routes wire to the real handlers
    (``ResultReadyHandler`` / ``CancelAckHandler`` / ``CancelRequestHandler``
    / ``ApprovalRequestHandler`` / ``HandoffRequestHandler``); non-terminal
    pass-through types stay on ``_StubNonTerminalHandler`` (forward to
    ``ctx.agent_service_callback`` and ACK).

    INV: every value of ``MailboxEnvelopeType`` MUST be a key in the returned
    table — otherwise the supervisor would fall into the unknown-type branch
    and ACK-drop legitimate envelopes. CI enforcement: see the unit test that
    counts entries.
    """
    # Late import — ``coordinator_progress_handler`` imports
    # ``HandlerOutcome`` from this module via late-import in the handler
    # body. Importing it at the top of ``mailbox_supervisor`` would still
    # work (no module-load-time cycle since the handler's
    # ``HandlerOutcome`` import is deferred), but the late-import here
    # mirrors the cost_rollup_service / coordinator_envelope_store
    # precedent and keeps the supervisor's top of file lean.
    from app.application.services.coordinator_progress_handler import (
        CoordinatorProgressUpdateHandler,
    )

    stub_nonterminal = _StubNonTerminalHandler()
    return {
        MailboxEnvelopeType.RESULT_READY: ResultReadyHandler(),
        MailboxEnvelopeType.CANCEL_ACK: CancelAckHandler(),
        MailboxEnvelopeType.CANCEL_REQUEST: CancelRequestHandler(),
        MailboxEnvelopeType.APPROVAL_REQUEST: ApprovalRequestHandler(),
        MailboxEnvelopeType.SPAWN_REQUEST: stub_nonterminal,
        MailboxEnvelopeType.SPAWN_ACK: stub_nonterminal,
        # [C2 PR-8 Task 8.3 §13.4] PROGRESS_UPDATE now wraps coordinator-step
        # children with lineage tagging; non-coordinator children fall through
        # to the same stub-forward path the legacy handler took.
        MailboxEnvelopeType.PROGRESS_UPDATE: CoordinatorProgressUpdateHandler(),
        MailboxEnvelopeType.APPROVAL_RESPONSE: stub_nonterminal,
        MailboxEnvelopeType.DEPENDENCY_BLOCKED: stub_nonterminal,
        MailboxEnvelopeType.HANDOFF_REQUEST: HandoffRequestHandler(),
    }


# ─── Supervisor main loop ─────────────────────────────────────────────────────


class MailboxSupervisor:
    """Per-root supervisor task (M1). Single asyncio.Task lifecycle.

    The supervisor runs ``ensure_group()`` once, then loops on XREADGROUP +
    handler dispatch. The loop is robust against handler exceptions (the
    envelope stays in PEL for XAUTOCLAIM retry — PR-3b) and against
    transient Redis errors (logged + 0.5s back-off).

    ``CancelledError`` is the ONLY exception that escapes the inner loop —
    the parent task must be cancellable for clean asyncio shutdown.
    """

    # PR-3b class-level tunables — tests monkeypatch these to drive periodic
    # autoclaim into the test window. Production keeps the spec constants.
    _XAUTOCLAIM_INTERVAL_S: float = MAILBOX_XAUTOCLAIM_PERIODIC_INTERVAL_SECONDS
    _XAUTOCLAIM_MIN_IDLE_MS: int = MAILBOX_PEL_IDLE_MS_FOR_CLAIM

    # PR-4 cascade tunables. Production keeps both at 1s/5s respectively; tests
    # monkeypatch via the instance attribute so the cancel/orphan ticks fire
    # within the test window. ``_CANCEL_CHECK_INTERVAL_S`` MUST be smaller
    # than ``CHILD_CANCEL_ACK_TIMEOUT_MS / 1000`` (30s) — otherwise the
    # auto-escalate deadline is missed. ``_ORPHAN_CHECK_INTERVAL_S`` MUST be
    # smaller than ``SUBAGENT_PROGRESS_STALE_AFTER_SECONDS`` (90s) for the
    # same reason.
    _CANCEL_CHECK_INTERVAL_S: float = 1.0
    _ORPHAN_CHECK_INTERVAL_S: float = 5.0
    # C3 PR-5 (spec §11.6 rollback runbook) — operator runs the rollback
    # SQL out of band; the supervisor needs to detect that all live
    # subagent children flipped to legacy and stop itself on the next
    # tick. codex r3 [HIGH CONTRACT] — 5s cadence (matched to orphan
    # check) keeps the M1 race window between SQL apply + supervisor
    # exit short; the predicate also runs BEFORE entry dispatch in
    # ``run()`` so the supervisor doesn't process new mailbox work
    # within a tick of detecting rollback.
    _ROLLBACK_CHECK_INTERVAL_S: float = 5.0

    def __init__(
        self,
        ctx: SupervisorContext,
        *,
        dispatch_table: Optional[dict[MailboxEnvelopeType, EnvelopeHandler]] = None,
        block_ms: int = MAILBOX_XREADGROUP_BLOCK_MS,
        count: int = MAILBOX_XREADGROUP_COUNT,
        idle_poll_sleep_s: float = 0.0,
    ) -> None:
        self._ctx = ctx
        # INV (codex r1 [P1] fix): an explicit dispatch_table MUST cover every
        # MailboxEnvelopeType. The unknown-type branch in _handle_envelope is a
        # defense against schema drift (new wire type added without dispatch
        # entry), NOT a soft fallback — falling through to ACK-drop a legitimate
        # type because the caller passed a partial override would silently
        # discard envelopes. PR-4 swaps specific terminal handlers by starting
        # from build_default_dispatch_table() and mutating entries; we validate
        # here so a typo or forgotten entry trips at construction, not in
        # production after a real envelope arrives.
        # INV (codex r1/r2/r4/r5): validation always runs on BOTH the user-
        # supplied dispatch_table AND the default — see _validate_dispatch_table
        # for the full rationale. The default path is validated too so a
        # future regression that drops an enum value or replaces a stub with
        # a sync def trips at construction, not in production after a real
        # envelope arrives.
        if dispatch_table is None:
            dispatch_table = build_default_dispatch_table()
        _validate_dispatch_table(dispatch_table)
        self._dispatch = dispatch_table
        self._consumer = RedisMailboxConsumer(
            ctx.redis, ctx.root_session_id, ctx.pod_id, ctx.instance_id
        )
        self._stopping = asyncio.Event()
        self._stopped = asyncio.Event()
        # XREADGROUP tuning seams. Production keeps the constants (1000ms blocking
        # read, batch=32). Tests pass ``block_ms=0`` because the fakeredis async
        # impl does NOT wake a blocked XREADGROUP on a concurrent XADD — a
        # blocking call would hang forever in unit tests. ``idle_poll_sleep_s``
        # adds a tiny back-off when running non-blocking so the loop doesn't
        # spin-burn CPU; production with block_ms>0 should keep it at 0.
        self._block_ms = block_ms
        self._count = count
        self._idle_poll_sleep_s = idle_poll_sleep_s
        # PR-3b reliability layer state ──────────────────────────────────────
        # last_seen[child_session_id] = monotonic seconds of latest *child-
        # origin* envelope (spec §9.2). SUPERVISOR_ECHO must NOT update —
        # otherwise a supervisor's own CANCEL_ACK after destroy() would
        # falsely report the child as alive.
        self._last_seen_mono: dict[str, float] = {}
        # _cancel_states[child_session_id] = _CancelState set by PR-4 cancel
        # handler. PR-3b ships the store; PR-4 wires the periodic auto-
        # escalate tick (spec §7.5: REQUEST_CANCEL → TERMINATE after
        # CHILD_CANCEL_ACK_TIMEOUT_MS).
        self._cancel_states: dict[str, _CancelState] = {}
        # XAUTOCLAIM throttle — set to last_run_mono so the *first* loop
        # iteration after _initial_xautoclaim() must wait one interval
        # before running the periodic sweep again.
        self._last_autoclaim_mono: float = 0.0
        # codex r9b [R9b-1, HIGH ARCH] — pod-restart clock recovery
        # scaffolding deferred to PR-5.
        #
        # ``_known_children`` is the input set for
        # ``_restore_last_seen_after_pod_restart`` (spec §9.3 — XREVRANGE
        # scan + Redis-time anchor to rehydrate ``_last_seen_mono``). The
        # original PR-3b/PR-3c plan wired this in ``SupervisorRegistry.spawn``
        # by querying the session repo for running children at supervisor
        # startup, but that plumbing budget (``session_factory`` /
        # ``AsyncSession`` lifecycle / ancestor-chain walk into
        # ``SupervisorContext``) widens PR-4's blast radius beyond the
        # audit repo for an additive recovery optimisation. PR-4 ships
        # the orphan tick + cascade pipeline; pod-restart clock recovery
        # is now slated for PR-5 alongside the deferred R3-1 cross-root
        # child_session_id verification (same plumbing budget).
        #
        # Runtime safety while unwired: ``_known_children`` defaults to
        # ``[]`` so ``_restore_last_seen_after_pod_restart`` is a safe
        # no-op (the for-loop over an empty list never enters). The next
        # child-origin envelope after restart refreshes
        # ``_last_seen_mono`` via ``_maybe_advance_last_seen`` on the
        # normal dispatch path — the recovery routine is an
        # optimisation, not a correctness invariant.
        #
        # Locked behaviour:
        # ``tests/domain/services/test_mailbox_supervisor.py::TestSupervisorPodRestartClockRecovery::test_unwired_known_children_default_is_safe_noop``
        # asserts the unwired default is a no-op (no exception, no
        # mutation of ``_last_seen_mono``).
        self._known_children: list[str] = []
        # PR-4 cascade-tick anchors (spec §7.5 + §8.4). Both default to 0.0
        # so the first tick after ``run()`` startup waits a full interval
        # before firing (avoids a spurious tick immediately after
        # ``_initial_xautoclaim`` returns).
        self._last_cancel_check_mono: float = 0.0
        self._last_orphan_check_mono: float = 0.0
        # C3 PR-5 (spec §11.6) — see ``_maybe_tick_check_rollback``.
        self._last_rollback_check_mono: float = 0.0
        # codex r6 [R6-2, HIGH CONTRACT] — supervisor-private side-table
        # mapping synthetic envelope_id → DestroyReason override.
        # ``_emit_cascade_terminate`` populates BEFORE publish; the
        # CancelRequestHandler TERMINATE branch reads+pops on dispatch.
        # Replaces the R2-6 wire-payload field so the frozen
        # CancelRequestPayload schema stays {reason, policy} (R6-2).
        self._cascade_destroy_overrides: dict[str, DestroyReason] = {}
        # Bind the cancel-state registration hook so ``CancelRequestHandler``
        # can stash REQUEST_CANCEL state for the auto-escalate tick (spec §8).
        # The hook is bound exactly once at construction; handlers only call
        # it, preserving the "ctx is read-only at handler time" contract.
        ctx.register_cancel_state = self._register_cancel_state
        # Codex F7+F9 (HIGH) — same binding pattern for the per-child
        # cleanup hook used by terminal handlers to drop tracking entries
        # after destroy.
        ctx.clear_child_tracking = self._clear_child_tracking
        # codex r6 [R6-2, HIGH CONTRACT] — supervisor-private side-table
        # threaded into ctx so ``CancelRequestHandler._terminate_outcome``
        # can read+pop the explicit DestroyReason override that
        # ``_emit_cascade_terminate`` stamped before publish. Shared by
        # reference; the field default on SupervisorContext is a fresh
        # dict per ctx instance (dataclass default_factory) so test
        # contexts also get one without manual wiring.
        ctx.cascade_destroy_overrides = self._cascade_destroy_overrides

    async def run(self) -> None:
        try:
            await self._consumer.ensure_group()
            # PR-3b spec §5.6 — startup XAUTOCLAIM drains any PEL entries
            # left over by a dead consumer on the same root (min_idle_ms=0
            # claims everything regardless of idle time, because the previous
            # consumer is by definition no longer reading).
            await self._initial_xautoclaim()
            await self._restore_coordinator_liveness_after_startup()
            # Readiness means the consumer group, dead-consumer PEL claim and
            # durable child-liveness recovery have all completed.
            ready = getattr(self, "_ready_event", None)
            if ready is not None:
                ready.set()
            while not self._stopping.is_set():
                try:
                    # C3 PR-5 spec §11.6 rollback runbook (codex r3 [HIGH
                    # CONTRACT]) — check rollback BEFORE consuming new
                    # mailbox entries / dispatching handlers, so the
                    # supervisor doesn't fire destroy() on a row whose
                    # control_plane just flipped to legacy. Throttled to
                    # _ROLLBACK_CHECK_INTERVAL_S so the DB query is
                    # bounded; if the check decides to stop it sets
                    # _stopping eagerly so this iteration's read does
                    # not block long.
                    await self._maybe_tick_check_rollback()
                    if self._stopping.is_set():
                        break
                    entries = await self._consumer.read(
                        count=self._count,
                        block_ms=self._block_ms,
                    )
                    # Codex r10 [P1] fix — per-entry fault isolation. The
                    # outer try/except below catches read()/sleep() errors
                    # and applies a 0.5s back-off. Without an inner try
                    # around _handle_envelope, an exception from one entry
                    # (most likely a transient XACK failure, which is NOT
                    # caught inside _handle_envelope) would bubble out of
                    # the for-loop, sleep 0.5s, then the next iteration
                    # reads ">" and skips the unprocessed remainder of
                    # this batch — those entries sit in PEL until PR-3b
                    # XAUTOCLAIM picks them up. Isolating each call so a
                    # single failing entry doesn't starve the rest of the
                    # batch is part of the skeleton's contract; PEL
                    # redelivery semantics for the failing entry itself
                    # are unchanged (no ACK → XAUTOCLAIM retries in PR-3b).
                    for redis_id, envelope in entries:
                        try:
                            await self._handle_envelope(redis_id, envelope)
                        except asyncio.CancelledError:
                            raise
                        except Exception:
                            logger.exception(
                                "supervisor entry handling failed root=%s id=%s "
                                "type=%s — leave in PEL, continue batch",
                                self._ctx.root_session_id,
                                redis_id,
                                envelope.type.value,
                            )
                    # PR-3b spec §5.6 — periodic XAUTOCLAIM (every
                    # _XAUTOCLAIM_INTERVAL_S seconds). Inline so the cadence
                    # is driven by the same loop's clock and we don't need a
                    # separate task.
                    await self._maybe_periodic_xautoclaim()
                    # PR-4 spec §8.4 — auto-escalate REQUEST_CANCEL when the
                    # child hasn't ACKed within CHILD_CANCEL_ACK_TIMEOUT_MS.
                    await self._maybe_tick_cancel_check()
                    # PR-4 spec §7.5 — cascade TERMINATE when a child's
                    # last_seen_mono is older than
                    # SUBAGENT_PROGRESS_STALE_AFTER_SECONDS.
                    await self._maybe_tick_check_orphans()
                    # (C3 PR-5 rollback check now runs at the TOP of
                    # this try-block — before consuming entries — so
                    # the supervisor doesn't dispatch new mailbox work
                    # after the operator has flipped child rows to
                    # legacy. See ``_maybe_tick_check_rollback`` and
                    # codex r3 [HIGH CONTRACT] rationale.)
                    if not entries and self._idle_poll_sleep_s > 0:
                        await asyncio.sleep(self._idle_poll_sleep_s)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "supervisor loop iteration failed root=%s; sleep 0.5s",
                        self._ctx.root_session_id,
                    )
                    await asyncio.sleep(0.5)
        finally:
            self._stopped.set()

    async def stop(self, *, drain_timeout_s: float = 5.0) -> None:
        """Request stop and wait for the run-loop to actually exit.

        This is NOT a fire-and-forget flag — callers can rely on
        ``await sup.stop()`` to drain in-flight handler work, up to
        ``drain_timeout_s`` seconds. Past that we log a warning and return
        (the orphaned task is the caller's problem; this method must not
        block shutdown indefinitely).
        """
        self._stopping.set()
        try:
            await asyncio.wait_for(self._stopped.wait(), timeout=drain_timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "supervisor stop drain timed out root=%s after %ss",
                self._ctx.root_session_id,
                drain_timeout_s,
            )

    async def _handle_envelope(
        self, redis_id: bytes, envelope: MailboxEnvelope
    ) -> None:
        """Dispatch one envelope through its handler and ACK on success.

        - Cross-root envelope (parent_session_id mismatch) → log + ACK to
          drain (defense-in-depth — see comment below).
        - Unknown envelope type → log warning + ACK (drain; don't loop).
        - Handler raises → leave in PEL (no ACK). PR-3b XAUTOCLAIM retries
          until ``reclaim_count > MAILBOX_POISON_MAX_RECLAIM`` then drops to
          dead-letter sink.
        - ``side_effect`` raises → same PEL retain (no ACK).
        - ``outcome.ack == False`` → leave in PEL deliberately (defer).

        **Consumer-side audit dedup is PR-3b scope.** Per spec §5.8 the
        audit-repo ``get_processed → upsert_processing → mark_processed``
        lifecycle wires in PR-3b reliability work alongside XAUTOCLAIM, so
        a redelivered terminal envelope doesn't re-fire destroy(). PR-3b
        ships this in two places:

        1. ``get_processed`` check between the cross-root guard and the
           poison gate — short-circuits redelivery of envelopes that
           already ran their side_effect (codex r6 [HIGH] fix).
        2. ``mark_processed`` immediately before XACK on the
           side_effect-success path — the marker the get_processed check
           consumes on redelivery (codex r6 [HIGH] fix).

        XAUTOCLAIM bumps go through ``upsert_processing`` +
        ``increment_reclaim`` (PR-3b Phase B) so reclaim_count is
        durable across pod restarts.
        """
        # Codex r8 [P1] fix — cross-root defense-in-depth on
        # ``parent_session_id``. The XREADGROUP stream key already scopes
        # reads to one root, so a mismatch on parent_session_id implies a
        # publisher bug or a misrouted/forged envelope. We refuse to
        # dispatch and ACK to drain instead of looping; the warning
        # surfaces the publisher bug in production logs.
        #
        # codex r3 [R3-1, CRITICAL ARCH] — we deliberately do NOT verify
        # that ``envelope.child_session_id`` actually descends from
        # ``ctx.root_session_id``. Doing so would require a session-repo
        # ancestor-chain traversal per terminal envelope plus the
        # accompanying ``session_factory`` / ``AsyncSession`` lifecycle
        # plumbing into ``SupervisorContext`` (currently the only DB
        # touch is the audit repo). Decision: keep the trust assumption
        # explicit by **publisher contract** rather than defensive
        # verification:
        #
        #   Publisher contract — every envelope published to
        #   ``actus:child:{root_session_id}:mailbox`` MUST set
        #   ``envelope.parent_session_id = root_session_id`` AND
        #   ``envelope.child_session_id`` MUST be a session whose
        #   ancestor chain includes ``root_session_id``. The supervisor
        #   verifies the first invariant (this guard) and trusts the
        #   publisher for the second.
        #
        # Negative integration test in
        # ``test_mailbox_cross_root_guard.py`` makes the trust visible: a
        # cross-root child_session_id with a matching parent_session_id
        # proceeds through the supervisor unblocked.
        #
        # TODO(PR-5 / PR-6 acceptance gate): add defensive cross-root
        # ``child_session_id`` verification via cached session-repo lookup
        # on the terminal-handler path. Plumbing budget required:
        # ``session_factory`` (or a pre-built session repo) into
        # ``SupervisorContext``, ``AsyncSession`` lifecycle inside the
        # supervisor (per-lookup or pool), per-supervisor LRU cache
        # keyed by child_session_id → root_session_id, and a recursion-
        # bounded parent-chain walk since sessions store ``parent_session
        # _id`` rather than a denormalised root column. Tracked outside
        # PR-4 because it widens the supervisor's blast radius beyond
        # the audit repo for a defence-in-depth check that the publisher
        # contract is supposed to make redundant.
        if envelope.parent_session_id != self._ctx.root_session_id:
            logger.warning(
                "cross-root envelope refused — supervisor root=%s but "
                "envelope.parent_session_id=%s (id=%s type=%s) — dropping+ack",
                self._ctx.root_session_id,
                envelope.parent_session_id,
                envelope.envelope_id,
                envelope.type.value,
            )
            await self._consumer.ack(redis_id)
            return

        # PR-3b spec §5.8 layer 2 — consumer-side audit dedup (codex r6
        # [HIGH] fix). The crash window between ``side_effect`` success and
        # XACK is recoverable iff the supervisor marked the envelope as
        # processed BEFORE the XACK. On crash-then-redelivery via XAUTOCLAIM,
        # this check finds the existing ``processed_at`` row and ACKs without
        # re-firing the handler — preventing double destroy() in PR-4 once
        # terminal handlers wire real destructive side effects.
        #
        # ``get_processed`` failure is logged + treated as "not processed"
        # (proceed to handler). The cost of a false-negative is at-most-once
        # re-fire (current state). A false-positive (treating fresh as
        # processed) would silently lose work, so degrade safely toward the
        # already-current behavior.
        #
        # Placed BEFORE the poison gate: a poison envelope that was already
        # processed should drain via this short-circuit, not via the poison
        # path's telemetry+hook+ACK chain (no need to fire poison cascade
        # for an envelope whose terminal side_effect already ran).
        try:
            already_processed = await self._ctx.audit_repo.get_processed(
                envelope.parent_session_id, envelope.envelope_id
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "audit_repo.get_processed failed envelope=%s — proceeding "
                "to handler (degrade safely; PR-4 dedup may double-fire)",
                envelope.envelope_id,
            )
            already_processed = False
        if already_processed:
            logger.info(
                "envelope already processed (audit dedup hit) — ACK without "
                "re-firing handler id=%s type=%s",
                envelope.envelope_id,
                envelope.type.value,
            )
            await self._consumer.ack(redis_id)
            return

        # PR-3b spec §5.7 — poison drop. An envelope that has been reclaimed
        # more than MAILBOX_POISON_MAX_RECLAIM times is treated as stuck (the
        # handler chain is consistently failing for some non-transient reason)
        # — ACK + drop + telemetry so the stream stops re-fighting it.
        # Placed BEFORE last_seen refresh: a poison envelope's reclaim_count
        # implies it's been stuck for at least MAX sweeps, so treating it as
        # a fresh heartbeat would be misleading.
        if envelope.reclaim_count > MAILBOX_POISON_MAX_RECLAIM:
            # Codex r4 [P1] fix — telemetry.emit is best-effort here. If the
            # OTel/exporter backend raises (network blip, sink misconfig),
            # we still MUST reach the ACK below, otherwise the same
            # redelivery loop the hook isolation was meant to prevent
            # comes back via the telemetry path: envelope stays in PEL →
            # XAUTOCLAIM next cycle → poison gate re-fires → emit raises
            # again. Same risk as r1 [P1] for ``_on_poison_drop``; same
            # treatment (CancelledError escapes, broad Exception logged).
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.poison_message_dropped",
                    {
                        "envelope_id": envelope.envelope_id,
                        "type": envelope.type.value,
                        "child_session_id": envelope.child_session_id,
                        "reclaim_count": envelope.reclaim_count,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "poison drop telemetry raised envelope=%s — ACK anyway",
                    envelope.envelope_id,
                )
            # Spec §5.7 step 3 hook — if terminal type AND child still
            # RUNNING, trigger orphan cascade. PR-4 overrides this method.
            # Codex r1 [P1] fix — isolate hook exception. If a PR-4 override
            # raises (e.g., orphan-cascade publisher errors), we MUST still
            # ACK; otherwise the envelope stays in PEL, XAUTOCLAIM picks it
            # up next cycle, poison gate re-fires, hook re-raises — infinite
            # loop with ever-growing reclaim_count that never reaches ACK.
            try:
                await self._on_poison_drop(envelope)
            except asyncio.CancelledError:
                raise
            except _CascadeFailedError as exc:
                # codex r8 [R8-5, HIGH CONTRACT] — the poison-drop hook's
                # cascade emitted a synthetic CANCEL_REQUEST, the XADD
                # failed, AND the direct-kill fallback raised a retryable
                # ``SandboxLifecycleError`` (see ``_cascade_publish_failed_
                # direct_kill`` and ``_CascadeFailedError`` for the
                # state-machine derivation). Spec §5.7 step 3 mandates that
                # the poison drop hook MUST fire a cascade so the M2
                # invariant (every running child reaches destroy() OR
                # RESULT_READY) holds. If the cascade fails we still ACK
                # the poison envelope to break the redelivery loop (the
                # alternative is leaving it in PEL forever, which we
                # already proved unrecoverable past MAX reclaim). The
                # dedicated telemetry event below surfaces the cleanup
                # failure to ops alerts — the broad-except branch beneath
                # logs but doesn't emit a distinguishable signal, so ops
                # would have to grep ``logger.exception`` to notice. Emit
                # a structured event so this raises an alert instead.
                try:
                    await self._ctx.telemetry.emit(
                        "mailbox.poison_cascade_failed_critical",
                        {
                            "envelope_id": envelope.envelope_id,
                            "child_session_id": envelope.child_session_id,
                            "type": envelope.type.value,
                            "error": str(exc),
                        },
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "poison_cascade_failed_critical telemetry raised "
                        "envelope=%s — ACK still proceeds",
                        envelope.envelope_id,
                    )
                logger.error(
                    "poison drop hook cascade failed envelope=%s child=%s "
                    "— the orphaned child may not have been destroyed; "
                    "ACK proceeds to break the redelivery loop. See "
                    "telemetry mailbox.poison_cascade_failed_critical.",
                    envelope.envelope_id,
                    envelope.child_session_id,
                )
            except Exception:
                logger.exception(
                    "poison drop hook raised envelope=%s — ACK anyway to "
                    "break the redelivery loop",
                    envelope.envelope_id,
                )
            await self._consumer.ack(redis_id)
            return

        # PR-3b spec §9.2 — last_seen refresh BEFORE dispatch. Only child-
        # origin envelopes advance the heartbeat clock. ``_handle_envelope``
        # is the single funnel for both XREADGROUP fresh-delivery AND
        # XAUTOCLAIM redelivery, so this captures both paths.
        await self._refresh_last_seen_if_child_origin(envelope)

        # PR-3b spec §5.8 layer 2 — pre-stage the audit row BEFORE dispatch
        # so the side_effect-success path's ``mark_processed`` finds an
        # existing row (codex r8 [HIGH] fix). The DB impl raises ``ValueError``
        # on missing row (db_mailbox_envelope_audit_repository.py:141-145),
        # so without this upsert the fresh XREADGROUP delivery path would
        # silently lose its dedup marker when ``mark_processed`` raises and
        # the broad-except below logs + ACKs anyway — re-opening the
        # double-destroy hole codex r6 had closed for the XAUTOCLAIM paths
        # (XAUTOCLAIM already calls upsert before increment_reclaim).
        #
        # ``upsert_processing`` is a real PG ``ON CONFLICT DO UPDATE`` (see
        # db_mailbox_envelope_audit_repository.py lines 63-98), idempotent
        # whether the row exists or not. On failure we proceed to dispatch
        # without the row — best-effort degradation that preserves the
        # pre-r8 (broken) behavior, NOT a regression vs current state.
        try:
            await self._ctx.audit_repo.upsert_processing(
                envelope, processing_at=self._ctx.now()
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "audit_repo.upsert_processing failed envelope=%s — "
                "proceeding without dedup row (mark_processed may raise "
                "later; ACK still fires, accepts at-most-once dedup-loss)",
                envelope.envelope_id,
            )

        handler = self._dispatch.get(envelope.type)
        if handler is None:
            logger.warning(
                "no handler for envelope type %s root=%s id=%s — drop+ack",
                envelope.type.value,
                self._ctx.root_session_id,
                envelope.envelope_id,
            )
            await self._consumer.ack(redis_id)
            return

        try:
            outcome = await handler.handle(envelope, self._ctx)
        except Exception:
            logger.exception(
                "handler raised for envelope id=%s type=%s — leave in PEL for retry",
                envelope.envelope_id,
                envelope.type.value,
            )
            # Do NOT ACK; PR-3b XAUTOCLAIM will retry; reclaim_count > MAX → poison drop.
            return

        # Codex r2 [P1] fix: side_effect success implies ACK (per spec §6.x),
        # regardless of ``outcome.ack``. PR-4 terminal handlers return
        # ``ack=False, side_effect=destroy(...)`` — when destroy() returns
        # normally, the envelope MUST be ACKed to avoid redelivery + double
        # destruction. The prior implementation only ACKed when ``ack=True``,
        # which would have re-fired terminal side effects on the next
        # XAUTOCLAIM sweep once PR-3b landed.
        if outcome.side_effect is not None:
            try:
                await outcome.side_effect()
            except Exception:
                logger.exception(
                    "side_effect raised for envelope id=%s — leave in PEL",
                    envelope.envelope_id,
                )
                return  # do NOT ack — PEL retain for XAUTOCLAIM retry
            # Codex r4 [R4-3, HIGH CONTRACT] — spec §5.8 layer 2 dedup
            # marker MUST land BEFORE the XACK. The previous policy ACKed
            # even when ``mark_processed`` raised (rationale: avoid
            # double-destroy on redelivery). That reasoning was inverted:
            # if we ACK without a durable marker, the next XAUTOCLAIM
            # sweep would never replay this envelope at all, so the
            # supposed "double-destroy" was a no-op in steady state, and
            # the ACK loss was the false comfort. The real failure mode
            # is different: side_effect already ran (destroy succeeded),
            # mark_processed failed → if we ACK now, the audit row never
            # gets ``processed_at``, but the envelope is gone from the
            # PEL → fine in steady state. Where it breaks is when the
            # supervisor restarts mid-PEL with stale entries: the new
            # supervisor has no audit row, no PEL entry, and no way to
            # reconcile that this envelope was already handled. The audit
            # row IS the load-bearing dedup contract per spec §5.8 ("the
            # marker is the truth"), so without it the contract is
            # silently downgraded.
            #
            # The fix is to leave the entry in the PEL: XAUTOCLAIM
            # redelivers, the terminal handler's destroy is idempotent
            # (RESULT_READY / CANCEL_ACK / CANCEL_TERMINATE all classify
            # ``SandboxAlreadyDestroyed`` and ``SandboxBindingMissing``
            # as terminal-success), and ``mark_processed`` is retried.
            # When the DB recovers, the marker lands and the entry ACKs
            # cleanly. If the failure persists past
            # ``MAILBOX_POISON_MAX_RECLAIM`` sweeps, the poison-drop
            # gate at handle_envelope entry fires, telemetry records the
            # incident, and the orphan-cascade hook can take over.
            try:
                await self._ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=self._ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # §5.8 hard rule — do NOT ACK without a durable dedup
                # marker. Idempotent destroy on the terminal handlers
                # makes redelivery safe; poison-drop is the eventual
                # backstop if the DB outage persists.
                logger.exception(
                    "audit_repo.mark_processed failed envelope=%s — "
                    "leaving entry in PEL for XAUTOCLAIM retry. The "
                    "terminal handlers' destroy() is idempotent "
                    "(AlreadyDestroyed / BindingMissing → terminal-"
                    "success), so the next redelivery short-circuits to "
                    "mark_processed without re-firing side effects. If "
                    "the DB stays down past MAILBOX_POISON_MAX_RECLAIM "
                    "reclaim sweeps, the poison-drop gate fires.",
                    envelope.envelope_id,
                )
                return  # PEL retain — no ACK
            await self._consumer.ack(redis_id)
            return

        # No side_effect — honor ``outcome.ack`` as the explicit ACK/defer
        # signal. ``ack=False`` is the legitimate "defer" path PR-3b reliability
        # layer relies on (e.g., dedup hit waiting for the in-flight handler).
        #
        # Codex F8 (HIGH) — mark_processed runs symmetrically with the
        # side_effect path. Handlers that return ``ack=True`` with no
        # side_effect (ApprovalRequestHandler stub, _StubNonTerminalHandler
        # for PROGRESS_UPDATE / SPAWN_REQUEST / ..., dedup-hit returns) MUST
        # leave a ``processed_at`` row so a redelivered envelope is detected
        # by the ``get_processed`` short-circuit at envelope entry. Without
        # this write, T7's ``test_duplicate_approval_request_dedups_via_audit``
        # asserts ``processed_at is not None`` and would FAIL on a real DB
        # run; more importantly, an ApprovalRequest that survives the SET NX
        # publisher window (e.g., publishers across pods) would re-fire the
        # deny response on every XAUTOCLAIM redelivery.
        #
        # For dedup-hit returns (where ``get_processed`` was already true at
        # handler entry), this is a no-op — the DB UPDATE preserves the
        # existing ``processed_at``. Safe + idempotent.
        if outcome.ack:
            # codex r5 [R5-4, HIGH CONTRACT] — mirror of the R4-3 fix on the
            # side_effect path. spec §5.8 hard rule: do NOT ACK without a
            # durable dedup marker. The previous policy ACKed even when
            # ``mark_processed`` raised, which broke the same invariant
            # R4-3 fixed for the side_effect path:
            #
            #   ApprovalRequestHandler returns ack=True after publishing
            #   APPROVAL_RESPONSE. If mark_processed fails and we ACK
            #   anyway, the envelope is gone from the PEL → no XAUTOCLAIM
            #   replay → audit row never gets ``processed_at`` → on the
            #   rare publisher-cross-pod redelivery window the deny
            #   response would re-fire. More importantly the audit trail
            #   silently downgrades the "the marker is the truth"
            #   contract.
            #
            # Fix: leave entry in PEL. XAUTOCLAIM redelivers; the handler's
            # idempotent path (publisher SET NX dedups APPROVAL_RESPONSE;
            # _StubNonTerminalHandler re-fires its callback — cheap)
            # re-runs and ``mark_processed`` is retried. When the DB
            # recovers, the marker lands and the entry ACKs cleanly. If
            # the failure persists past ``MAILBOX_POISON_MAX_RECLAIM``
            # sweeps, the poison-drop gate fires.
            try:
                await self._ctx.audit_repo.mark_processed(
                    envelope.parent_session_id,
                    envelope.envelope_id,
                    processed_at=self._ctx.now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "audit_repo.mark_processed failed on ack=True "
                    "envelope=%s — leaving entry in PEL for XAUTOCLAIM "
                    "retry per §5.8 (handler is idempotent: publisher "
                    "SET NX dedups APPROVAL_RESPONSE; stub callback "
                    "re-fire is cheap). Matches the R4-3 side_effect fix.",
                    envelope.envelope_id,
                )
                return  # PEL retain — no ACK
            await self._consumer.ack(redis_id)

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase A — heartbeat last_seen gate (spec §9.2)
    # ──────────────────────────────────────────────────────────────────────

    def _is_child_origin(self, envelope: MailboxEnvelope) -> bool:
        """Spec §9.2 — only envelopes that (i) belong to a child session,
        (ii) are in the child→parent direction (CHILD_TO_PARENT_TYPES), and
        (iii) were produced by the child agent itself (NOT the supervisor's
        echo / external publisher) tell us the child is alive.
        """
        return (
            bool(envelope.child_session_id)
            and envelope.type in CHILD_TO_PARENT_TYPES
            and envelope.producer_role == ProducerRole.CHILD_AGENT
        )

    async def _refresh_last_seen_if_child_origin(
        self, envelope: MailboxEnvelope
    ) -> None:
        if not self._is_child_origin(envelope):
            return
        liveness = self._ctx.liveness_service
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        kind = payload.get("kind")
        kind_value = kind.value if hasattr(kind, "value") else kind
        if (
            liveness is not None
            and envelope.type == MailboxEnvelopeType.PROGRESS_UPDATE
            and kind_value == ProgressKind.HEARTBEAT.value
        ):
            accepted = await liveness.record_heartbeat(envelope)
            if not accepted:
                return
            if envelope.child_session_id not in self._known_children:
                self._known_children.append(envelope.child_session_id)
        elif liveness is not None:
            # A SPAWN_ACK or ordinary progress event can be the first event
            # observed after dispatch. If the DB-authorized startup lease is
            # already present, identify the child as coordinator-owned but do
            # not turn this non-heartbeat event into fresh liveness. From this
            # point orphan detection reads the durable Redis authority.
            if envelope.child_session_id in self._known_children:
                return
            get_lease = getattr(liveness, "get_lease", None)
            if get_lease is not None:
                lease = await get_lease(envelope.child_session_id)
                if lease is not None:
                    if (
                        lease.root_session_id == self._ctx.root_session_id
                        and lease.parent_session_id == self._ctx.root_session_id
                        and lease.child_session_id == envelope.child_session_id
                    ):
                        self._known_children.append(envelope.child_session_id)
                    # A live-but-mismatched lease is not legacy. Fail closed
                    # instead of granting it a local-clock liveness path.
                    return
        self._last_seen_mono[envelope.child_session_id] = self._ctx.clock()

    def get_last_seen(self, child_session_id: str) -> Optional[float]:
        """Public read accessor for the orphan-watch / cascade tick logic
        (PR-4 wires the consumer). Returns ``None`` if we've never observed
        a child-origin envelope for this child on this supervisor instance.
        """
        return self._last_seen_mono.get(child_session_id)

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase B — XAUTOCLAIM startup + periodic (spec §5.6)
    # ──────────────────────────────────────────────────────────────────────

    async def _initial_xautoclaim(self) -> None:
        """Spec §5.6 startup XAUTOCLAIM — bring orphaned PEL from prior dead
        consumer into this supervisor's ownership.

        Startup sweep claims every entry in PEL regardless of idle time
        (``min_idle_ms=0``). Used when this supervisor instance takes over a
        root from a dead consumer (the dead consumer can't ACK, so XPENDING
        will still show its idle entries). ``count=1000`` is a soft batch
        cap; if there are more we'll catch the rest on the periodic tick.

        INVARIANT: ``min_idle_ms=0`` relies on the M1 single-writer-per-root
        invariant (one MailboxSupervisor per ``root_session_id`` globally,
        enforced by PR-3c's agent_task_runner integration + reconcile_orphans).
        If M1 is violated (e.g., overlap during rolling pod restart with a
        stale supervisor still active on a different pod), this WILL steal
        entries from the active sibling. The ``SupervisorRegistry`` is
        per-pod only; cross-pod M1 enforcement is the deploy-time
        responsibility (pod lifecycle hooks + reconcile_orphans).
        Spec-design decision per plan §5.6; codex r2 [P1] verified as not an
        impl bug — documenting reliance here so future operators don't relax
        ``min_idle_ms`` without first re-evaluating the M1 contract.
        """
        try:
            claimed = await self._consumer.autoclaim(
                min_idle_ms=0,  # see INVARIANT above — M1 ensures sibling is dead
                count=1000,
            )
            for redis_id, envelope in claimed:
                # Codex r2 [P1] fix — ensure audit row exists BEFORE
                # ``increment_reclaim``. The DB impl raises ``ValueError`` on
                # missing row (see ``db_mailbox_envelope_audit_repository.py``
                # lines 141-145). Without this upsert the broad-except below
                # would mask that ValueError, falling back to
                # ``envelope.reclaim_count + 1`` — but ``envelope.reclaim_count``
                # is fresh-parsed from the stream entry on every XAUTOCLAIM
                # sweep, so the fallback would forever return 1 and the poison
                # drop gate (``> MAILBOX_POISON_MAX_RECLAIM``) could never trip.
                # ``upsert_processing`` is a real PG ``ON CONFLICT DO UPDATE``
                # (idempotent — see same file lines 63-98), so calling it
                # before every reclaim is safe even when the row already
                # exists.
                #
                # Codex r1 [P0] fix (preserved) — persist reclaim_count via
                # audit_repo. Without DB persistence the in-memory
                # ``model_copy`` increment was lost on the next XAUTOCLAIM
                # sweep (the stream entry's raw envelope is re-parsed each
                # time), so poison drop never fired. Initial sweep entries
                # are by definition reclaims (the dead consumer couldn't
                # ACK), so each claimed entry counts as a retry for the
                # poison threshold.
                try:
                    await self._ctx.audit_repo.upsert_processing(
                        envelope, processing_at=self._ctx.now()
                    )
                    persisted_count = await self._ctx.audit_repo.increment_reclaim(
                        envelope.parent_session_id,
                        envelope.envelope_id,
                        "initial_xautoclaim_after_restart",
                    )
                except Exception:
                    logger.exception(
                        "audit_repo upsert+increment_reclaim failed envelope=%s "
                        "— best-effort fallback to in-memory bump",
                        envelope.envelope_id,
                    )
                    persisted_count = envelope.reclaim_count + 1
                envelope = envelope.model_copy(
                    update={"reclaim_count": persisted_count}
                )
                # Codex r3 [P1] fix — per-entry exception isolation. XAUTOCLAIM
                # has already transferred ownership of the entire claimed batch
                # to THIS supervisor; if ``_handle_envelope`` raises for one
                # entry (e.g., XACK transient failure) and the loop aborts,
                # the rest of the batch is now pending under this consumer
                # and will NOT be redelivered via XREADGROUP ``>``. The skipped
                # entries would only resurface on the next XAUTOCLAIM sweep
                # after ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` (60s default), so
                # one transient failure could starve up to 999 startup-claimed
                # orphans for a minute. Mirror the main-loop pattern (codex
                # r10 [P1] fix in PR-3a): re-raise CancelledError, broadly
                # log+continue everything else.
                try:
                    await self._handle_envelope(redis_id, envelope)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "initial XAUTOCLAIM entry handling failed root=%s "
                        "id=%s type=%s — leave in PEL, continue batch",
                        self._ctx.root_session_id,
                        redis_id,
                        envelope.type.value,
                    )
            # Codex r2 [P2] + r5 [P2] fix — anchor the periodic-tick clock so
            # the first ``_maybe_periodic_xautoclaim`` call waits one full
            # interval instead of firing immediately. ``time.monotonic()``
            # returns process-uptime in seconds (typically >> 30 immediately
            # after boot of any non-trivial process), so leaving
            # ``self._last_autoclaim_mono`` at ``0.0`` causes the periodic
            # tick to fire on the very first loop iteration — doubling the
            # XAUTOCLAIM work right after ``_initial_xautoclaim`` (which
            # already claimed everything with ``min_idle=0``).
            #
            # r5 [P2]: anchor now lands BEFORE the closing telemetry emit.
            # If telemetry raises (network blip, sink misconfig), the outer
            # broad-except below catches it and the anchor would otherwise
            # be skipped — re-opening r2's premature-tick bug via the
            # observability path. Placement is still INSIDE the try so an
            # earlier XAUTOCLAIM failure leaves the anchor at 0.0 (degrades
            # safely — periodic attempts ASAP).
            self._last_autoclaim_mono = self._ctx.clock()
            # Codex r5 [P2] — telemetry.emit is best-effort here. Isolating
            # it separately means a sink crash doesn't get re-logged as
            # "initial XAUTOCLAIM failed" via the outer broad-except, which
            # would be misleading observability for an observability fault.
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.initial_autoclaim_completed",
                    {"root": self._ctx.root_session_id, "count": len(claimed)},
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "initial_autoclaim_completed telemetry raised root=%s "
                    "— continuing without the emit",
                    self._ctx.root_session_id,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "initial XAUTOCLAIM failed root=%s", self._ctx.root_session_id
            )

    async def _maybe_periodic_xautoclaim(self) -> None:
        """Periodic sweep — runs every ``_XAUTOCLAIM_INTERVAL_S`` seconds
        from the main loop. Uses ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` so we
        don't steal entries the original consumer might still be processing.
        """
        now = self._ctx.clock()
        if now - self._last_autoclaim_mono < self._XAUTOCLAIM_INTERVAL_S:
            return
        self._last_autoclaim_mono = now
        try:
            claimed = await self._consumer.autoclaim(
                min_idle_ms=self._XAUTOCLAIM_MIN_IDLE_MS,
                count=100,
            )
            for redis_id, envelope in claimed:
                # Codex r2 [P1] fix — ensure audit row exists BEFORE
                # ``increment_reclaim``. The DB impl raises ``ValueError``
                # on missing row (``db_mailbox_envelope_audit_repository.py``
                # lines 141-145). Without this upsert the broad-except below
                # would mask that ValueError, falling back to
                # ``envelope.reclaim_count + 1`` which is forever 1 (the
                # stream entry's raw envelope is fresh-parsed each sweep),
                # so the poison drop gate (``> MAILBOX_POISON_MAX_RECLAIM``)
                # could never trip. ``upsert_processing`` is a real PG
                # ``ON CONFLICT DO UPDATE`` (idempotent — see same file
                # lines 63-98), safe to call before every reclaim.
                #
                # Codex r1 [P0] fix (preserved) — persist reclaim_count via
                # audit_repo. See ``_initial_xautoclaim`` for the full
                # rationale: the prior ``model_copy`` rebind was a transient
                # bump that never made it back to the stream entry, so
                # subsequent XAUTOCLAIM sweeps re-parsed the original
                # ``reclaim_count`` and the poison drop gate
                # (``> MAILBOX_POISON_MAX_RECLAIM``) was effectively dead
                # code in production.
                try:
                    await self._ctx.audit_repo.upsert_processing(
                        envelope, processing_at=self._ctx.now()
                    )
                    persisted_count = await self._ctx.audit_repo.increment_reclaim(
                        envelope.parent_session_id,
                        envelope.envelope_id,
                        "xautoclaim_redelivered",
                    )
                except Exception:
                    logger.exception(
                        "audit_repo upsert+increment_reclaim failed envelope=%s "
                        "— best-effort fallback to in-memory bump",
                        envelope.envelope_id,
                    )
                    persisted_count = envelope.reclaim_count + 1
                envelope = envelope.model_copy(
                    update={"reclaim_count": persisted_count}
                )
                # Codex r3 [P1] fix — per-entry exception isolation. Same
                # rationale as ``_initial_xautoclaim``: XAUTOCLAIM transferred
                # ownership of the batch to THIS supervisor; aborting the
                # loop on one entry's failure would leave the rest pending
                # under this consumer (not redelivered via XREADGROUP ``>``)
                # for another ``MAILBOX_PEL_IDLE_MS_FOR_CLAIM`` cycle.
                try:
                    await self._handle_envelope(redis_id, envelope)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "periodic XAUTOCLAIM entry handling failed root=%s "
                        "id=%s type=%s — leave in PEL, continue batch",
                        self._ctx.root_session_id,
                        redis_id,
                        envelope.type.value,
                    )
            await self._ctx.telemetry.emit(
                "mailbox.periodic_autoclaim_tick",
                {"root": self._ctx.root_session_id, "count": len(claimed)},
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "periodic XAUTOCLAIM failed root=%s", self._ctx.root_session_id
            )

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase C — poison drop hook (spec §5.7)
    # ──────────────────────────────────────────────────────────────────────

    async def _on_poison_drop(self, envelope: MailboxEnvelope) -> None:
        """Spec §5.7 step 3 — when a *terminal-type* poison envelope is dropped
        and a child_session_id is present, fire a synthetic
        CANCEL_REQUEST(TERMINATE) so the orphaned child still gets destroyed.

        Without this hook the M2 invariant (every running child must reach
        either RESULT_READY or destroy()) silently breaks: the poison drop
        ACKs the envelope and we lose the only signal that the child needs
        cleanup. Cascading via a fresh CANCEL_REQUEST routes through the
        same TERMINATE side-effect chain that ``CancelRequestHandler``
        already exercises, including stop_session and
        ``destroy(ORPHAN_TIMEOUT)`` (R2-6 thread, R3-8 doc).

        codex r8 [R8-4, MEDIUM PERF] — if the *dropped* envelope is itself
        a synthetic cascade CANCEL_REQUEST that has poisoned (we have
        already exhausted XAUTOCLAIM retries against it), clean up the
        matching ``_cascade_destroy_overrides`` entry. ``_emit_cascade_terminate``
        stamps the override BEFORE publishing the synthetic envelope, and
        the only pop sites are the successful side_effect path inside
        ``CancelRequestHandler._terminate_outcome`` (R7-4) and the
        publish-failure fallback inside ``_emit_cascade_terminate``
        (R6-3). If the synthetic envelope poisons via XAUTOCLAIM
        redelivery (not via XADD failure), neither pop site runs and the
        side-table accumulates one stale entry per poisoned cascade — a
        slow leak in long-lived supervisors. Capture-then-pop so the
        cascade CANCEL_REQUEST branch below can reuse the override.

        codex r10 [R10-1, HIGH ARCH] — when the *dropped* envelope is a
        synthetic cascade CANCEL_REQUEST with policy=TERMINATE, the
        orphan/cancel-tick that produced it has already cleared
        ``_last_seen_mono[child]`` / ``_cancel_states[child]`` (the
        publish succeeded so the early-clear was correct from the
        tick's POV). If the handler then exhausts retries via
        XAUTOCLAIM, the original ACKed-and-dropped contract leaves the
        child with no remaining retry signal: no tracking entry to
        re-fire the tick, no envelope in the PEL. Cascade the
        emergency direct-kill path here so the orphaned child still
        gets destroyed, reusing the side-table's
        ``destroy_reason`` override (so ORPHAN_TIMEOUT cascades keep
        their forensic tag; non-tagged cascades fall back to
        FORCE_TERMINATE — same default the handler would have used).
        """
        # R8-4 — capture-then-drop any stale cascade override keyed on this
        # envelope_id. We must capture before popping so the cascade
        # CANCEL_REQUEST direct-kill branch below can reuse the override.
        # Non-cascade envelopes never appear in the side-table so the get
        # is a no-op None for them.
        cascade_override = self._cascade_destroy_overrides.pop(
            envelope.envelope_id, None
        )

        # codex r10 [R10-1, HIGH ARCH] — cascade CANCEL_REQUEST poison drop
        # branch. The synthetic envelope we published earlier has
        # exhausted retries; tracking is already gone (orphan/cancel
        # tick cleared it post-publish), so the child has no remaining
        # retry signal and the ACKed-and-dropped envelope would
        # otherwise leak the orphan. Fall back to the direct-kill
        # emergency path (callback → destroy) so M2 still holds.
        if (
            envelope.type == MailboxEnvelopeType.CANCEL_REQUEST
            and envelope.child_session_id
        ):
            try:
                payload = CancelRequestPayload.model_validate(envelope.payload)
            except Exception:
                payload = None
            if payload is not None and payload.policy == CancelPolicy.TERMINATE:
                await self._cascade_publish_failed_direct_kill(
                    envelope.child_session_id,
                    reason="poison_drop_cascade_envelope",
                    destroy_reason=cascade_override
                    or DestroyReason.FORCE_TERMINATE,
                    synthetic_envelope_id=envelope.envelope_id,
                    synthetic_envelope=envelope,
                )
                return

        terminal_types = {
            MailboxEnvelopeType.RESULT_READY,
            MailboxEnvelopeType.CANCEL_ACK,
        }
        if envelope.type in terminal_types and envelope.child_session_id:
            # codex r2 [R2-6, HIGH CONTRACT] — poison-drop fallback is
            # an orphan-cleanup variant (the child is unreachable per
            # the failing destroy retries). Thread ORPHAN_TIMEOUT so the
            # eventual destroy lands in the DB with the right reason
            # tag for ops/forensics rather than FORCE_TERMINATE.
            await self._emit_cascade_terminate(
                envelope.child_session_id,
                reason="poison_drop_terminal_envelope",
                destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
            )

    # ──────────────────────────────────────────────────────────────────────
    # PR-3b Phase D — pod-restart clock recovery (spec §9.3)
    # ──────────────────────────────────────────────────────────────────────

    async def _restore_last_seen_after_pod_restart(self) -> None:
        """Spec §9.3 — explicit Redis-time-anchored clock recovery.

        After a pod restart, monotonic clocks reset to 0. We can't trust the
        envelope's ``emitted_at`` (wall-clock from a possibly-skewed publisher),
        so we use the redis stream's entry id (millis since Unix epoch on
        the Redis server) as the reference point:

            age_ms = redis_now_ms - entry_ms        # how long ago the envelope was added
            last_seen_mono = clock_now - age_ms/1000  # convert into local monotonic frame

        codex r9b [R9b-1, HIGH ARCH] — caller wiring deferred to PR-5.
        PR-3b/PR-3c originally scoped ``SupervisorRegistry.spawn`` to
        populate ``_known_children`` from the session repo at supervisor
        startup and call this routine after ``ensure_group()``. That
        plumbing widens PR-4's blast radius (``session_factory`` /
        ``AsyncSession`` lifecycle + ancestor-chain walk into
        ``SupervisorContext``) for an additive recovery optimisation, so
        PR-4 ships the routine plus its in-isolation unit coverage and
        the actual wiring follows in PR-5 alongside the deferred R3-1
        cross-root child_session_id verification (same plumbing budget;
        see the comment block on ``_known_children`` in ``__init__``).
        Runtime safety while unwired: ``_known_children`` defaults to
        ``[]`` so the for-loop is a no-op and the next child-origin
        envelope refreshes ``_last_seen_mono`` via the normal dispatch
        path (``_maybe_advance_last_seen``).

        Idempotent — if XREVRANGE / TIME both fail we just skip; the next
        child-origin envelope will refresh ``_last_seen_mono`` naturally.
        """
        now_mono = self._ctx.clock()
        try:
            redis_time = await self._ctx.redis.time()  # (seconds, microseconds)
            redis_now_ms = redis_time[0] * 1000 + redis_time[1] // 1000
        except Exception:
            logger.warning(
                "redis TIME failed during clock recovery root=%s — skipping",
                self._ctx.root_session_id,
            )
            return

        stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
            root_session_id=self._ctx.root_session_id
        )
        for child_id in self._known_children:
            entry_ms = await self._latest_child_origin_entry_ms(
                stream_key, child_id
            )
            if entry_ms is None:
                continue
            age_ms = max(0, redis_now_ms - entry_ms)
            self._last_seen_mono[child_id] = now_mono - (age_ms / 1000.0)

    async def _restore_coordinator_liveness_after_startup(self) -> None:
        """Recover RUNNING coordinator children without respawning runners."""
        repo = self._ctx.session_repo
        liveness = self._ctx.liveness_service
        finder = (
            getattr(repo, "find_running_mailbox_children_for_parent", None)
            if repo is not None
            else None
        )
        if liveness is None or finder is None:
            await self._restore_last_seen_after_pod_restart()
            return

        rows = await finder(self._ctx.root_session_id)
        now = self._ctx.clock()
        redis_now_ms: int | None = None
        restored: list[str] = []
        for row in rows:
            child_id = str(row.session_id)
            run_id = getattr(row, "coordinator_run_id", None)
            work_unit_id = getattr(row, "work_unit_id", None)
            if not run_id or not work_unit_id:
                continue
            lease = await liveness.get_lease(child_id)
            lease_matches = bool(
                lease is not None
                and lease.root_session_id == self._ctx.root_session_id
                and lease.parent_session_id == self._ctx.root_session_id
                and lease.child_session_id == child_id
                and lease.coordinator_run_id == run_id
                and lease.work_unit_id == work_unit_id
            )
            if not lease_matches:
                try:
                    # Durable lease is the first authority. If it is absent,
                    # recover the latest trusted child-origin stream age;
                    # only a child with neither source receives a fresh 90s
                    # startup grace window.
                    startup_age_seconds: float | None = None
                    if lease is None:
                        stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
                            root_session_id=self._ctx.root_session_id
                        )
                        entry_ms = await self._latest_child_origin_entry_ms(
                            stream_key, child_id,
                        )
                        if entry_ms is not None:
                            if redis_now_ms is None:
                                redis_now_ms = await self._redis_server_now_ms()
                            startup_age_seconds = max(
                                0.0,
                                (redis_now_ms - entry_ms) / 1_000.0,
                            )
                    lease = await liveness.record_startup_lease(
                        root_session_id=self._ctx.root_session_id,
                        parent_session_id=self._ctx.root_session_id,
                        child_session_id=child_id,
                        coordinator_run_id=str(run_id),
                        work_unit_id=str(work_unit_id),
                        last_seen_age_seconds=startup_age_seconds,
                    )
                except Exception as exc:
                    from app.application.services.coordinator_liveness_lease_service import (
                        CoordinatorLivenessLeaseRejected,
                    )
                    if not isinstance(exc, CoordinatorLivenessLeaseRejected):
                        raise
                    # A concurrent terminal handler may have installed its
                    # tombstone after the RUNNING query. Do not invent a
                    # runner or local liveness signal; that terminal envelope
                    # remains in the stream/PEL and owns cleanup.
                    logger.info(
                        "coordinator startup lease registration lost race "
                        "root=%s child=%s",
                        self._ctx.root_session_id,
                        child_id,
                        exc_info=True,
                    )
                    continue
            restored.append(child_id)
            # Compatibility/read accessor only. Orphan authority below reads
            # the durable lease and its exact age, not this startup timestamp.
            self._last_seen_mono[child_id] = now
        self._known_children = restored

    async def _redis_server_now_ms(self) -> int:
        """Return Redis TIME in milliseconds or fail closed.

        Stream IDs and this timestamp share the same Redis clock. Application
        wall/monotonic clocks must not participate in cross-pod recovery age.
        """
        value = await self._ctx.redis.time()
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("Redis TIME must contain seconds and microseconds")

        def parse_component(component: object, name: str) -> int:
            if isinstance(component, bool):
                raise TypeError(f"Redis TIME {name} must be an integer")
            if isinstance(component, int):
                return component
            if isinstance(component, bytes):
                component = component.decode("ascii")
            if isinstance(component, str):
                stripped = component.strip()
                if stripped.isdigit():
                    return int(stripped)
            raise TypeError(f"Redis TIME {name} must be an integer")

        seconds = parse_component(value[0], "seconds")
        microseconds = parse_component(value[1], "microseconds")
        if seconds < 0 or not 0 <= microseconds < 1_000_000:
            raise ValueError("Redis TIME components are outside their valid range")
        return seconds * 1_000 + microseconds // 1_000

    async def _latest_child_origin_entry_ms(
        self, stream_key: str, child_id: str
    ) -> Optional[int]:
        """XREVRANGE scan for the latest child-origin envelope for one
        child_id. Returns the millisecond component of the Redis stream
        entry id (which is ``<ms>-<seq>``).

        Caps at 100 entries scanned per child — assumes a per-child
        progress cadence faster than that across the recent stream history,
        which the spec §4.3 SUBAGENT_PROGRESS_HEARTBEAT_INTERVAL_SECONDS=15
        guarantees in practice.
        """
        try:
            entries = await self._ctx.redis.xrevrange(stream_key, count=100)
        except Exception:
            return None
        for redis_id, fields in entries:
            raw = fields.get(b"envelope") or fields.get("envelope")
            if raw is None:
                continue
            try:
                env = MailboxEnvelope.model_validate_json(
                    raw.decode() if isinstance(raw, bytes) else raw
                )
            except Exception:
                continue
            payload = env.payload if isinstance(env.payload, dict) else {}
            kind = payload.get("kind")
            kind_value = kind.value if hasattr(kind, "value") else kind
            if (
                env.parent_session_id == self._ctx.root_session_id
                and env.child_session_id == child_id
                and env.type == MailboxEnvelopeType.PROGRESS_UPDATE
                and env.producer_role == ProducerRole.CHILD_AGENT
                and kind_value == ProgressKind.HEARTBEAT.value
                and env.correlation_id == f"hb:{child_id}"
            ):
                id_str = (
                    redis_id.decode()
                    if isinstance(redis_id, bytes)
                    else redis_id
                )
                ms_part = id_str.split("-")[0]
                try:
                    return int(ms_part)
                except ValueError:
                    continue
        return None

    # ──────────────────────────────────────────────────────────────────────
    # PR-4 cascade — register_cancel_state + tick orchestration (spec §7.5 + §8)
    # ──────────────────────────────────────────────────────────────────────

    async def _register_cancel_state(
        self, child_id: str, policy: CancelPolicy, mono: float
    ) -> None:
        """Bound to ``SupervisorContext.register_cancel_state`` in __init__.

        Stashes the cascade state for ``_maybe_tick_cancel_check`` to read on
        the next tick. The hook is async to keep symmetry with the rest of
        ``SupervisorContext``'s callables; the actual work is pure dict
        assignment.

        codex r6 [R6-1, HIGH ARCH] — preserve the earliest
        ``requested_at_mono`` when the same child receives another
        REQUEST_CANCEL. Pre-fix this unconditionally overwrote the entry,
        which meant XAUTOCLAIM redelivery (or any second CANCEL_REQUEST
        for the same child) would reset the timestamp and the §8.4 auto-
        escalate tick would see "fresh" state that hasn't yet exceeded
        ``CHILD_CANCEL_ACK_TIMEOUT_MS`` — the escalate timer would never
        fire. The TERMINATE transition is the only legitimate path that
        re-keys the cancel state for a given child; that path calls
        ``clear_child_tracking`` (via the terminal handler side_effect)
        BEFORE the next register call lands, so the slot is empty when a
        TERMINATE-triggered registration arrives and the policy-mismatch
        guard below would fall through to the fresh assignment anyway.
        """
        existing = self._cancel_states.get(child_id)
        if (
            existing is not None
            and existing.policy == CancelPolicy.REQUEST_CANCEL
            and policy == CancelPolicy.REQUEST_CANCEL
        ):
            # Same child, same REQUEST_CANCEL policy → second envelope
            # MUST NOT reset the auto-escalate timer. No-op.
            return
        self._cancel_states[child_id] = _CancelState(
            child_session_id=child_id,
            policy=policy,
            requested_at_mono=mono,
        )

    def _clear_child_tracking(self, child_id: str) -> None:
        """Bound to ``SupervisorContext.clear_child_tracking`` in __init__.

        Codex F7+F9 (HIGH) — drop both ``_cancel_states[child]`` and
        ``_last_seen_mono[child]`` after a terminal handler destroys the
        child. Without this cleanup:

        * F7: ``_cancel_states`` lingers with policy=REQUEST_CANCEL after
          CANCEL_ACK arrives → auto-escalate tick (spec §8.4) sees stale
          state aged past ``CHILD_CANCEL_ACK_TIMEOUT_MS`` → fires synthetic
          CANCEL_REQUEST(TERMINATE) → CancelRequestHandler hits
          ``SandboxAlreadyDestroyed`` (idempotent terminal-success) → spurious
          envelope spam + wasted lifecycle work on every tick.

        * F9: ``_last_seen_mono`` lingers after RESULT_READY / CANCEL_ACK /
          TERMINATE → orphan detector (spec §7.5) sees stale heartbeat past
          ``SUBAGENT_PROGRESS_STALE_AFTER_SECONDS`` → fires synthetic
          CANCEL_REQUEST(TERMINATE) → AlreadyDestroyed → same spurious work.

        Synchronous because cleanup is pure dict pop with no I/O — keeps the
        terminal handler side_effect simple and avoids spawning an event-loop
        round-trip per call.
        """
        self._cancel_states.pop(child_id, None)
        self._last_seen_mono.pop(child_id, None)
        try:
            self._known_children.remove(child_id)
        except ValueError:
            pass

    async def _maybe_tick_cancel_check(self) -> None:
        """Spec §8.4 — promote stuck REQUEST_CANCEL → TERMINATE.

        Runs no more often than every ``_CANCEL_CHECK_INTERVAL_S`` from the
        main loop. For each tracked ``_CancelState`` with policy
        REQUEST_CANCEL whose ``requested_at_mono`` has aged past
        ``CHILD_CANCEL_ACK_TIMEOUT_MS``, publishes a synthetic
        CANCEL_REQUEST(TERMINATE) — re-dispatched on the next loop iteration
        through ``CancelRequestHandler``'s TERMINATE branch — and clears the
        in-memory state so we don't double-cascade.

        ``CANCEL_AUTO_ESCALATE_TO_TERMINATE=False`` short-circuits the tick
        entirely for operators who need the spec §8.4 promotion disabled.
        """
        # Read the flag through this module's namespace so tests can flip
        # the behavior via monkeypatch on the supervisor module path
        # (the import-time constant value is otherwise frozen by Python).
        from app.application.services import (
            mailbox_supervisor as _ms,
        )  # local import keeps the cycle minimal

        if not _ms.CANCEL_AUTO_ESCALATE_TO_TERMINATE:
            return

        now = self._ctx.clock()
        if now - self._last_cancel_check_mono < self._CANCEL_CHECK_INTERVAL_S:
            return
        self._last_cancel_check_mono = now

        for child_id, state in list(self._cancel_states.items()):
            if state.policy != CancelPolicy.REQUEST_CANCEL:
                continue
            elapsed_ms = (now - state.requested_at_mono) * 1000.0
            if elapsed_ms <= CHILD_CANCEL_ACK_TIMEOUT_MS:
                continue
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.cascade_auto_escalate_terminate",
                    {
                        "child_session_id": child_id,
                        "elapsed_ms": int(elapsed_ms),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_auto_escalate_terminate telemetry raised root=%s "
                    "child=%s — continuing without the emit",
                    self._ctx.root_session_id,
                    child_id,
                )
            try:
                await self._emit_cascade_terminate(
                    child_id, reason="cancel_ack_timeout"
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_terminate publish failed root=%s child=%s — "
                    "leaving state in place for next tick to retry",
                    self._ctx.root_session_id,
                    child_id,
                )
                continue
            # Only drop the state after the publish succeeds — otherwise a
            # transient Redis blip would silently swallow the promotion.
            #
            # codex r10 [R10-1, HIGH ARCH] — same shape as the orphan tick
            # at the bottom of ``_maybe_tick_check_orphans``: per-child
            # tracking is dropped immediately post-publish. If the
            # synthetic cascade envelope then exhausts XAUTOCLAIM retries
            # and reaches ``_on_poison_drop`` (handler chain failing), the
            # child would have NO remaining retry signal (tracking gone +
            # envelope ACKed). The cascade-CANCEL_REQUEST branch in
            # ``_on_poison_drop`` catches that case and fires the
            # direct-kill fallback so the orphan still gets destroyed.
            self._cancel_states.pop(child_id, None)

    async def _maybe_tick_check_orphans(self) -> None:
        """Spec §7.5 — orphan detection.

        Runs every ``_ORPHAN_CHECK_INTERVAL_S`` from the main loop. For each
        ``last_seen_mono`` entry older than
        ``SUBAGENT_PROGRESS_STALE_AFTER_SECONDS``, publish a synthetic
        ``CANCEL_REQUEST(TERMINATE, reason="orphan_timeout")`` AND populate
        the supervisor-private ``_cascade_destroy_overrides[envelope_id]``
        side-table with ``DestroyReason.ORPHAN_TIMEOUT``. ``CancelRequest
        Handler._terminate_outcome`` reads that side-table during dispatch
        so the eventual ``sandbox_lifecycle.destroy`` call is tagged
        ``ORPHAN_TIMEOUT`` rather than the default ``FORCE_TERMINATE``.
        After a successful publish the heartbeat slot is cleared so the
        next tick doesn't re-cascade; on failure the slot is retained so
        the next ``_ORPHAN_CHECK_INTERVAL_S`` tick retries.

        codex r3 [R3-8, LOW DOC] — earlier wording said
        ``destroy(FORCE_TERMINATE)``; that was stale after R2-6 made the
        orphan tick thread ``DestroyReason.ORPHAN_TIMEOUT`` through the
        cascade payload so binding history distinguishes orphan-triggered
        destroys from parent-initiated FORCE_TERMINATE cascades.

        codex r6 [R6-2, HIGH CONTRACT] — the cascade payload field was
        removed; the override moved off the wire and into the supervisor-
        private ``_cascade_destroy_overrides`` side-table. The wire
        schema is restored to ``{reason, policy}``.

        codex r7 [R7-8, MEDIUM DOC] — earlier docstring still referenced
        the ``destroy_reason=ORPHAN_TIMEOUT`` payload field, which no
        longer exists post-R6. See also R7-4: the override is now popped
        only on side_effect success so PEL redelivery re-applies the
        same ``ORPHAN_TIMEOUT`` on retry.
        """
        now = self._ctx.clock()
        if now - self._last_orphan_check_mono < self._ORPHAN_CHECK_INTERVAL_S:
            return
        self._last_orphan_check_mono = now

        candidate_ids = list(dict.fromkeys(
            [*self._last_seen_mono.keys(), *self._known_children]
        ))
        for child_id in candidate_ids:
            liveness = self._ctx.liveness_service
            if liveness is not None and child_id in self._known_children:
                lease = await liveness.get_lease(child_id)
                if not liveness.is_stale(lease):
                    continue
                stale_seconds = SUBAGENT_PROGRESS_STALE_AFTER_SECONDS
            else:
                last_seen = self._last_seen_mono.get(child_id)
                if last_seen is None:
                    continue
                stale_seconds = now - last_seen
                if stale_seconds <= SUBAGENT_PROGRESS_STALE_AFTER_SECONDS:
                    continue
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.orphan_detected",
                    {
                        "child_session_id": child_id,
                        "stale_seconds": int(stale_seconds),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "orphan_detected telemetry raised root=%s child=%s — "
                    "continuing without the emit",
                    self._ctx.root_session_id,
                    child_id,
                )
            try:
                # codex r2 [R2-6, HIGH CONTRACT] — thread ORPHAN_TIMEOUT
                # through to the destroy call so ops can distinguish
                # orphan-triggered destroys from parent-initiated
                # FORCE_TERMINATE cascades in the binding history.
                await self._emit_cascade_terminate(
                    child_id,
                    reason="orphan_timeout",
                    destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "orphan cascade publish failed root=%s child=%s — "
                    "leaving last_seen in place for next tick to retry",
                    self._ctx.root_session_id,
                    child_id,
                )
                continue
            # Drop the heartbeat slot so the next tick doesn't re-cascade.
            #
            # codex r10 [R10-1, HIGH ARCH] — the slot is cleared after a
            # successful publish so the next tick doesn't double-fire.
            # If the published cascade envelope then exhausts XAUTOCLAIM
            # retries (handler chain failing repeatedly), the child has
            # NO remaining retry signal here — both tracking and the
            # envelope are gone. The cascade-CANCEL_REQUEST branch in
            # ``_on_poison_drop`` catches that case and fires the
            # direct-kill fallback (reusing the
            # ``_cascade_destroy_overrides`` entry for ORPHAN_TIMEOUT).
            self._last_seen_mono.pop(child_id, None)
            try:
                self._known_children.remove(child_id)
            except ValueError:
                pass

    async def _check_should_stop_for_rollback(self) -> None:
        """C3 PR-5 (spec §11.6 rollback runbook + R1 P2.2 NULL-coalesce).

        After the operator runs the rollback SQL

            UPDATE sessions
               SET subagent_control_plane = 'legacy'
             WHERE root_session_id = :root_id
               AND subagent_control_plane = 'mailbox';

        every live subagent child of this root becomes legacy-plane (their
        AgentService finalize path will then own destroy via the four gated
        suspend sites in ``app/application/services/agent_service.py``).
        Once that happens this supervisor has no further work to do — every
        SPAWN_REQUEST it would dispatch belongs to a session whose
        ``subagent_control_plane`` has been rolled back — so it stops
        itself.

        **R1 P2.2 NULL-coalesce.** Pre-C3 backfill leaves
        ``subagent_control_plane`` as NULL for legacy-managed children; the
        rollback SQL above skips them because ``NULL <> 'mailbox'``. A
        naive ``all(c.subagent_control_plane == 'legacy')`` check would
        then BLOCK the rollback because NULL ≠ 'legacy'. The fix is to
        coalesce: ``(c.subagent_control_plane or 'legacy') == 'legacy'``
        so NULL and 'legacy' are both treated as canonical legacy.

        Optional plumbing: tests and pre-PR-5 callers that build a
        ``SupervisorContext`` without ``session_repo`` get a safe no-op;
        the method also early-returns on any read failure rather than
        raising into the run loop. When ``stop_self_callback`` is wired
        (DI factory case) we schedule it via ``asyncio.create_task`` so
        the registry's ``stop`` pops the slot and cancels the run task
        AND ``health_check`` stops listing the root. When the callback
        is absent (direct construction in tests) we only set
        ``self._stopping`` so the run loop exits on the next iteration;
        the slot, if any, remains until external cleanup
        (``stop_all`` / ``stop`` / restart-loop reaping). The eager
        ``_stopping.set()`` runs in both cases so cancel propagation
        timing doesn't gate the exit.

        **Live-row filter (codex r3/r4 [HIGH CONTRACT]).** Terminal
        rows (``completed`` / ``timed_out``) keep their
        ``subagent_control_plane='mailbox'`` value because their
        ``destroy`` already ran. A naive "all subagents are legacy"
        predicate would then never converge after rollback because of
        the leftover terminal rows. The predicate therefore considers
        only NON-terminal rows — ``status NOT IN ('completed',
        'timed_out')``. This is BROADER than spec §11.6's rollback SQL
        (which only flips ``status IN ('running','finishing',
        'pending')``); it intentionally also covers WAITING /
        TAKEOVER* statuses that the spec SQL omits but the repo's
        live-set definition (``domain/repositories/session_repository.py:97``)
        includes. The spec doc itself needs a follow-up doc PR to
        align — until then the runtime predicate is the authoritative
        contract.

        Invocation: PR-5 wires this method into the main run loop via
        ``_maybe_tick_check_rollback`` (throttled to
        ``_ROLLBACK_CHECK_INTERVAL_S``). No HTTP admin route is exposed
        in PR-5; the integration tests in
        ``tests/integration/test_mailbox_migration.py`` exercise the
        method via direct supervisor construction. A future PR can
        wire an admin/cli entry point if operators want to force the
        check between ticks.

        **Self-cancel safety (codex r2 [HIGH CONTRACT]).** The registry-
        injected ``stop_self_callback`` is
        ``lambda: registry.stop(root_session_id)``, which does
        ``self._slots.pop(...)`` → ``slot.task.cancel()`` →
        ``await slot.task``. When invoked from within the supervisor's
        own run loop (i.e. from this tick), ``slot.task`` IS the current
        task — awaiting it would deadlock. The fix is to fire the
        callback as a separate task (``asyncio.create_task``) so the
        cancel propagates into the current task without awaiting it,
        AND eagerly set ``self._stopping`` so the run loop also exits
        on the next iteration if the cancel propagation is delayed.
        ``SupervisorRegistry._restart_crashed`` treats cancelled tasks
        as clean shutdown (not crashes) and does NOT resurrect the
        slot, so the supervisor stays dead post-rollback.
        """
        repo = self._ctx.session_repo
        if repo is None:
            return  # no DI plumbing — caller intentionally not wired

        try:
            root = await repo.get_by_id(self._ctx.root_session_id)
        except Exception:
            logger.warning(
                "rollback-stop check: get_by_id(root=%s) failed; skipping",
                self._ctx.root_session_id,
                exc_info=True,
            )
            return
        if root is None:
            return  # root deleted — supervisor will be reaped elsewhere

        try:
            children = await repo.find_descendants(
                self._ctx.root_session_id,
                user_id=root.user_id,
                max_depth=1,
                limit=1024,
            )
        except Exception:
            logger.warning(
                "rollback-stop check: find_descendants(root=%s) failed; skipping",
                self._ctx.root_session_id,
                exc_info=True,
            )
            return

        # codex r4 [HIGH CONTRACT] — "non-terminal" must mirror the
        # repository's authoritative definition, not the narrower spec
        # §11.6 SQL WHERE clause. The repo's
        # ``find_running_mailbox_subagent_sessions`` (see
        # ``domain/repositories/session_repository.py:97``) treats
        # ``status NOT IN ('completed', 'timed_out')`` as the live
        # set — which intentionally includes ``WAITING`` /
        # ``TAKEOVER_PENDING`` / ``TAKEOVER`` (paused but supervisor-
        # owned). The spec SQL leaves those rows on ``'mailbox'``; if
        # we filtered them OUT here, a running mailbox WAITING child
        # would be ignored and the supervisor would stop while it
        # still owns destroy for that child. NOT-IN-terminal also
        # forward-protects against future non-terminal status
        # additions. Reading status via ``_status_value`` so domain
        # enum + raw-string both work (callers always pass domain
        # Session objects, but defensive — repo unit tests sometimes
        # return shaped dicts).
        _ROLLBACK_TERMINAL_STATUSES = {"completed", "timed_out"}

        def _status_value(c) -> str:
            status = getattr(c, "status", None)
            if status is None:
                return ""
            value = getattr(status, "value", None)
            return value if isinstance(value, str) else str(status)

        all_subagent_children = [
            c for c in children if c.worker_type == "subagent"
        ]
        if not all_subagent_children:
            return  # truly fresh root — no subagents yet, don't stop

        live_subagent_children = [
            c
            for c in all_subagent_children
            if _status_value(c) not in _ROLLBACK_TERMINAL_STATUSES
        ]
        if not live_subagent_children:
            # codex r5 [HIGH CONTRACT] — DO NOT stop on "all subagents
            # terminal". Runner's terminal sequence at
            # ``agent_task_runner._terminal_op`` commits the DB
            # status FIRST (line ~3118), then publishes the
            # ``RESULT_READY``/``CANCEL_ACK`` envelope (line ~3149).
            # The race window between these two ops would let this
            # tick observe "all terminal" and stop the supervisor
            # BEFORE the terminal envelope is read by
            # ``ResultReadyHandler`` — losing the supervisor's
            # ``destroy()`` call. Reconcile_orphans / pod restart
            # eventually clean up, but the M1 single-writer contract
            # is broken for that envelope window.
            #
            # The cost of NOT stopping here is bounded: a few ticks of
            # supervisor + DB queries until external cleanup (pod
            # restart, ``registry.stop``, ``stop_all``). Rollback-stop
            # is the only safe self-stop signal — operator-driven SQL
            # is the synchronization point, not the runner's terminal
            # write. A future PR can add a stronger "drained PEL +
            # all-terminal" stop signal once the PEL idle check is
            # cheap enough.
            return

        # R1 P2.2 — NULL is canonical legacy; coalesce before comparison.
        # Explicit ``is None`` check rather than ``or 'legacy'`` so the
        # empty string ``''`` — should it ever appear despite the DB
        # CHECK constraint — would still fail the predicate rather than
        # masquerading as legacy. Matches the SQL ``COALESCE(value,
        # 'legacy') == 'legacy'`` semantics exactly.
        def _plane_or_legacy(c) -> str:
            value = getattr(c, "subagent_control_plane", None)
            return "legacy" if value is None else value

        all_legacy = all(
            _plane_or_legacy(c) == "legacy"
            for c in live_subagent_children
        )
        if not all_legacy:
            return

        logger.info(
            "rollback-stop: root=%s all %d LIVE (non-terminal) subagent "
            "children are legacy-plane (NULL coalesced; non-terminal "
            "matches repo definition status NOT IN ('completed','timed_out') "
            "— covers WAITING/TAKEOVER too) — stopping supervisor (spec §11.6)",
            self._ctx.root_session_id,
            len(live_subagent_children),
        )
        # Eagerly signal the run loop to exit on the next iteration so we
        # don't depend on cancel propagation timing.
        self._stopping.set()
        # Prefer registry-side stop so the slot is popped from
        # ``SupervisorRegistry._slots`` (health_check stops listing it).
        # Fire-and-forget — see method docstring "Self-cancel safety":
        # awaiting ``registry.stop(rid)`` from this task would deadlock
        # because ``registry.stop`` awaits ``slot.task`` which IS the
        # current task. ``asyncio.create_task`` schedules it concurrently;
        # the cancel propagates to the current run loop and ``run()``
        # exits via CancelledError. The fire-and-forget task itself
        # completes naturally once the run loop exits.
        if self._ctx.stop_self_callback is not None:
            try:
                asyncio.create_task(
                    self._ctx.stop_self_callback(),
                    name=f"mailbox-sup-rollback-stop:{self._ctx.root_session_id}",
                )
            except Exception:
                logger.exception(
                    "rollback-stop: scheduling registry stop_self_callback "
                    "failed root=%s — relying on _stopping.set() to exit run loop",
                    self._ctx.root_session_id,
                )
        # If no callback is wired (e.g. direct construction in tests
        # without the registry), ``self._stopping.set()`` above is the
        # only signal — the run loop checks it on the next iteration.

    async def _maybe_tick_check_rollback(self) -> None:
        """C3 PR-5 (spec §11.6) — throttled wrapper around
        ``_check_should_stop_for_rollback``. Runs at most every
        ``_ROLLBACK_CHECK_INTERVAL_S`` from the main loop so the DB
        query is bounded. Errors raised by the check itself are
        swallowed inside the helper; this wrapper only enforces the
        cadence so a transient DB blip doesn't poison the run loop.

        codex r2 [HIGH CONTRACT] regression fix — PR-5 R1 shipped
        ``_check_should_stop_for_rollback`` but forgot to wire it into
        the tick set; spec §11.6 expects the supervisor to detect the
        rollback SQL on the next tick. The R2 wiring closes that gap.
        """
        now = self._ctx.clock()
        if now - self._last_rollback_check_mono < self._ROLLBACK_CHECK_INTERVAL_S:
            return
        self._last_rollback_check_mono = now
        await self._check_should_stop_for_rollback()

    async def _emit_cascade_terminate(
        self,
        child_id: str,
        *,
        reason: str,
        destroy_reason: Optional[DestroyReason] = None,
    ) -> None:
        """Publish a synthetic CANCEL_REQUEST(TERMINATE) so the main loop
        re-dispatches through ``CancelRequestHandler`` on the next read.

        ``producer_role=SUPERVISOR`` (not SUPERVISOR_ECHO) — this is a fresh
        cascade message originating from the supervisor, NOT an echo of a
        child→parent envelope. Empty ``child_id`` is rejected with a no-op
        (defensive guard for cross-root forgery paths the supervisor already
        filters elsewhere).

        codex r2 [R2-5, HIGH CONTRACT] — envelope_id MUST fit the audit
        table's ``String(64)`` column. The previous format
        ``cascade:{reason}:{child_id}:{ts_ms}`` could reach 70+ characters
        with UUID child_ids (36 char) + ``cancel_ack_timeout`` reason.
        We now compose ``cascade:{sha256(reason:child_id:ts_ms)[:32]}``
        — always 39 chars — stable per (reason, child, time) tuple. Same
        (reason, child_id, time) → same envelope_id (idempotent across
        retries inside one tick); different times → different ids.

        codex r3 [R3-5, HIGH CONTRACT] — ``correlation_id`` is ALSO bounded
        by ``String(64)`` (``infrastructure/models/mailbox_envelope_audit
        .py:51``). The previous comment incorrectly claimed correlation_id
        was unbounded; the audit-repo upsert would silently fail (or row
        write would truncate / raise) when ``reason`` was a long string
        like ``poison_drop_terminal_envelope`` (28 chars) combined with a
        UUID child_id (36 chars) — total 73 chars including the
        ``cascade:`` prefix. Apply the same hash-or-truncate strategy as
        envelope_id: ``cascade:{sha256(reason:child_id)[:32]}`` = 40 chars,
        stable per (reason, child) tuple. We deliberately omit ``ts_ms``
        from the correlation hash so that all retries within one cascade
        tick share a correlation_id (cross-envelope grep stays useful).

        codex r2 [R2-6, HIGH CONTRACT] (superseded by R6-2) — ``destroy_reason``
        was originally threaded into the public payload so the TERMINATE
        handler could pass it to ``SandboxLifecycleService.destroy``
        instead of the hardcoded ``DestroyReason.FORCE_TERMINATE``. codex
        r6 [R6-2] moved this off the wire and into the supervisor-private
        ``_cascade_destroy_overrides`` side-table so the
        ``CancelRequestPayload`` schema remains frozen at C3 ship at
        ``{reason, policy}``. We populate the side-table BEFORE
        ``publisher.publish`` so the in-process handler can read+pop
        the override on dispatch; external producers cannot reach the
        dict, which moots the R3-6 producer_role guard.
        """
        if not child_id:
            return
        now_mono = self._ctx.clock()
        ts_ms = int(now_mono * 1000)
        # 32-hex-char digest fits within the 64-char audit column with
        # the ``cascade:`` prefix (39 chars total). sha256 is collision-
        # resistant; truncation to 128 bits is safe at supervisor scale.
        env_key = f"{reason}:{child_id}:{ts_ms}"
        env_digest = hashlib.sha256(env_key.encode("utf-8")).hexdigest()[:32]
        # R3-5 — correlation_id also bounded to 64. Hash (reason, child_id)
        # without ts_ms so retries inside one cascade tick share the id
        # for cross-envelope grep. 40 chars total with ``cascade:`` prefix.
        corr_key = f"{reason}:{child_id}"
        corr_digest = hashlib.sha256(corr_key.encode("utf-8")).hexdigest()[:32]
        synthetic_envelope_id = f"cascade:{env_digest}"
        cascade_env = MailboxEnvelope(
            envelope_id=synthetic_envelope_id,
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            parent_session_id=self._ctx.root_session_id,
            child_session_id=child_id,
            correlation_id=f"cascade:{corr_digest}",
            emitted_at=self._ctx.now(),
            producer_role=ProducerRole.SUPERVISOR,
            payload=CancelRequestPayload(
                reason=reason,
                policy=CancelPolicy.TERMINATE,
            ).model_dump(mode="json"),
        )
        # codex r6 [R6-2] — stamp the override BEFORE publish. The handler
        # reads+pops the entry on dispatch; if publish fails (R6-3 fallback)
        # the entry is explicitly removed below since the handler will
        # never see the envelope.
        if destroy_reason is not None:
            self._cascade_destroy_overrides[synthetic_envelope_id] = (
                destroy_reason
            )
        try:
            await self._ctx.publisher.publish(cascade_env)
        except asyncio.CancelledError:
            # Caller (orphan/poison/cancel-ack-timeout tick) handles
            # cancellation; clean up the side-table so a future
            # synthetic envelope sharing the id doesn't accidentally
            # inherit an override from this aborted cascade.
            self._cascade_destroy_overrides.pop(synthetic_envelope_id, None)
            raise
        except Exception:
            # codex r6 [R6-3] — direct stop+destroy fallback so an XADD
            # failure still kills the orphaned child (spec §7.5 / §7.6
            # hard rule). Cleanup the side-table since the handler will
            # never see this envelope.
            self._cascade_destroy_overrides.pop(synthetic_envelope_id, None)
            # codex r7 [R7-7, HIGH CONTRACT] — propagate
            # ``_CascadeFailedError`` from the fallback so callers can
            # preserve per-child tracking + retry on the next tick.
            # Non-terminal cases (already destroyed / binding
            # missing) in the fallback path do NOT raise — they are
            # success-equivalent. Only retryable
            # ``SandboxLifecycleError`` in the fallback bubbles up.
            #
            # codex r10 [R10-2, HIGH CONTRACT] — pass the synthetic
            # envelope through so the fallback can fire
            # ``agent_service_callback`` BEFORE ``destroy`` (spec §7.6
            # stop-before-destroy order, even in the XADD-failure
            # emergency path).
            await self._cascade_publish_failed_direct_kill(
                child_id,
                reason=reason,
                destroy_reason=destroy_reason or DestroyReason.FORCE_TERMINATE,
                synthetic_envelope_id=synthetic_envelope_id,
                synthetic_envelope=cascade_env,
            )
            # Swallow non-cascade-failed paths — the cascade is
            # considered done (we tried XADD, we fell back to direct
            # destroy and it succeeded or hit a terminal state).
            # Re-raising would block the caller's continue-to-next-child
            # loop. _CascadeFailedError from the fallback above is the
            # exception: it MUST propagate so the caller skips tracking
            # cleanup.

    async def _cascade_publish_failed_direct_kill(
        self,
        child_id: str,
        *,
        reason: str,
        destroy_reason: DestroyReason,
        synthetic_envelope_id: str,
        synthetic_envelope: Optional[MailboxEnvelope] = None,
    ) -> None:
        """codex r6 [R6-3, HIGH CONTRACT] — spec §7.5 + §7.6: "XADD record
        is best-effort; XADD failure still kills." When
        ``_emit_cascade_terminate``'s publish raises, the synthetic
        CANCEL_REQUEST never lands in the stream so
        ``CancelRequestHandler`` will never destroy this child. Fall
        back to a direct ``sandbox_lifecycle.destroy`` so the orphaned
        child still gets cleaned up.

        codex r10 [R10-2, HIGH CONTRACT] — spec §7.6 mandates
        ``stop → destroy`` order. Earlier rounds skipped the callback in
        this emergency path on the rationale that we lacked a useful
        envelope; the cascade synthetic envelope is now threaded
        through so the callback (which signals the child to stop
        cooperatively in production via ``AgentService.stop_session``)
        fires before destroy. The callback is best-effort — any
        exception is logged and swallowed so a callback fault never
        blocks the load-bearing destroy step. ``CancelledError`` still
        propagates so caller-driven cancellation aborts cleanly.
        """
        logger.exception(
            "cascade publish failed root=%s child=%s reason=%s — "
            "falling back to direct destroy(%s)",
            self._ctx.root_session_id,
            child_id,
            reason,
            destroy_reason.value,
        )
        # codex r10 [R10-2, HIGH CONTRACT] — fire callback BEFORE destroy
        # so the stop-before-destroy ordering is preserved even in the
        # XADD-failure emergency path. Best-effort: log + swallow so a
        # callback fault doesn't block the load-bearing destroy.
        if synthetic_envelope is not None:
            try:
                await self._ctx.agent_service_callback(synthetic_envelope)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade direct-kill callback failed root=%s child=%s — "
                    "continuing to destroy",
                    self._ctx.root_session_id,
                    child_id,
                )
        try:
            await self._ctx.sandbox_lifecycle.destroy(child_id, destroy_reason)
        except SandboxAlreadyDestroyed:
            # codex r8 [R8-6, MEDIUM CONTRACT] — terminal-success branch.
            # The child is already gone from the lifecycle service's POV,
            # so per-child tracking (_last_seen_mono / _cancel_states) is
            # now stale and must be cleared. Without this, the next
            # orphan tick (_ORPHAN_CHECK_INTERVAL_S) re-fires a cascade
            # against the same already-destroyed child — same bug shape
            # as F7+F9 mirrored from the in-handler clear_child_tracking
            # invariant. Mirrored across all three terminal-success
            # branches (AlreadyDestroyed / BindingMissing / success).
            self._clear_child_tracking(child_id)
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.cascade_xadd_failed_direct_kill_already_destroyed",
                    {
                        "child_session_id": child_id,
                        "reason": reason,
                        "destroy_reason": destroy_reason.value,
                        "synthetic_envelope_id": synthetic_envelope_id,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_xadd_failed_direct_kill_already_destroyed "
                    "telemetry raised root=%s child=%s",
                    self._ctx.root_session_id,
                    child_id,
                )
        except SandboxBindingMissing:
            # R8-6 — terminal-success branch (binding gone). Same
            # reasoning as the AlreadyDestroyed branch above: clear stale
            # per-child tracking so the orphan tick doesn't re-cascade.
            self._clear_child_tracking(child_id)
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.cascade_xadd_failed_direct_kill_binding_missing",
                    {
                        "child_session_id": child_id,
                        "reason": reason,
                        "destroy_reason": destroy_reason.value,
                        "synthetic_envelope_id": synthetic_envelope_id,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_xadd_failed_direct_kill_binding_missing "
                    "telemetry raised root=%s child=%s",
                    self._ctx.root_session_id,
                    child_id,
                )
        except SandboxLifecycleError as e:
            # codex r7 [R7-7, HIGH CONTRACT] — earlier rounds logged +
            # returned, after which the orphan tick / cancel auto-escalate
            # cleared per-child tracking → cascade was lost (no future
            # tick will retry; no PEL entry exists since XADD failed; no
            # in-memory state remains). Signal the caller via
            # ``_CascadeFailedError`` so it preserves tracking and the
            # next ``_ORPHAN_CHECK_INTERVAL_S`` / cancel tick retries the
            # cascade end-to-end (Option A + C combined from the R7-7
            # finding).
            logger.exception(
                "cascade publish failed AND direct destroy raised "
                "SandboxLifecycleError root=%s child=%s reason=%s err=%s — "
                "raising _CascadeFailedError so caller preserves tracking "
                "for next-tick retry",
                self._ctx.root_session_id,
                child_id,
                reason,
                e,
            )
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.cascade_xadd_failed_direct_kill_retryable_failed",
                    {
                        "child_session_id": child_id,
                        "reason": reason,
                        "destroy_reason": destroy_reason.value,
                        "synthetic_envelope_id": synthetic_envelope_id,
                        "error": str(e),
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_xadd_failed_direct_kill_retryable_failed "
                    "telemetry raised root=%s child=%s",
                    self._ctx.root_session_id,
                    child_id,
                )
            raise _CascadeFailedError(
                f"cascade publish + direct destroy both failed "
                f"child={child_id} reason={reason} err={e}"
            ) from e
        else:
            # codex r8 [R8-6, MEDIUM CONTRACT] — destroy succeeded via the
            # direct-kill fallback (XADD failed but the local destroy
            # call landed). The child is gone; per-child tracking is now
            # stale and the orphan tick would otherwise see the lingering
            # ``_last_seen_mono[child]`` and re-cascade. Mirror the
            # terminal-handler invariant (F7+F9 / R4-4) here so the
            # direct-kill path leaves tracking in the same clean state
            # the normal CancelRequestHandler.TERMINATE path produces.
            self._clear_child_tracking(child_id)
            try:
                await self._ctx.telemetry.emit(
                    "mailbox.cascade_xadd_failed_direct_kill",
                    {
                        "child_session_id": child_id,
                        "reason": reason,
                        "destroy_reason": destroy_reason.value,
                        "synthetic_envelope_id": synthetic_envelope_id,
                    },
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "cascade_xadd_failed_direct_kill telemetry raised "
                    "root=%s child=%s",
                    self._ctx.root_session_id,
                    child_id,
                )
