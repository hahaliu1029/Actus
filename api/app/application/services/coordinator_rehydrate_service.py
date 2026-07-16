"""C2 v1 CoordinatorRehydrateService (spec §12.3) — crash-recovery state reconstruction.

Called from ``parallel_execution_subgraph.dispatch_node`` BEFORE first-time
dispatch decides between (peek=None → first-time) and (peek=N → resume).
The peeked coordinator_run_id is passed in; this service queries the DB
to reconstruct the (wu_id → child_session_id) map and the terminal-envelope
replay buffer.

Domain contracts consumed:
- ``SessionRepository.find_children_by_coordinator_run`` (PR-7 Task 7.2)
- ``CoordinatorResultEnvelopeStoreRepository.find_terminal_envelopes_by_run``
- ``CoordinatorApplyAuditRepository.find_latest_for_run``

Emits:
- ``HealthEvent`` (via injected ``emit_event`` callable) on rollback_partial
  and crash_mid_apply (the apply owner lease is continuously absent for the
  reconciliation grace window).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlreadyAppliedInfo:
    """[spec §12.3 Step 8] Apply audit row already exists for this run.

    ``status`` is one of: 'success' (apply completed) | 'rollback_partial'
    (apply failed AND rollback could not restore everything — manual
    recovery required) | 'crash_mid_apply' (owner lease stayed absent for the
    reconciliation grace — pod crashed mid-apply) | 'in_progress_recent'
    (owner lease is live, could not be checked safely, or is still within the
    missing-owner grace window).
    """
    status: str
    audit_id: int


@dataclass(frozen=True)
class TerminalEnvelopeRecord:
    """[r3 P1-3] Full terminal record discriminated by envelope_type.

    RESULT_READY vs CANCEL_ACK have different downstream semantics — the
    rehydrator must preserve both kinds so the subgraph can populate
    worker_results with the actual outcome (vs assuming all terminals are
    RESULT_READY).
    """
    envelope_type: str  # 'RESULT_READY' | 'CANCEL_ACK'
    payload: dict[str, Any]
    child_session_id: str
    received_at: datetime


@dataclass(frozen=True)
class RehydrateResult:
    """Aggregate of all 7 step outputs handed back to dispatch_node."""
    child_session_ids: dict[str, str]                       # wu_id → session_id
    pending: list[str]                                      # wu_ids without terminal envelope
    terminal: dict[str, TerminalEnvelopeRecord]             # wu_id → record
    already_applied: Optional[AlreadyAppliedInfo] = None


@dataclass(frozen=True)
class ApplyLeaseObservation:
    """Atomic Redis-time observation of apply ownership and missing age."""

    owner_is_live: bool
    missing_for_seconds: float


_APPLY_RECONCILE_GRACE_SECONDS = 30
_APPLY_RECONCILE_MARKER_TTL_SECONDS = 86_400


class ApplyLeaseObserver(Protocol):
    """Atomically reconcile the canonical apply lock and missing marker."""

    async def __call__(
        self,
        coordinator_run_id: str,
        *,
        marker_ttl_seconds: int,
    ) -> ApplyLeaseObservation: ...


class ApplyCrashFence(Protocol):
    """Run one audit CAS while exclusively owning the canonical apply key.

    ``None`` means the lock was busy or ownership became ambiguous. ``False``
    is a completed DB CAS miss; ``True`` is a completed DB transition.
    """

    async def __call__(
        self,
        coordinator_run_id: str,
        operation: Callable[[], Awaitable[bool]],
        *,
        minimum_missing_seconds: int,
    ) -> bool | None: ...


class CoordinatorRehydrateService:
    """Read-only crash-recovery scanner. Stateless across invocations.

    Constructor takes the 3 domain repos plus infrastructure hooks:
    ``publisher`` (mailbox publisher for unexpected-child CANCEL_REQUEST —
    NOT used in detect_existing_run; held for future API growth) and
    ``emit_event`` (async callable accepting a HealthEvent for the
    rollback_partial / crash_mid_apply alerts). Production also injects the
    atomic apply-lease observer used to distinguish a healthy long-running
    apply from a crashed owner without relying on pod wall clocks.
    """

    def __init__(
        self,
        *,
        session_repository,
        envelope_store,
        audit_repository,
        publisher=None,
        emit_event: Optional[Callable[[Any], Awaitable[None]]] = None,
        apply_lease_observer: Optional[ApplyLeaseObserver] = None,
        apply_crash_fence: Optional[ApplyCrashFence] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        apply_reconcile_grace_seconds: int = _APPLY_RECONCILE_GRACE_SECONDS,
        apply_reconcile_marker_ttl_seconds: int = (
            _APPLY_RECONCILE_MARKER_TTL_SECONDS
        ),
    ) -> None:
        if (
            isinstance(apply_reconcile_grace_seconds, bool)
            or not isinstance(apply_reconcile_grace_seconds, int)
            or apply_reconcile_grace_seconds <= 0
        ):
            raise ValueError(
                "apply_reconcile_grace_seconds must be a positive integer",
            )
        if (
            isinstance(apply_reconcile_marker_ttl_seconds, bool)
            or not isinstance(apply_reconcile_marker_ttl_seconds, int)
            or apply_reconcile_marker_ttl_seconds
            <= apply_reconcile_grace_seconds
        ):
            raise ValueError(
                "apply_reconcile_marker_ttl_seconds must be an integer "
                "greater than apply_reconcile_grace_seconds",
            )
        self._sr = session_repository
        self._es = envelope_store
        self._ar = audit_repository
        self._publisher = publisher
        self._emit_event = emit_event
        self._apply_lease_observer = apply_lease_observer
        self._apply_crash_fence = apply_crash_fence
        self._clock = clock
        self._apply_reconcile_grace_seconds = apply_reconcile_grace_seconds
        self._apply_reconcile_marker_ttl_seconds = (
            apply_reconcile_marker_ttl_seconds
        )

    async def detect_existing_run(
        self,
        *,
        coordinator_run_id: str,
        parent_session_id: str,
        emit_event: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> Optional[RehydrateResult]:
        """Run the 7-step rehydrate algorithm.

        Returns ``None`` when no prior dispatch happened at this
        ``coordinator_run_id`` (first-time dispatch path). Otherwise
        returns a fully-populated ``RehydrateResult`` the dispatcher can
        use to re-fan-out only the still-pending work units.
        """
        # Steps 1-2: query children by (run_id, parent)
        children = await self._sr.find_children_by_coordinator_run(
            coordinator_run_id=coordinator_run_id,
            parent_session_id=parent_session_id,
        )
        if not children:
            return None

        # Step 3: derive wu_id → (session_id, status) maps from children
        wu_to_session: dict[str, str] = {}
        wu_status: dict[str, str] = {}
        for c in children:
            wu_id = c.work_unit_id
            if wu_id is None:
                # Defensive — coordinator_run_id was set so work_unit_id
                # should be too (PR-3 invariant). If somehow null, skip;
                # dispatcher will see it as missing and re-spawn.
                logger.warning(
                    "rehydrate: child %s has coordinator_run_id=%s but "
                    "work_unit_id=None; skipping in wu map",
                    c.id, coordinator_run_id,
                )
                continue
            wu_to_session[wu_id] = c.id
            wu_status[wu_id] = (
                c.status.value if hasattr(c.status, "value") else str(c.status)
            )

        # Step 4: load terminal envelopes from store (preserves
        # envelope_type discrimination per [r3 P1-3]).
        #
        # [codex R5 P2 -- deferred to PR-7+] Read-side defense-in-depth
        # gap: the write-side (``DbCoordinatorResultEnvelopeStoreRepository
        # .persist_terminal``) enforces ``_MAX_PAYLOAD_BYTES`` (64KB),
        # ``_pii_guard`` (email/phone regex), and
        # ``_filter_minimum_rehydrate`` (whitelist) BEFORE INSERT. A
        # DB-injection or backup-restore-poisoned row would bypass those
        # guards on the READ path (this method). ``PatchManifest`` shape
        # is re-validated downstream at
        # ``parallel_execution_subgraph._build_pre_results_from_terminal``
        # via ``PatchManifest.model_validate``, so manifest-shape exploit
        # is blocked -- but free-text bypass / oversize replay is still
        # latent. PR-7+ should add a shared read-side normalizer mirroring
        # the write-side guards.
        envelopes = await self._es.find_terminal_envelopes_by_run(
            coordinator_run_id,
        )
        terminal_records: dict[str, TerminalEnvelopeRecord] = {}
        for e in envelopes:
            terminal_records[e.work_unit_id] = TerminalEnvelopeRecord(
                envelope_type=e.envelope_type,
                payload=e.payload,
                child_session_id=e.child_session_id,
                received_at=e.received_at,
            )

        # Step 5: pending = children still running (no terminal envelope
        # observed yet). Sorted for deterministic Send fan-out.
        #
        # [codex R2 P1] SessionStatus.value is LOWERCASE
        # ("pending"/"running" -- see ``api/app/domain/models/session.py``
        # SessionStatus enum). The earlier uppercase tuple matched
        # nothing, which silently filtered every truly-pending child out
        # of the Send fan-out -- the reducer would then see an
        # incomplete worker_results set on every rehydrate.
        pending = sorted(
            wu_id
            for wu_id, status in wu_status.items()
            if wu_id not in terminal_records
            and status in ("pending", "running")
        )

        # Steps 6-7 (missing/unexpected child handling) → dispatcher; here
        # we just return data.

        # Step 8: check apply audit row, emit HealthEvent for the two
        # crash-recovery alerts.
        already_applied = await self._check_already_applied(
            coordinator_run_id, emit_event=emit_event,
        )

        return RehydrateResult(
            child_session_ids=wu_to_session,
            pending=pending,
            terminal=terminal_records,
            already_applied=already_applied,
        )

    async def _check_already_applied(
        self, coordinator_run_id: str, *, emit_event=None,
    ) -> Optional[AlreadyAppliedInfo]:
        audit = await self._ar.find_latest_for_run(coordinator_run_id)
        if audit is None:
            return None
        if audit.status == "success":
            return AlreadyAppliedInfo(status="success", audit_id=audit.id)
        if audit.status == "rollback_partial":
            await self._emit_health(
                code="coordinator_apply_rollback_partial",
                reason=(
                    f"audit {audit.id} rollback_partial on run "
                    f"{coordinator_run_id}; manual recovery required"
                ),
                coordinator_run_id=coordinator_run_id,
                audit_id=audit.id,
                emit_event=emit_event,
            )
            return AlreadyAppliedInfo(
                status="rollback_partial", audit_id=audit.id,
            )
        if audit.status == "crash_mid_apply":
            # The first confirmed grace violation is persisted below. A
            # terminal audit is the monotonic authority after the Redis marker
            # retention TTL expires, so later scans cannot regress to recent.
            return AlreadyAppliedInfo(
                status="crash_mid_apply", audit_id=audit.id,
            )
        if audit.status == "in_progress":
            now = self._clock()
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            started_at = audit.started_at
            # Some ORMs may yield naive datetime; normalize defensively.
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            age = (now - started_at).total_seconds()

            if self._apply_lease_observer is not None:
                try:
                    observation = await self._apply_lease_observer(
                        coordinator_run_id,
                        marker_ttl_seconds=(
                            self._apply_reconcile_marker_ttl_seconds
                        ),
                    )
                except Exception:
                    # One atomic Redis operation owns both the lease probe and
                    # marker transition. An ambiguous/failed observation can
                    # never prove a crash.
                    logger.exception(
                        "rehydrate: atomic apply lease observation failed run=%s",
                        coordinator_run_id,
                    )
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=audit.id,
                    )

                if (
                    not isinstance(observation, ApplyLeaseObservation)
                    or not isinstance(observation.owner_is_live, bool)
                ):
                    logger.error(
                        "rehydrate: invalid atomic apply lease observation "
                        "run=%s value=%r",
                        coordinator_run_id,
                        observation,
                    )
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=audit.id,
                    )
                if observation.owner_is_live:
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=audit.id,
                    )
                missing_age = observation.missing_for_seconds
                if (
                    isinstance(missing_age, bool)
                    or not isinstance(missing_age, (int, float))
                    or not math.isfinite(float(missing_age))
                    or missing_age < 0
                ):
                    logger.error(
                        "rehydrate: invalid atomic apply lease observation "
                        "run=%s value=%r",
                        coordinator_run_id,
                        observation,
                    )
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=audit.id,
                    )
                return await self._classify_missing_apply_owner(
                    coordinator_run_id=coordinator_run_id,
                    audit=audit,
                    audit_age=age,
                    missing_age=float(missing_age),
                    emit_event=emit_event,
                )

            # Missing composition is never crash evidence. Production tests pin
            # the observer to the shared canonical apply-lock key.
            logger.error(
                "rehydrate: atomic apply lease observer missing run=%s",
                coordinator_run_id,
            )
            return AlreadyAppliedInfo(
                status="in_progress_recent", audit_id=audit.id,
            )
        # Other terminal failure statuses (digest_drift / write_io_error /
        # apply_aborted etc.) → not "already applied" — caller may retry.
        return None

    async def _classify_missing_apply_owner(
        self,
        *,
        coordinator_run_id: str,
        audit: Any,
        audit_age: float,
        missing_age: float,
        emit_event=None,
    ) -> Optional[AlreadyAppliedInfo]:
        if missing_age >= self._apply_reconcile_grace_seconds:
            if self._apply_crash_fence is None:
                logger.error(
                    "rehydrate: apply crash fence missing run=%s",
                    coordinator_run_id,
                )
                return AlreadyAppliedInfo(
                    status="in_progress_recent", audit_id=audit.id,
                )

            async def persist_crash_if_still_in_progress() -> bool:
                latest = await self._ar.find_latest_for_run(
                    coordinator_run_id,
                )
                if (
                    latest is None
                    or latest.id != audit.id
                    or latest.status != "in_progress"
                ):
                    return False
                return await self._ar.update_terminal(
                    audit.id,
                    status="crash_mid_apply",
                    failed_reason=(
                        "apply owner continuously missing beyond "
                        "reconciliation grace"
                    ),
                )

            try:
                persisted = await self._apply_crash_fence(
                    coordinator_run_id,
                    persist_crash_if_still_in_progress,
                    minimum_missing_seconds=(
                        self._apply_reconcile_grace_seconds
                    ),
                )
            except Exception:
                logger.exception(
                    "rehydrate: failed to fence/persist crash_mid_apply run=%s "
                    "audit=%s",
                    coordinator_run_id,
                    audit.id,
                )
                return AlreadyAppliedInfo(
                    status="in_progress_recent", audit_id=audit.id,
                )
            if persisted is None:
                # A rollback/replacement owner won the canonical lock, or the
                # fence lost ownership. Neither state proves a crash.
                return AlreadyAppliedInfo(
                    status="in_progress_recent", audit_id=audit.id,
                )
            if not isinstance(persisted, bool):
                logger.error(
                    "rehydrate: invalid apply crash fence result run=%s "
                    "value=%r",
                    coordinator_run_id,
                    persisted,
                )
                return AlreadyAppliedInfo(
                    status="in_progress_recent", audit_id=audit.id,
                )
            if not persisted:
                # A concurrent terminal writer won first. Re-read the latest
                # audit authority; never overwrite or regress that outcome.
                latest = await self._ar.find_latest_for_run(
                    coordinator_run_id,
                )
                if latest is None:
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=audit.id,
                    )
                if latest.status == "success":
                    return AlreadyAppliedInfo(
                        status="success", audit_id=latest.id,
                    )
                if latest.status == "crash_mid_apply":
                    return AlreadyAppliedInfo(
                        status="crash_mid_apply", audit_id=latest.id,
                    )
                if latest.status == "rollback_partial":
                    await self._emit_health(
                        code="coordinator_apply_rollback_partial",
                        reason=(
                            f"audit {latest.id} rollback_partial on run "
                            f"{coordinator_run_id}; manual recovery required"
                        ),
                        coordinator_run_id=coordinator_run_id,
                        audit_id=latest.id,
                        emit_event=emit_event,
                    )
                    return AlreadyAppliedInfo(
                        status="rollback_partial", audit_id=latest.id,
                    )
                if latest.status == "in_progress":
                    return AlreadyAppliedInfo(
                        status="in_progress_recent", audit_id=latest.id,
                    )
                return None
            await self._emit_health(
                code="coordinator_apply_crash_mid_apply",
                reason=(
                    f"audit {audit.id} in_progress for {audit_age:.0f}s and "
                    f"apply owner missing for {missing_age:.0f}s on run "
                    f"{coordinator_run_id}; pod crashed mid-apply; "
                    f"manual recovery required"
                ),
                coordinator_run_id=coordinator_run_id,
                audit_id=audit.id,
                emit_event=emit_event,
            )
            return AlreadyAppliedInfo(
                status="crash_mid_apply", audit_id=audit.id,
            )
        return AlreadyAppliedInfo(
            status="in_progress_recent", audit_id=audit.id,
        )

    async def _emit_health(
        self,
        *,
        code: str,
        reason: str,
        coordinator_run_id: str,
        audit_id: int,
        emit_event=None,
    ) -> None:
        # Prefer the call-time emitter (per-run event queue closure) over the
        # construction-time singleton; production builds the service with
        # ``emit_event=None`` so the call-time emitter is the live path. Never
        # mutate ``self._emit_event`` — the singleton stays untouched.
        emitter = emit_event or self._emit_event
        if emitter is None:
            return
        # Late import — keeps the application module's load path lean and
        # mirrors patch_applier.py:687-704 emit pattern (live HealthEvent
        # signature uses status/reason/action/metrics — NOT level/message).
        from app.domain.models.event import HealthEvent, HealthStatus
        try:
            # [finish-core R1-P1] Rehydrate recovery alerts are INFORMATIONAL:
            # the current (retry) session is NOT terminating — main_graph's
            # ALREADY_APPLIED short-circuit only returns an operator-facing
            # summary string and the step completes normally. Emitting
            # ``TERMINATING`` would make the frontend
            # (``session-store.ts`` resolveStatusFromEvent) sticky-map the live
            # session to ``timed_out``. ``DEGRADED`` is frontend-informational
            # (keeps current status); the specific condition is carried in
            # ``metrics.code`` for operators / dashboards.
            await emitter(HealthEvent(
                status=HealthStatus.DEGRADED,
                reason=reason,
                action="manual_recovery_required",
                metrics={
                    "code": code,
                    "coordinator_run_id": coordinator_run_id,
                    "audit_id": audit_id,
                },
            ))
        except Exception:
            # Best-effort — rehydrate must still return its result even if
            # the event sink is down.
            logger.exception(
                "rehydrate: emit_event failed for code=%s run=%s",
                code, coordinator_run_id,
            )
