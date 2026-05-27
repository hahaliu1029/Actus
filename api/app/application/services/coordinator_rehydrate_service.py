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
  and crash_mid_apply (apply audit row stuck in_progress > 5min).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlreadyAppliedInfo:
    """[spec §12.3 Step 8] Apply audit row already exists for this run.

    ``status`` is one of: 'success' (apply completed) | 'rollback_partial'
    (apply failed AND rollback could not restore everything — manual
    recovery required) | 'crash_mid_apply' (in_progress > 5min — pod crashed
    mid-apply) | 'in_progress_recent' (in_progress < 5min — another pod
    holds the Redis lock).
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


# Threshold for "apply row stuck in_progress" → pod crashed mid-apply.
# 5min covers worst-case healthy apply duration (multi-file with rollback
# verify), so longer than this is safely interpreted as a crashed pod.
_CRASH_MID_APPLY_SECONDS = 300


class CoordinatorRehydrateService:
    """Read-only crash-recovery scanner. Stateless across invocations.

    Constructor takes the 3 domain repos + 2 optional infrastructure hooks:
    ``publisher`` (mailbox publisher for unexpected-child CANCEL_REQUEST —
    NOT used in detect_existing_run; held for future API growth) and
    ``emit_event`` (async callable accepting a HealthEvent for the
    rollback_partial / crash_mid_apply alerts).
    """

    def __init__(
        self,
        *,
        session_repository,
        envelope_store,
        audit_repository,
        publisher=None,
        emit_event: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        self._sr = session_repository
        self._es = envelope_store
        self._ar = audit_repository
        self._publisher = publisher
        self._emit_event = emit_event

    async def detect_existing_run(
        self,
        *,
        coordinator_run_id: str,
        parent_session_id: str,
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
        already_applied = await self._check_already_applied(coordinator_run_id)

        return RehydrateResult(
            child_session_ids=wu_to_session,
            pending=pending,
            terminal=terminal_records,
            already_applied=already_applied,
        )

    async def _check_already_applied(
        self, coordinator_run_id: str,
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
            )
            return AlreadyAppliedInfo(
                status="rollback_partial", audit_id=audit.id,
            )
        if audit.status == "in_progress":
            now = datetime.now(timezone.utc)
            started_at = audit.started_at
            # Some ORMs may yield naive datetime; normalize defensively.
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
            age = (now - started_at).total_seconds()
            if age > _CRASH_MID_APPLY_SECONDS:
                await self._emit_health(
                    code="coordinator_apply_crash_mid_apply",
                    reason=(
                        f"audit {audit.id} in_progress for {age:.0f}s on "
                        f"run {coordinator_run_id}; pod crashed mid-apply; "
                        f"manual recovery required"
                    ),
                    coordinator_run_id=coordinator_run_id,
                    audit_id=audit.id,
                )
                return AlreadyAppliedInfo(
                    status="crash_mid_apply", audit_id=audit.id,
                )
            return AlreadyAppliedInfo(
                status="in_progress_recent", audit_id=audit.id,
            )
        # Other terminal failure statuses (digest_drift / write_io_error /
        # apply_aborted etc.) → not "already applied" — caller may retry.
        return None

    async def _emit_health(
        self,
        *,
        code: str,
        reason: str,
        coordinator_run_id: str,
        audit_id: int,
    ) -> None:
        if self._emit_event is None:
            return
        # Late import — keeps the application module's load path lean and
        # mirrors patch_applier.py:687-704 emit pattern (live HealthEvent
        # signature uses status/reason/action/metrics — NOT level/message).
        from app.domain.models.event import HealthEvent, HealthStatus
        try:
            await self._emit_event(HealthEvent(
                status=HealthStatus.TERMINATING,
                reason=reason,
                action="hard_terminate",
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
