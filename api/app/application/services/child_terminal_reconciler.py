"""C2b child-row reaper — durable backstop for zombie-RUNNING coordinator children.

Spec: docs/superpowers/specs/2026-06-09-c2b-child-row-reaper-design.md

A coordinator-spawned child session row can leak as a permanent ``RUNNING`` zombie
when (a) the runner's best-effort terminal row write fails, or (b) the runner
process is SIGKILLed after its terminal envelope was persisted but before/without
the row write. This module is a *match-only* startup sweep: for each still-RUNNING
foreground mailbox subagent child that ALREADY has a persisted terminal envelope in
the coordinator result-envelope store, it terminalizes the lagging row to MATCH
that envelope (via the A4-1 ``SessionStateMachine.terminate`` CAS). It NEVER
synthesises an envelope and NEVER touches a child that has no persisted envelope
(leaving such a child RUNNING preserves ``reconcile_orphans``' supervisor re-spawn
trigger — spec §4.2 / INV-R2).

Invariants (spec §7): INV-R1 (writes only via ``ssm.terminate``), INV-R2 (never
creates a terminal-row-without-envelope — match-only), INV-R3 (idempotent,
CAS-loser no-ops), INV-R4 (no SessionModeChangedEvent emit), INV-R5 (no
supervisor/domain change; application-layer only).
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from app.domain.models.session import SessionStatus

logger = logging.getLogger(__name__)

# Envelope-type discriminators (stored verbatim by the supervisor's
# ``persist_terminal``; see mailbox_supervisor.py:613/:873 and
# MailboxEnvelopeType in domain/models/mailbox_envelope.py:30/:34).
_RESULT_READY = "RESULT_READY"
_CANCEL_ACK = "CANCEL_ACK"
# Payload sentinels that map a child to the row's TIMED_OUT terminal state:
# ResultReadyOutcome.TIMED_OUT.value (mailbox_envelope.py:74) and
# CancelAckPayload.final_state == "force_terminated" (mailbox_envelope.py:247).
_OUTCOME_TIMED_OUT = "timed_out"
_FINAL_STATE_FORCE_TERMINATED = "force_terminated"

# The closed terminal SessionStatus pair (domain/models/session.py:23-24;
# both absorbing — db_session_repository.py:314-316).
_TERMINAL_STATUSES = (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT)


def row_terminal_from_envelope(
    envelope_type: str, payload: dict[str, Any] | None
) -> tuple[SessionStatus, str]:
    """Pure: derive the ``(status, reason)`` a lagged child row should take to
    MATCH its already-persisted terminal envelope.

    Mirrors the runner's own row writes (agent_task_runner.py:4345/4396/4424):
    only a real wallclock timeout / force-terminate maps to ``TIMED_OUT``; every
    other real outcome (success / failed / cancelled / needs-authorization) maps
    to ``COMPLETED``. The rich outcome detail lives in the envelope; the row only
    distinguishes the two-state terminal pair. Totals safely on missing / null /
    unknown payload fields (defaults to ``COMPLETED/natural``).
    """
    p = payload or {}
    if envelope_type == _RESULT_READY and p.get("outcome") == _OUTCOME_TIMED_OUT:
        return SessionStatus.TIMED_OUT, "watchdog_timeout"
    if (
        envelope_type == _CANCEL_ACK
        and p.get("final_state") == _FINAL_STATE_FORCE_TERMINATED
    ):
        return SessionStatus.TIMED_OUT, "watchdog_timeout"
    return SessionStatus.COMPLETED, "natural"


async def terminalize_row(
    child_session_id: str,
    status: SessionStatus,
    reason: str,
    *,
    state_machine: Any,
    uow_factory: Any,
) -> bool:
    """Terminalize one child row via the A4-1 ``ssm.terminate`` CAS (INV-R1).

    Returns ``True`` iff this call won the CAS (the row was non-terminal and is
    now terminal); ``False`` if DI is absent (defensive no-op) or the row had
    already raced to terminal (e.g. the runner's own write won — INV-R3). Opens
    its own UoW so each child is an isolated transaction. No
    ``emit_session_mode_changed`` (INV-R4) and no bg-terminal notification
    (foreground children — spec §4.1). ``asyncio.CancelledError`` propagates
    (the DBUnitOfWork rolls back on the exceptional exit).
    """
    if state_machine is None or uow_factory is None:
        logger.warning(
            "child_row_reaper: terminalize_row skipped — missing DI "
            "(state_machine/uow_factory) for child=%s",
            child_session_id,
        )
        return False
    async with uow_factory() as uow:
        ok = await state_machine.terminate(
            child_session_id, status, reason, session_repo=uow.session
        )
        # Caller-owned commit (ssm.terminate does not commit). The UoW's
        # __aexit__ commit is then a no-op on the now-clean session.
        await uow.db_session.commit()
        return ok


async def match_terminal_envelope_to_row(
    child_id: str,
    run_id: str | None,
    wu_id: str | None,
    *,
    state_machine: Any,
    uow_factory: Any,
    envelope_store: Any,
    session_repo: Any,
) -> bool:
    """Per-child: terminalize a lagged RUNNING coordinator child to MATCH its
    already-persisted terminal envelope. Returns ``True`` iff it terminalized the
    row; ``False`` for every skip path. Skips (no row write) when:

    * the row is gone, already terminal, or no longer ``execution_mode ==
      'foreground'`` (R24 defensive re-read — a backgrounded/recoverable reopened
      turn is owned by ``reconcile_running_background_at_boot``, not the reaper);
    * the child has no coordinator lineage (``run_id`` / ``wu_id`` NULL — a
      non-coordinator mailbox subagent);
    * NO persisted envelope matches both ``wu_id`` AND ``child_session_id``
      (match-only — never backfills; a no-envelope child is the spec §6 residual).
    """
    row = await session_repo.get_by_id(child_id)
    if row is None:
        return False
    if row.status in _TERMINAL_STATUSES:
        return False
    if row.execution_mode != "foreground":
        return False
    if run_id is None or wu_id is None:
        return False
    envelopes = await envelope_store.find_terminal_envelopes_by_run(run_id)
    match = next(
        (
            e
            for e in envelopes
            if e.work_unit_id == wu_id and e.child_session_id == child_id
        ),
        None,
    )
    if match is None:
        return False
    status, reason = row_terminal_from_envelope(match.envelope_type, match.payload)
    return await terminalize_row(
        child_id,
        status,
        reason,
        state_machine=state_machine,
        uow_factory=uow_factory,
    )


@dataclass
class SweepStats:
    """Outcome counters for one startup sweep (logged when non-trivial)."""

    scanned: int = 0
    terminalized: int = 0
    skipped: int = 0
    errored: int = 0


async def sweep_running_mailbox_children(
    *,
    session_repo: Any,
    envelope_store: Any,
    state_machine: Any,
    uow_factory: Any,
) -> SweepStats:
    """The sole reaper: scan RUNNING foreground mailbox subagent children and
    terminalize each whose terminal envelope is already persisted (match-only).

    The query (``find_running_mailbox_children``) is NOT wrapped here — a query
    failure propagates to the caller's OUTER best-effort try (main.py) so a
    query/DI failure cannot abort lifespan startup. The INNER per-child try here
    isolates one bad child so it cannot stop the sweep. ``asyncio.CancelledError``
    always propagates.
    """
    stats = SweepStats()
    children = await session_repo.find_running_mailbox_children()
    for child in children:
        stats.scanned += 1
        try:
            did = await match_terminal_envelope_to_row(
                child.session_id,
                child.coordinator_run_id,
                child.work_unit_id,
                state_machine=state_machine,
                uow_factory=uow_factory,
                envelope_store=envelope_store,
                session_repo=session_repo,
            )
            if did:
                stats.terminalized += 1
            else:
                stats.skipped += 1
        except asyncio.CancelledError:
            raise
        except Exception:
            stats.errored += 1
            logger.exception(
                "child_row_reaper: failed to reconcile child=%s", child.session_id
            )
    return stats
