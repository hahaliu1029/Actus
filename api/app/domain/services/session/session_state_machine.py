"""SessionStateMachine — per-session control mode owner.

INV-3 (PE-0 CI gate): no method on this ABC or its impls may call
ApprovalStateWriter.write / write_audit_only / delete_grant.
INV-4-hard (A4-1, SHIPPED): the SSM subpackage is the sole CALLER of the
sessions.status repo mutators; non-SSM status writes fail CI (Gate A/B in
tests/invariants/test_inv4_ssm_single_writer.py).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

from app.domain.models.session import SessionStatus
from app.domain.repositories.session_repository import SessionRepository
from app.domain.models.event import SessionModeChangedEvent
from app.domain.services.session.mode_event import (
    ModeChangedEventSink,
    build_session_mode_changed_event,
)


class SessionStateMachine(ABC):
    """Per-session mode owner; concrete impl in default_state_machine.py."""

    @abstractmethod
    async def get_mode(self, session_id: str) -> SessionStatus: ...

    @abstractmethod
    async def get_mode_with_revision(
        self, session_id: str,
    ) -> tuple[SessionStatus, int]:
        """Return (mode, mode_revision) snapshot.

        Used by PermissionEngine to detect TAKEOVER race during slow
        stages. mode_revision is BIGINT strict-monotonic +1 per
        transition. PE compares revision values, NOT mode values
        (the latter misses RUNNING -> TAKEOVER -> RUNNING round-trip).
        """

    @abstractmethod
    async def request_takeover(self, session_id: str, reason: str) -> None:
        """RUNNING -> TAKEOVER_PENDING. mode_revision +1."""

    @abstractmethod
    async def release_takeover(self, session_id: str) -> None:
        """TAKEOVER -> RUNNING. mode_revision +1."""

    @abstractmethod
    async def enter_finishing(self, session_id: str) -> None:
        """RUNNING -> FINISHING. Coordinates with b4-1d drain contract."""

    @abstractmethod
    async def complete(self, session_id: str) -> None:
        """FINISHING -> COMPLETED. Terminal."""

    @abstractmethod
    async def transition(
        self,
        session_id: str,
        from_state: SessionStatus,
        to_state: SessionStatus,
        reason: str,
        extra_values: Mapping[str, Any] | None = None,
    ) -> bool:
        """CAS: UPDATE ... WHERE status=:from_state, mode_revision+=1.
        Returns True if row was updated (won the race), False otherwise.

        ``extra_values`` (optional) is forwarded to
        ``SessionRepository.transition_status`` so terminal metadata
        (``completed_at`` / ``terminal_reason`` / ``execution_phase``) can be
        written atomically alongside the status CAS."""

    @abstractmethod
    async def set_mode(
        self,
        session_id: str,
        to: SessionStatus,
        reason: str,
        *,
        session_repo: SessionRepository,
    ) -> None:
        """A4-1 caller-owned non-terminal status write.

        Issues exactly ``session_repo.update_status(session_id, to)`` (blind
        by-id, bumps mode_revision) and returns. Does NOT read the revision,
        NOT emit, NOT open or commit a transaction — the CALLER owns the UoW,
        commit, lock, and event emit. ``reason`` is carried for telemetry and
        consumed by the A4-2 single-emitter: the caller passes it to
        ``emit_session_mode_changed`` when building the SessionModeChangedEvent.
        ``set_mode`` itself never emits.

        NOTE: heterogeneous transaction contract vs the dormant CAS mutators
        above (which open their own UoW). The control-mode emit is reconciled by
        A4-2's ``emit_session_mode_changed`` (single construct-and-dispatch
        entry with a caller-owned sink).
        """

    @abstractmethod
    async def terminate(
        self,
        session_id: str,
        to: SessionStatus,
        terminal_reason: str,
        *,
        session_repo: SessionRepository,
    ) -> bool:
        """A4-1 caller-owned terminal status write.

        Issues exactly ``session_repo.update_to_terminal(session_id, to,
        terminal_reason)`` and returns its idempotency bool — ``False`` when the
        row is already terminal OR its execution_phase is terminating/terminated;
        else ``True``. Does NOT open or commit a transaction (caller-owned).
        """

    async def emit_session_mode_changed(
        self,
        session_id: str,
        *,
        to: SessionStatus | str,
        from_mode: Optional[str],
        reason: str,
        mode_revision: Optional[int],
        sink: ModeChangedEventSink,
    ) -> SessionModeChangedEvent:
        """A4-2 SINGLE emit entry. Builds the canonical SessionModeChangedEvent
        and dispatches it to the caller-owned sink. Does NOT read the repo, open
        a txn, hold a uow_factory, or touch seq/idle — the CALLER owns the
        runtime sink AND the read-your-writes mode_revision (which it MUST pass
        in; the SSM never re-reads it, to avoid a concurrent-transition causal
        mismatch — see spec §6.2). Concrete (not abstract): stateless, so every
        ABC subclass inherits it."""
        event = build_session_mode_changed_event(
            to=to, from_mode=from_mode, reason=reason, mode_revision=mode_revision
        )
        await sink(session_id, event)
        return event
