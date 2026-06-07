"""SessionStateMachine — per-session control mode owner.

INV-3 (PE-0 CI gate): no method on this ABC or its impls may call
ApprovalStateWriter.write / write_audit_only / delete_grant.
INV-4-hard (A4-1, SHIPPED): the SSM subpackage is the sole CALLER of the
sessions.status repo mutators; non-SSM status writes fail CI (Gate A/B in
tests/invariants/test_inv4_ssm_single_writer.py).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping

from app.domain.models.session import SessionStatus
from app.domain.repositories.session_repository import SessionRepository


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
        the future A4-2 single-emitter; unused by A4-1.

        NOTE: heterogeneous transaction contract vs the dormant CAS mutators
        above (which open their own UoW). Reconciliation is A4-2.
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
