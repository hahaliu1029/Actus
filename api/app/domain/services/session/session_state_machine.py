"""SessionStateMachine — per-session control mode owner.

INV-3 (PE-0 CI gate): no method on this ABC or its impls may call
ApprovalStateWriter.write / write_audit_only / delete_grant.
INV-4-soft (PE-0): non-SSM writes to sessions.status are warning-only.
INV-4-hard (A4-1): writes outside SSM fail CI.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Mapping

from app.domain.models.session import SessionStatus


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
