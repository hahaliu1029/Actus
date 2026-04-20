"""R5 CS4 domain repository ABC for tool_approval_grants."""

from abc import ABC, abstractmethod
from typing import Optional

from app.domain.models.approval_grant import ApprovalDecision, ApprovalGrant


class ApprovalGrantRepository(ABC):
    """Grants repository contract.

    Phase 1 scope: create + narrow read paths + delete (for writer rollback).
    Revoke / policy_id / supersede chains are Phase 2 PermissionEngine concerns
    and are intentionally absent.
    """

    @abstractmethod
    async def create(self, decision: ApprovalDecision) -> str:
        """Insert a new grant row. Returns ``decision_id``.

        Raises ``sqlalchemy.exc.IntegrityError`` (Postgres UNIQUE violation)
        when ``confirmation_id`` already exists or when SmartApprove partial
        UNIQUE catches a duplicate ``(user, session, tool, arg_digest, effect)``
        with ``confirmation_id IS NULL``.
        """

    @abstractmethod
    async def find_by_confirmation_id(
        self, confirmation_id: str,
    ) -> Optional[ApprovalGrant]:
        """Idempotent read-back for ``confirmation_id`` UNIQUE collision path.

        Returns ``None`` when no row exists (genuinely unexpected
        IntegrityError).
        """

    @abstractmethod
    async def find_active_grants(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
    ) -> list[ApprovalGrant]:
        """Hot read for Reader.check().

        Returns non-expired grants for ``(user_id, tool_name)`` across
        ``always`` scope and the given ``session_id``. Reader applies
        priority ordering (always_deny > always_allow > session_allow).
        """

    @abstractmethod
    async def find_smart_approve_dedup(
        self,
        user_id: str,
        session_id: Optional[str],
        tool_name: str,
        arg_digest: str,
        effect: str,
    ) -> Optional[ApprovalGrant]:
        """Partial-UNIQUE read-back for the SmartApprove (confirmation_id IS NULL) path.

        Matches index ``ux_tool_approval_grants_smart_approve_dedup``.
        """

    @abstractmethod
    async def delete(self, decision_id: str) -> None:
        """Remove a grant row.

        Used by writer rollback when background ``task.resume()`` kickoff
        fails — allows next ``/resume`` with same ``confirmation_id`` to
        become ``newly_created=True`` again.
        """
