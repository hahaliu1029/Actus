"""EscalationProvider Protocol — pluggable Stage P.2 escalation source.

PE-0 only registers one provider: SmartApproveProvider. PE-N adds no
plugin API; adding a provider requires editing DefaultPermissionEngine's
constructor.
"""

from __future__ import annotations

from typing import Optional, Protocol

from app.domain.models.tool_result import ToolOutcome
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.tool_call_spec import ToolCallSpec


class EscalationProvider(Protocol):
    """Async escalation source used by PE in Stage P.2.

    Implementations MUST NOT call ApprovalStateWriter (INV-1b). They
    return a ToolOutcome; PE will persist downstream effects.

    Errors policy (codex P2-11): provider exceptions / timeouts MUST be
    caught by the caller (DefaultPermissionEngine) and converted to an
    Asked outcome with reason.code='smart_approve_unavailable_falling_through'
    — never bubble to HTTP.
    """

    name: str

    async def resolve(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        upstream_outcome: Optional[ToolOutcome],
    ) -> ToolOutcome: ...
