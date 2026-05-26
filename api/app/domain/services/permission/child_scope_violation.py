"""ChildScopeViolation raised by DefaultPermissionEngine when ChildScopeGate denies.

Caught by CoordinatorChildRunner finalizer (PR-4) -> RESULT_READY(needs_authorization).
"""
from __future__ import annotations

from app.domain.services.permission.child_scope_gate import ScopeDecision


class ChildScopeViolation(Exception):
    def __init__(
        self,
        decision: ScopeDecision,
        *,
        tool_name: str,
        target_path: str | None = None,
    ) -> None:
        super().__init__(f"ChildScopeGate denied {tool_name}: {decision.value}")
        self.decision = decision
        self.tool_name = tool_name
        self.target_path = target_path
