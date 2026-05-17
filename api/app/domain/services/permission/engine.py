"""PermissionEngine ABC.

Type-safe facade over the existing _run_policy_chain (react_graph.py:260).
Three entry points:

  evaluate(call, ctx)            -> ToolOutcome
    First-time evaluation in tool_node. May return AllowSuccess / AllowError
    / Denied / Asked / Passthrough. INV-5 path A.

  preflight_resume(call, ctx, signal) -> PreflightResumeResult
    HTTP-layer entry (agent_service.preflight_resume_tool_confirmation).
    Atomically claims the pending ConfirmationQueue entry with a 16-byte
    claim_nonce. Does NOT write grant.

  commit_resume(call, ctx, signal, claim_nonce) -> ToolOutcome
    Graph-layer entry (interrupt_helper after LangGraph resume). Validates
    the nonce against the queue and writes the grant (or audit-only for
    once-deny). Returns the final ToolOutcome that tool_node consumes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.models.tool_result import ToolOutcome
from app.domain.services.permission.context import (
    EvaluationContext,
    PreflightResumeResult,
    ResumeSignal,
)
from app.domain.services.permission.tool_call_spec import ToolCallSpec


class PermissionEngine(ABC):
    """R5 CS4 unbreakable: PE is the only upstream caller of
    ApprovalStateWriter.{write, write_audit_only, delete_grant}.
    INV-1b enforces this at CI.

    INV-2 (PE-0 CI gate): PE never calls SessionStateMachine mutators
    (request_takeover, release_takeover, enter_finishing, complete,
    transition). PE only READS session mode via get_mode_with_revision.
    """

    @abstractmethod
    async def evaluate(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
    ) -> ToolOutcome: ...

    @abstractmethod
    async def preflight_resume(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        resume_signal: ResumeSignal,
    ) -> PreflightResumeResult: ...

    @abstractmethod
    async def commit_resume(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        resume_signal: ResumeSignal,
        claim_nonce: str,
    ) -> ToolOutcome: ...

    @abstractmethod
    async def cleanup_pending_confirmation(
        self,
        session_id: str,
        tool_call_id: str,
    ) -> None:
        """Remove a 'processing' queue entry that will never be committed.

        Called by interrupt_helper on error paths where pe.commit_resume was
        never reached (e.g. SSM failure) or returned a PolicyConflict that
        left the queue entry in 'processing' state.  Swallows all exceptions
        so cleanup failures never shadow the original error.
        """
        ...
