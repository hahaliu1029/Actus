"""EvaluationContext, ResumeSignal, PreflightResumeResult.

These are immutable DTOs (frozen dataclass) — passed by value between
HTTP boundary, PermissionEngine, and graph node. No SQLAlchemy types.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import ToolOutcome

if TYPE_CHECKING:
    from app.domain.services.permission.child_permission_context import (
        ChildPermissionContext,
    )
    from app.domain.services.permission.confirmation_queue import ConfirmationDetail


@dataclass(frozen=True)
class EvaluationContext:
    """Per-evaluate call context.

    session_mode_revision is the BIGINT monotonic counter from
    sessions.mode_revision (strict monotonic +1 per SSM transition).
    PE uses it to detect TAKEOVER race during slow stages (SmartApprove
    LLM call). Value comparison on session_mode alone misses
    RUNNING -> TAKEOVER -> RUNNING round-trip.
    """

    session_mode: SessionStatus
    session_mode_revision: int
    retry_count: int = 0
    prior_outcomes: tuple[ToolOutcome, ...] = ()
    # ^ per-session history of ToolOutcome variants for multi-agent slot (Phase 3+).
    # permission/ is in the Rule 1 whitelist (PE-0 expansion).
    request_id: str = ""
    # multi-agent slot — Phase 3+ scope, not used in PE-0
    parent_agent_id: Optional[str] = None
    # [C2 PR-2 §5.4] None when root session; injected by ChildAgentRunnerFactory
    child_permission_context: Optional["ChildPermissionContext"] = None


@dataclass(frozen=True)
class ResumeSignal:
    """HTTP-layer resume payload, lifted into a domain DTO.

    Built by agent_service.preflight_resume_tool_confirmation from
    request.tool_confirmation (api/app/interfaces/endpoints/session_routes.py
    POST /v1/sessions/{sid}/chat — tool_confirmation sub-field carries
    tool_call_id / action / scope).

    grant_scope is the wide wire union — "once" is the audit-only
    short-lived deny, "session" / "always" are persistent grants.
    See plan preamble "Spec / reality reconciliations" for context.
    """

    confirmation_id: str
    action: Literal["approve", "deny"]
    grant_scope: Literal["once", "session", "always"]
    actor: str = "user_click"


@dataclass(frozen=True)
class PreflightResumeResult:
    """Result of pe.preflight_resume — race-safe claim handoff to
    commit_resume. claim_nonce is 16-byte hex (32 chars), stored in
    Redis Hash under (session_id, tool_call_id) via
    ConfirmationQueue.mark_processing_if_pending.

    Sweeper rescue path (find_orphaned_processing) uses
    processing_started_at to detect workers that died after preflight
    but before commit_resume ran.
    """

    claim_nonce: str
    processing_started_at: datetime
    detail: "ConfirmationDetail"
