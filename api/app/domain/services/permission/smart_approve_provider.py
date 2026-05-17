"""SmartApproveProvider — adapts the legacy string-returning SmartApprove
service into the EscalationProvider Protocol.

Timeouts and unexpected exceptions fall through to Asked (not Denied) —
this preserves the existing react_graph._stage_p2_smart_approve behavior
(codex round-1 P2-11). Denied is reserved for "smart_approve LLM
explicitly said deny."

C-P1-2 correction applied: SmartApprove.evaluate real signature is
keyword-friendly, NOT (call, ctx). This adapter calls it with explicit
kwargs: tool_name, tool_args, risk_level, matched_patterns, task_context.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.domain.models.tool_result import (
    AllowSuccess,
    Asked,
    DecisionReason,
    Denied,
    ToolOutcome,
)
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.tool_call_spec import ToolCallSpec

_FALL_THROUGH_CODE = "smart_approve_unavailable_falling_through"


class SmartApproveProvider:
    """Adapts SmartApprove.evaluate(tool_name, tool_args, risk_level,
    matched_patterns, task_context) -> str into EscalationProvider.resolve
    -> ToolOutcome.

    String mapping:
      "approve"  -> AllowSuccess(via="smart_approve")
      "deny"     -> Denied(type="smart_approve", code="llm_denied")
      any other  -> Asked(type="smart_approve", code="escalated_to_user")
      timeout    -> Asked(type="approval_policy", code=_FALL_THROUGH_CODE)
      exception  -> Asked(type="approval_policy", code=_FALL_THROUGH_CODE)

    P1#4: When medium_only=True, HIGH-risk tool calls are NOT sent to the
    LLM for automatic evaluation; they fall directly through to Asked so the
    user must confirm.  This mirrors the legacy smart_approve_medium_only
    behaviour where HIGH always requires explicit human confirmation.
    """

    name = "smart_approve"

    def __init__(
        self,
        smart_approve: Any,
        timeout_seconds: float = 30.0,
        medium_only: bool = False,
    ) -> None:
        self._inner = smart_approve
        self._timeout = timeout_seconds
        self._medium_only = medium_only

    async def resolve(
        self,
        call: ToolCallSpec,
        ctx: EvaluationContext,
        upstream_outcome: Optional[ToolOutcome],
    ) -> ToolOutcome:
        cid = f"{call.session_id}:{call.tool_call_id}"
        risk_level = (
            call.risk_assessment.final_level.name.lower()
            if call.risk_assessment is not None
            else "unknown"
        )

        # P1#4: medium_only=True → skip LLM evaluation for HIGH-risk tools.
        # Return Asked so the user is required to confirm explicitly.
        if self._medium_only and risk_level == "high":
            return Asked(
                content="高风险操作需要人工确认（smart_approve_medium_only=True）",
                reason=DecisionReason(
                    type="approval_policy",
                    code="smart_approve_medium_only_high_skip",
                    message="smart_approve skipped for HIGH risk (medium_only=True); escalating to user",
                ),
                confirmation_id=cid,
            )
        matched_patterns = (
            list(call.risk_assessment.matched_patterns)
            if call.risk_assessment is not None
            else []
        )
        try:
            raw = await asyncio.wait_for(
                self._inner.evaluate(
                    tool_name=call.tool_name,
                    tool_args=dict(call.tool_args),
                    risk_level=risk_level,
                    matched_patterns=matched_patterns,
                    task_context=ctx.request_id or "",
                ),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            return Asked(
                content="智能审批超时，请人工确认",
                reason=DecisionReason(
                    type="approval_policy",
                    code=_FALL_THROUGH_CODE,
                    message="smart_approve timed out; falling through to user confirmation",
                ),
                confirmation_id=cid,
            )
        except Exception:
            return Asked(
                content="智能审批不可用，请人工确认",
                reason=DecisionReason(
                    type="approval_policy",
                    code=_FALL_THROUGH_CODE,
                    message="smart_approve raised; falling through to user confirmation",
                ),
                confirmation_id=cid,
            )

        if raw == "approve":
            return AllowSuccess(
                content="smart_approve LLM 自动放行",
                data={"via": "smart_approve", "grant_scope": "session"},
            )
        if raw == "deny":
            return Denied(
                content="smart_approve LLM 拒绝",
                reason=DecisionReason(
                    type="smart_approve",
                    code="llm_denied",
                    message="smart_approve returned deny",
                ),
            )
        # Any other string (e.g. "escalate") — fall through to user
        return Asked(
            content="智能审批升级，需人工确认",
            reason=DecisionReason(
                type="smart_approve",
                code="escalated_to_user",
                message="smart_approve escalated decision to user",
            ),
            confirmation_id=cid,
        )
