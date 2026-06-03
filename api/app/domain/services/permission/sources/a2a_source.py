"""A2aSource — PE-3 stateless A2A adapter.

Per spec §3 + Q1=(a) behavior parity: A2A tools carry no on-disk scan and no
risk metadata. A2aSource returns a constant LOW RiskAssessment so the default
(AUTO / no-policy) path auto-allows (default_engine step 7a) — preserving
today's fire-and-forget behavior — while user ApprovalPolicy(ASK/DENY) +
tool_approval grants still apply (steps 4/5/6/7b).

Stateless: no refresher, no Redis, no live A2ATool.manager dependency, so it
registers at PE construction like McpSource/NativeSource (no late-register).
A2A is tool-level authorization: step 5.5 ``dataclasses.replace`` only swaps
``risk_assessment`` and does NOT backfill the top-level ``primary_arg`` /
``dir_arg`` / ``arg_digest`` the reader/grant matcher reads, so per-agent
rules cannot match A2A (the target agent is the ``id`` arg, not the tool name;
spec §7). Mirrors cc's no-parameter-level-pattern stance.
"""

from __future__ import annotations

from typing import ClassVar

from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel


class A2aSource(PermissionSource):
    """PE-3: A2A source — constant non-dangerous baseline (behavior parity)."""

    tool_source: ClassVar[str] = "a2a"

    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        # RiskAssessment has 11 required fields, no defaults (risk_assessor.py:33-47).
        # ``final_level`` is the DECISION field (step 7 risk gate). The rest are
        # required + display-adjacent safe baselines, NOT cosmetic: ``matched_patterns``
        # flows into the ASK ConfirmationDetail + SmartApprove provider (so give []);
        # ``risk_reason`` is required and shown only on the skill step-9 branch (a2a
        # uses the canned "user confirmation required", so a2a's risk_reason is not
        # displayed today, but still set a safe baseline). ``primary_arg/dir_arg/
        # arg_digest`` are read from the TOP-LEVEL ToolCallSpec, not from here
        # (spec §7). LOW (not NONE) marks "real external tool, not classified
        # dangerous" + stays distinguishable in audit.
        return RiskAssessment(
            tool_name=call.tool_name,
            tool_args=dict(call.tool_args),
            static_level=RiskLevel.LOW,
            dynamic_level=RiskLevel.LOW,
            final_level=RiskLevel.LOW,          # decision field (step 7 gate)
            risk_reason="a2a baseline (PE-3 parity: no per-tool risk scoring)",
            matched_patterns=[],
            suggested_alternative=None,
            primary_arg=call.primary_arg or "",
            dir_arg=call.dir_arg,
            arg_digest=call.arg_digest or "",
        )
