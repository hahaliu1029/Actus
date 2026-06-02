"""McpSource — PE-2 stateless MCP adapter.

Per spec §3.1/§3.2 + Q1=(a) behavior parity: MCP tools carry no on-disk scan
and no risk metadata. McpSource returns a constant LOW RiskAssessment so the
default (AUTO / no-policy) path auto-allows (default_engine step 7a) —
preserving today's fire-and-forget behavior — while user ApprovalPolicy
(ASK/DENY) + tool_approval grants still apply (steps 5/6/7b).

Stateless: no refresher, no Redis, no live ``_mcp_tool`` dependency, so it
registers at PE construction like NativeSource (no late-register). MCP is
tool-level authorization: step 5.5 ``dataclasses.replace`` only swaps
``risk_assessment`` and does NOT backfill the top-level ``primary_arg`` /
``dir_arg`` / ``arg_digest`` the reader/grant matcher reads, so arg-level
pattern rules cannot match MCP (matches cc's no-parameter-level-pattern model;
spec §3.4).
"""

from __future__ import annotations

from typing import ClassVar

from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel


class McpSource(PermissionSource):
    """PE-2: MCP source — constant non-dangerous baseline (behavior parity)."""

    tool_source: ClassVar[str] = "mcp"

    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        # ``final_level`` drives the risk gate (step 7: LOW ∉ dangerous → auto-allow).
        # ``matched_patterns`` / ``risk_reason`` can flow into the ASK confirmation
        # payload, so we set safe empty / baseline values. The grant matcher reads
        # the TOP-LEVEL ToolCallSpec args (primary_arg / dir_arg / arg_digest), not
        # these, so arg-level pattern rules cannot match MCP (spec §3.4). LOW (not
        # NONE) marks "real external tool, not classified dangerous" and stays
        # distinguishable from native NONE-tier in audit (spec §3.3).
        return RiskAssessment(
            tool_name=call.tool_name,
            tool_args=dict(call.tool_args),
            static_level=RiskLevel.LOW,
            dynamic_level=RiskLevel.LOW,
            final_level=RiskLevel.LOW,
            risk_reason="mcp baseline (PE-2 parity: no per-tool risk scoring)",
            matched_patterns=[],
            suggested_alternative=None,
            primary_arg=call.primary_arg or "",
            dir_arg=call.dir_arg,
            arg_digest=call.arg_digest or "",
        )
