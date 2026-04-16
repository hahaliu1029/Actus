"""Skill 专用风险评估器 — 独立于 RiskAssessor，不改其签名"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel


def _extract_skill_primary_arg(tool_args: dict[str, Any]) -> str:
    """Extract skill call's primary_arg (for always scope fnmatch)"""
    strings = [
        str(v) for k, v in sorted(tool_args.items())
        if isinstance(v, str)
    ]
    if strings:
        return "\0".join(strings)
    return json.dumps(tool_args, sort_keys=True)[:200]


def _compute_skill_arg_digest(
    tool_name: str,
    tool_args: dict[str, Any],
    risk_level: RiskLevel,
    runtime_type: SkillRuntimeType,
) -> str:
    if runtime_type == SkillRuntimeType.A2A:
        return _hash_all(tool_args)
    if risk_level >= RiskLevel.HIGH:
        return _hash_all(tool_args)
    if runtime_type == SkillRuntimeType.MCP:
        return _hash_all(tool_args)
    # native + medium/low: tool-level cache
    return hashlib.sha256(tool_name.encode()).hexdigest()[:16]


def _hash_all(tool_args: dict[str, Any]) -> str:
    normalized = json.dumps(tool_args, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


class SkillRiskAssessor:
    """Skill tool risk assessment, independent from native RiskAssessor"""

    def assess(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        risk_level: RiskLevel,
        runtime_type: SkillRuntimeType,
        trust_origin: str,
    ) -> RiskAssessment:
        primary_arg = _extract_skill_primary_arg(tool_args)
        arg_digest = _compute_skill_arg_digest(
            tool_name, tool_args, risk_level, runtime_type,
        )
        return RiskAssessment(
            tool_name=tool_name,
            tool_args=tool_args,
            static_level=risk_level,
            dynamic_level=RiskLevel.NONE,
            final_level=risk_level,
            risk_reason=f"skill risk: trust={trust_origin}, runtime={runtime_type.value}",
            matched_patterns=[],
            suggested_alternative=None,
            primary_arg=primary_arg,
            dir_arg=None,
            arg_digest=arg_digest,
        )
