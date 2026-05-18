"""Canonical SkillCallMetadata + SourceMetadata TypeAlias.

This module is the no-cycle home for source-specific metadata DTOs.
``sources/base.py`` (PermissionSource ABC) depends on ``ToolCallSpec``;
``ToolCallSpec.source_metadata`` type depends on ``SourceMetadata``.
Placing ``SourceMetadata`` in ``sources/base.py`` would create the cycle
``tool_call_spec → sources.base → tool_call_spec``. Keeping it here (sibling
of ``tool_call_spec.py``, zero PE-internal deps) breaks the cycle; the
``sources/`` subpackage only re-exports.

Spec ref: 2026-05-18-pe-1-skill-source-internalize-design.md §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.risk_assessor import RiskLevel


@dataclass(frozen=True)
class SkillCallMetadata:
    """Per spec §3.1 Q3.1 (minimal + tool_name + skill_id + content_hash).

    Attributes
    ----------
    tool_name:
        LangChain StructuredTool name — refresher's binding lookup key
        (``SkillTool._tool_bindings[tool_name]``; refresh_risk_if_stale
        also takes tool_name not skill_id, spec Round 1 P0#2).
    skill_id:
        Used only for Redis lock/cache key + log/exception messages.
        NOT included in record_decision attrs (spec Risk #2; would be
        silently dropped by CANONICAL_ATTRIBUTES filter).
    content_hash:
        Snapshot of ``SkillsGuard.compute_content_hash(skill_dir)`` at
        build time. Refresher compares against on-disk to decide
        fresh / refreshed. ``None`` when skill was never scanned.
    risk_level:
        Cached final_risk from ``SkillTool._tool_bindings[tool_name]
        ["final_risk"]`` — the in-place-updated source of truth (NOT
        ``tool_fn.metadata['risk_level']`` which is a stale snapshot).
    runtime_type:
        Skill runtime (NATIVE / MCP / A2A).
    trust_origin:
        "builtin" | "user_installed" | etc. — input to SkillRiskAssessor
        for risk_reason string.
    scan_verdict:
        "safe" | "dangerous" | "unscanned" — last trust matrix scan result.
    """

    tool_name: str
    skill_id: str
    content_hash: str | None
    risk_level: RiskLevel
    runtime_type: SkillRuntimeType
    trust_origin: str
    scan_verdict: str


SourceMetadata: TypeAlias = "SkillCallMetadata"
# PE-2 will widen to: SourceMetadata = SkillCallMetadata | McpCallMetadata
# PE-3 will widen to: SourceMetadata = SkillCallMetadata | McpCallMetadata | A2ACallMetadata
