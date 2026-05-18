"""ToolCallSpec — PE input DTO. Frozen dataclass, no FastAPI / SQLAlchemy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from app.domain.services.permission.source_metadata import SourceMetadata
    from app.domain.services.risk_assessor import RiskAssessment


@dataclass(frozen=True)
class ToolCallSpec:
    """Per-tool-call invocation context input to PermissionEngine.

    ``tool_args`` should be already sanitized (no PII / secret literals).
    ``primary_arg`` / ``dir_arg`` / ``arg_digest`` follow S1 RiskAssessment
    normalization (``api/app/domain/services/risk_assessor.py``) so they
    match ``tool_approval_rules.command_pattern`` / ``dir_pattern`` keys.

    PE-1 adds ``source_metadata`` — typed Union of source-specific metadata
    set by the caller (tool_node) before invoking pe.evaluate. SkillSource
    requires SkillCallMetadata here; NativeSource ignores it and falls back
    to ``risk_assessment``.

    HARD: when ``tool_source == "skill"``, caller-prefilled
    ``risk_assessment`` is IGNORED — SkillSource recomputes from
    ``source_metadata`` (Risk #1 hard rule, spec §5.5).
    """

    tool_name: str
    tool_args: Mapping[str, Any]
    tool_source: str           # "native" | "skill" | "mcp" | "a2a"
    user_id: str
    session_id: str
    primary_arg: str | None = None
    dir_arg: str | None = None
    arg_digest: str | None = None
    risk_assessment: Optional["RiskAssessment"] = None
    tool_call_id: str = ""     # LangChain tool call id; empty for synthetic
    source_metadata: Optional["SourceMetadata"] = None  # PE-1 §3.2
