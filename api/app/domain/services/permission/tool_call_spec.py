"""ToolCallSpec — PE input DTO. Frozen dataclass, no FastAPI / SQLAlchemy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from app.domain.services.risk_assessor import RiskAssessment


@dataclass(frozen=True)
class ToolCallSpec:
    """Per-tool-call invocation context input to PermissionEngine.

    `tool_args` should be already sanitized (no PII / secret literals).
    `primary_arg` / `dir_arg` / `arg_digest` follow S1 RiskAssessment
    normalization (`api/app/domain/services/risk_assessor.py`) so they
    match `tool_approval_rules.command_pattern` / `dir_pattern` keys.
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
