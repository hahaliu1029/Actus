"""R5 CS4 domain model: ApprovalGrant + ApprovalDecision DTO.

Grant 是"用户决定"的持久记录，不是"tool 执行结果"的记录：
- 用户点 approve 的那一刻 decision 落地；tool 后续成功/失败与 grant 持久性无关
- DB UNIQUE on ``confirmation_id`` 把"原子 claim + 单 writer + 幂等 resume"合并成同一机制
- ``effect='deny'`` 也写一条 grant（Reader Phase 1 不 surface，但审计保留决策证据）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatch
from typing import Literal, Optional

from pydantic import BaseModel, Field


Scope = Literal["session", "always"]
Effect = Literal["approve", "deny"]
ToolSource = Literal["native", "mcp", "a2a", "skill"]
SourceType = Literal["user_click", "smart_approve", "system"]


class ApprovalGrant(BaseModel):
    """CS4 grant record: persistent evidence of an approval decision."""

    decision_id: str
    user_id: str
    session_id: Optional[str] = None
    tool_name: str
    tool_source: ToolSource
    arg_digest: str
    primary_arg: str = ""
    dir_arg: str = ""
    scope: Scope
    effect: Effect
    source_type: SourceType
    confirmation_id: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        from_attributes = True

    def matches(self, primary_arg: str, dir_arg: Optional[str]) -> bool:
        """Shell-glob match on ``primary_arg`` + optional ``dir_arg``.

        Mirrors the glob semantics of the legacy ``ToolApprovalRule.matches``.
        PE-4d1 retired the legacy ``tool_approval_rules`` fallback, so the
        Reader now evaluates grants only — this is the sole glob matcher on the
        live read path.
        """
        if not fnmatch(primary_arg, self.primary_arg):
            return False
        if self.dir_arg:
            if dir_arg is None:
                return False
            if not fnmatch(dir_arg, self.dir_arg):
                return False
        return True


@dataclass(frozen=True)
class ApprovalDecision:
    """CS4 Writer input.

    Single narrow shape. ``__post_init__`` invariants mirror DB CHECK
    constraints so semantic errors surface before hitting the database.
    """

    user_id: str
    session_id: Optional[str]
    tool_name: str
    tool_source: ToolSource
    arg_digest: str
    primary_arg: str
    dir_arg: str
    scope: Scope
    effect: Effect
    source_type: SourceType
    confirmation_id: Optional[str]
    expires_at: Optional[datetime]
    risk_level: str

    def __post_init__(self) -> None:
        if self.scope == "always" and (
            self.session_id is not None or self.expires_at is not None
        ):
            raise ValueError(
                "always scope requires session_id=None and expires_at=None"
            )
        if self.scope == "session" and (
            self.session_id is None or self.expires_at is None
        ):
            raise ValueError(
                "session scope requires session_id and expires_at to be set"
            )
