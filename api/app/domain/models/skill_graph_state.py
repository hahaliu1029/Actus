from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SkillGraphStatus = Literal[
    "wait_generate",
    "generating",
    "wait_install",
    "installing",
    "done",
    "cancelled",
    "error",
]

SKILL_GRAPH_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"done", "cancelled", "error"}
)


class SkillGraphState(BaseModel):
    """Skill 创建子图的持久化状态模型（§3.1）。

    存储在 session.memories["_skill_graph"]，子图激活时为单一真源。
    """

    model_config = ConfigDict(extra="ignore")

    status: SkillGraphStatus = "wait_generate"
    pending_action: Literal["generate", "install"] | None = None
    approval_status: Literal["pending", "approved"] | None = None
    original_request: str = ""
    blueprint: dict[str, Any] | None = None
    blueprint_json: str = ""
    skill_data: str = ""
    last_tool_call_id: str = ""
    saved_tool_result_json: str = ""
    last_error: str = ""
    retry_count: int = 0
    updated_at: datetime = Field(default_factory=datetime.now)

    # -- 辅助属性 --

    @property
    def is_terminal(self) -> bool:
        return self.status in SKILL_GRAPH_TERMINAL_STATUSES

    @property
    def is_executing(self) -> bool:
        """是否处于瞬态执行中（防重入标记）。"""
        return self.status in {"generating", "installing"}
