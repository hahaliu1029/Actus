from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class SkillCreationState(BaseModel):
    """Skill 创建链路的持久化等待状态。"""

    pending_action: Literal["generate", "install"] | None = None
    approval_status: Literal["pending"] | None = None
    last_tool_name: Literal["brainstorm_skill", "generate_skill", "install_skill"] | None = None
    last_tool_call_id: str = ""
    saved_tool_result_json: str = ""
    blueprint: dict[str, Any] | None = None
    blueprint_json: str = ""
    skill_data: str = ""
    requested_at: datetime = Field(default_factory=datetime.now)
