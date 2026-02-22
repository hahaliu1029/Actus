"""用户工具偏好领域模型"""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """工具类型枚举"""

    MCP = "mcp"
    A2A = "a2a"
    SKILL = "skill"


class UserToolPreference(BaseModel):
    """用户工具偏好领域模型

    记录每个用户对 MCP/A2A/Skill 工具的个人启用/禁用偏好
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    tool_type: ToolType
    tool_id: str  # server_name / a2a_id / skill_id
    enabled: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
