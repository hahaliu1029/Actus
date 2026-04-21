"""用户工具扩展启用领域模型。"""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ToolType(str, Enum):
    """工具类型枚举"""

    MCP = "mcp"
    A2A = "a2a"
    SKILL = "skill"


class UserToolEnablement(BaseModel):
    """用户对扩展工具 (MCP/A2A/Skill) 的启用/禁用偏好。

    仅管"这个扩展是否进工具池"，不管审批。审批走 UserToolApprovalPolicy。
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
