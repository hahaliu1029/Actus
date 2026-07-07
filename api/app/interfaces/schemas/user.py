"""用户相关 Schema"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.domain.models.user_tool_approval_policy import ApprovalPolicy


class UserStatusUpdateRequest(BaseModel):
    """更新用户状态请求（管理员）"""

    status: str = Field(..., description="用户状态: active, inactive, banned")


class UserListResponse(BaseModel):
    """用户列表响应"""

    users: list = Field(default_factory=list, description="用户列表")
    total: int = Field(default=0, description="总数")


class ToolPreferenceRequest(BaseModel):
    """工具偏好请求"""

    enabled: bool = Field(..., description="是否启用")


class ToolWithPreference(BaseModel):
    """带用户偏好的工具信息"""

    tool_id: str = Field(..., description="工具 ID")
    tool_name: str = Field(..., description="工具名称")
    description: Optional[str] = Field(None, description="工具描述")
    enabled_global: bool = Field(..., description="全局启用状态")
    enabled_user: bool = Field(..., description="用户个人启用状态")
    slug: Optional[str] = Field(
        None, description="Skill slug（仅 skill 分支填充，MCP/A2A 为 None）"
    )


class MCPToolListResponse(BaseModel):
    """MCP 工具列表响应（带用户偏好）"""

    tools: list[ToolWithPreference] = Field(default_factory=list)


class A2AToolListResponse(BaseModel):
    """A2A 工具列表响应（带用户偏好）"""

    tools: list[ToolWithPreference] = Field(default_factory=list)


class ToolPolicyRequest(BaseModel):
    """工具审批 policy 请求体。"""

    policy: ApprovalPolicy = Field(..., description="审批策略: auto / ask / deny")


class ToolPolicyResponse(BaseModel):
    """工具审批 policy 响应体。"""

    tool_name: str = Field(..., description="工具 canonical 名")
    policy: ApprovalPolicy = Field(..., description="审批策略")
    updated_at: datetime = Field(..., description="最后更新时间")
