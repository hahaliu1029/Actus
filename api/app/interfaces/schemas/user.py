"""用户相关 Schema"""

from typing import Optional

from pydantic import BaseModel, Field


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


class MCPToolListResponse(BaseModel):
    """MCP 工具列表响应（带用户偏好）"""

    tools: list[ToolWithPreference] = Field(default_factory=list)


class A2AToolListResponse(BaseModel):
    """A2A 工具列表响应（带用户偏好）"""

    tools: list[ToolWithPreference] = Field(default_factory=list)
