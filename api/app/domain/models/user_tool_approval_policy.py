"""用户工具审批偏好领域模型。

显式覆盖表：缺行 = "用户未设置偏好"，由 Phase 2 PermissionEngine
回落到默认决策链，不代表任何持久化默认值。tool_name 覆盖全量工具
（native / MCP / Skill / A2A），按 canonical runtime name 存储。
"""

from datetime import datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ApprovalPolicy(str, Enum):
    """用户对单个工具调用的审批偏好。"""

    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


class UserToolApprovalPolicy(BaseModel):
    """用户对某个工具调用的显式审批偏好。"""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    # tool_name: canonical runtime tool name（无统一前缀规则，按真实注册名存）。
    # 样例：
    #   native:        shell_execute / memory_search / search_web
    #   a2a:           get_remote_agent_cards / call_remote_agent
    #   mcp discovery: list_mcp_tools / get_mcp_tool
    #   mcp dynamic:   mcp_{server}_{tool}   (单下划线；server 可含连字符)
    #   skill static:  brainstorm_skill / generate_skill / install_skill
    #   skill dynamic: skill_{slug}_{tool}
    tool_name: str
    policy: ApprovalPolicy
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True
