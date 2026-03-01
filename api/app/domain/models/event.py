import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .file import File
from .plan import Plan, Step
from .search import SearchResultItem
from .tool_result import ToolResult


class PlanEventStatus(str, Enum):
    """规划事件状态"""

    CREATED = "created"  # 已创建
    UPDATED = "updated"  # 已更新
    COMPLETED = "completed"  # 已完成


class StepEventStatus(str, Enum):
    """步骤事件状态"""

    STARTED = "started"  # 已开始
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"  # 失败


class ToolEventStatus(str, Enum):
    """工具事件状态类型枚举"""

    CALLING = "calling"  # 调用中
    CALLED = "called"  # 调用完毕


class ControlAction(str, Enum):
    """接管控制事件动作"""

    REQUESTED = "requested"
    STARTED = "started"
    REJECTED = "rejected"
    RENEWED = "renewed"
    EXPIRED = "expired"
    ENDED = "ended"
    REOPENED = "reopened"


class ControlScope(str, Enum):
    """接管范围"""

    SHELL = "shell"
    BROWSER = "browser"


class ControlSource(str, Enum):
    """接管事件来源"""

    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class BaseEvent(BaseModel):
    """基础事件类型"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 事件id
    type: Literal[""] = ""  # 事件的类型
    created_at: datetime = Field(default_factory=datetime.now)  # 事件创建时间


class PlanEvent(BaseEvent):
    """规划事件类型"""

    type: Literal["plan"] = "plan"
    plan: Plan  # 规划
    status: PlanEventStatus = PlanEventStatus.CREATED  # 规划事件状态


class TitleEvent(BaseEvent):
    """标题事件类型"""

    type: Literal["title"] = "title"
    title: str = ""  # 标题


class StepEvent(BaseEvent):
    """子任务/步骤事件"""

    type: Literal["step"] = "step"
    step: Step  # 步骤信息
    status: StepEventStatus = StepEventStatus.STARTED


class MessageEvent(BaseEvent):
    """消息事件，包含人类消息和AI消息"""

    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system"] = "assistant"  # 消息角色
    message: str = ""  # 消息本身
    stream_id: Optional[str] = None  # 同一条消息流式更新id
    partial: bool = False  # 是否为流式中间片段
    attachments: List[File] = Field(default_factory=list)  # 附件列表信息


class BrowserToolContent(BaseModel):
    """浏览器工具扩展内容"""

    screenshot: str  # 浏览器快照截图


class SearchToolContent(BaseModel):
    """搜索工具内容"""

    results: List[SearchResultItem]  # 搜索结果列表


class ShellToolContent(BaseModel):
    """Shell工具内容"""

    console: Any  # 控制台内容


class FileToolContent(BaseModel):
    """文件工具内容"""

    content: str  # 文件内容


class MCPToolContent(BaseModel):
    """MCP工具内容"""

    result: Any  # MCP工具结果


class A2AToolContent(BaseModel):
    """A2A智能体工具内容"""

    a2a_result: Any  # A2A智能体调用结果


class SkillToolContent(BaseModel):
    """Skill 工具内容"""

    skill_result: Any  # Skill 工具执行结果


ToolContent = Union[
    BrowserToolContent,
    SearchToolContent,
    ShellToolContent,
    FileToolContent,
    MCPToolContent,
    A2AToolContent,
    SkillToolContent,
]


class ToolEvent(BaseEvent):
    """工具事件"""

    type: Literal["tool"] = "tool"
    tool_call_id: str  # 工具调用id
    tool_name: str  # 工具箱/工具集的名字
    tool_content: Optional[ToolContent] = None  # 工具扩展内容
    function_name: str  # LLM调用函数/工具名字
    function_args: Dict[str, Any]  # LLM生成的工具调用参数
    function_result: Optional[ToolResult] = None  # 工具调用结果
    status: ToolEventStatus = ToolEventStatus.CALLING  # 工具事件状态


class WaitEvent(BaseEvent):
    """等待事件，等待用户输入确认"""

    type: Literal["wait"] = "wait"


class ControlEvent(BaseEvent):
    """接管控制事件"""

    type: Literal["control"] = "control"
    action: ControlAction
    scope: Optional[ControlScope] = None
    source: ControlSource = ControlSource.SYSTEM
    reason: Optional[str] = None
    handoff_mode: Optional[str] = None
    request_status: Optional[str] = None
    takeover_id: Optional[str] = None
    expires_at: Optional[datetime] = None

    @model_validator(mode="after")
    def _validate_requested_scope(self) -> "ControlEvent":
        if self.action == ControlAction.REQUESTED and self.scope is None:
            raise ValueError("control.requested 事件必须提供 scope")
        return self


class ErrorEvent(BaseEvent):
    """错误事件"""

    type: Literal["error"] = "error"
    error: str = ""  # 错误信息


class DoneEvent(BaseEvent):
    """结束事件类型"""

    type: Literal["done"] = "done"


# 定义应用事件类型声明
Event = Annotated[
    Union[
        PlanEvent,
        TitleEvent,
        StepEvent,
        MessageEvent,
        ToolEvent,
        WaitEvent,
        ControlEvent,
        ErrorEvent,
        DoneEvent,
    ],
    Field(discriminator="type"),
]
