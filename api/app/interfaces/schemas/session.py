from datetime import datetime
from typing import List, Literal, Optional

from app.domain.models.file import File
from app.domain.models.message import SkillConfirmationAction
from app.domain.models.session import SessionStatus
from app.interfaces.schemas.event import AgentSSEEvent
from pydantic import BaseModel, ConfigDict, Field


class CreateSessionResponse(BaseModel):
    """创建会话响应结构"""

    session_id: str  # 会话id


class SupervisorSnapshot(BaseModel):
    """B3-core PR-1 §3.3 — supervisor snapshot for resume responses.

    PR-1 shipped the response shape; PR-4 wires producer-side snapshots from
    ``agent_service.get_events_since``.
    """

    execution_mode: Literal["foreground", "background"]
    execution_phase: Literal[
        "running", "recovering", "idle", "suspended", "terminating", "terminated"
    ]
    background_reason: Optional[Literal["explicit", "auto_degrade"]] = None
    expires_at: Optional[datetime] = None
    retry_budget_remaining: int
    suspended_reason: Optional[str] = None
    terminal_reason: Optional[str] = None
    last_progress_at: Optional[datetime] = None
    is_alive: bool
    cancellation_state: Literal["none", "cancelling", "cancelled"] = "none"


class ListSessionItem(BaseModel):
    """会话列表条目基础信息"""

    session_id: str = ""
    title: str = ""
    sample_session_id: Optional[str] = None
    latest_message: str = ""
    latest_message_at: Optional[datetime] = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.PENDING
    unread_message_count: int = 0
    supervisor_snapshot: Optional[SupervisorSnapshot] = None


class ListSessionResponse(BaseModel):
    """获取会话列表基础信息响应结构"""

    sessions: List[ListSessionItem]


class BackgroundQuotaResponse(BaseModel):
    """后台任务额度读模型"""

    system_used: int
    system_limit: int
    user_used: int
    user_limit: int


class ToolConfirmationAction(BaseModel):
    """危险工具确认请求"""

    action: Literal["approve", "deny"]
    scope: Literal["once", "session", "always"]
    tool_call_id: str


class ChatRequest(BaseModel):
    """聊天请求结构"""

    message: Optional[str] = None  # 人类消息
    attachments: Optional[List[str]] = Field(
        default_factory=list
    )  # 附件列表(传递的是文件id列表)
    skill_confirmation_action: Optional[SkillConfirmationAction] = None
    tool_confirmation: Optional[ToolConfirmationAction] = None
    event_id: Optional[str] = None  # 最新事件id
    timestamp: Optional[int] = None  # 当前时间戳


class CancelSessionRequest(BaseModel):
    """取消会话请求结构"""

    model_config = ConfigDict(extra="forbid")

    reason: Literal["user_cancel"] = "user_cancel"


class GetSessionResponse(BaseModel):
    """获取会话详情响应结构"""

    session_id: str
    title: Optional[str] = None
    status: SessionStatus
    events: List[AgentSSEEvent] = Field(default_factory=list)
    supervisor_snapshot: Optional[SupervisorSnapshot] = None


class EventsSinceResponse(BaseModel):
    """增量事件恢复响应"""

    events: List[AgentSSEEvent] = Field(default_factory=list)
    session_status: SessionStatus  # KEEP — frontend at session-store.ts:888 reads this
    has_more: bool = False
    # B3-core PR-1 §3.3 additions:
    last_seq: int = 0
    supervisor_snapshot: Optional[SupervisorSnapshot] = None


class GetSessionFilesResponse(BaseModel):
    """获取会话文件列表响应结构"""

    files: List[File] = Field(default_factory=list)


class FileReadRequest(BaseModel):
    """需要读取的沙箱文件请求结构"""

    filepath: str


class FileReadResponse(BaseModel):
    """需要读取的沙箱文件响应结构体"""

    filepath: str
    content: str


class ShellReadRequest(BaseModel):
    """需要读取的沙箱shell请求结构体"""

    session_id: str  # Shell会话id


class ConsoleRecord(BaseModel):
    """控制台记录模型，包含ps1、command、output"""

    ps1: str
    command: str
    output: str


class ShellReadResponse(BaseModel):
    """需要读取的沙箱shell响应结构体"""

    session_id: str
    output: str
    console_records: List[ConsoleRecord] = Field(default_factory=list)


class GetTakeoverResponse(BaseModel):
    """获取会话接管状态响应结构"""

    status: SessionStatus
    takeover_id: Optional[str] = None
    request_status: Optional[str] = None
    reason: Optional[str] = None
    scope: Optional[str] = None
    handoff_mode: Optional[str] = None
    expires_at: Optional[int] = None


class StartTakeoverRequest(BaseModel):
    """启动接管请求结构"""

    scope: Literal["shell", "browser"] = "shell"


class StartTakeoverResponse(BaseModel):
    """启动接管响应结构"""

    status: SessionStatus
    request_status: str
    scope: str
    takeover_id: Optional[str] = None
    reason: Optional[str] = None
    expires_at: Optional[int] = None


class RejectTakeoverRequest(BaseModel):
    """拒绝接管请求结构"""

    decision: Literal["continue", "terminate"]


class RejectTakeoverResponse(BaseModel):
    """拒绝接管响应结构"""

    status: SessionStatus
    reason: str


class EndTakeoverRequest(BaseModel):
    """结束接管请求结构"""

    handoff_mode: Literal["continue", "complete"] = "continue"


class EndTakeoverResponse(BaseModel):
    """结束接管响应结构"""

    status: SessionStatus
    handoff_mode: str


class ReopenTakeoverResponse(BaseModel):
    """补救接管响应结构"""

    status: SessionStatus
    request_status: str
    reason: Optional[str] = None
    remaining_seconds: Optional[float] = None


class RetryFromSuspendResponse(BaseModel):
    """后台挂起任务重试响应结构"""

    status: SessionStatus
    request_status: Literal["resumed"]
    retry_budget_remaining: int
    expires_at: Optional[int] = None


class RenewTakeoverRequest(BaseModel):
    """续期接管请求结构"""

    takeover_id: str


class RenewTakeoverResponse(BaseModel):
    """续期接管响应结构"""

    status: SessionStatus
    request_status: str
    takeover_id: str
    expires_at: Optional[int] = None
