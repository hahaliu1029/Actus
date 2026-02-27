from datetime import datetime
from typing import List, Literal, Optional

from app.domain.models.file import File
from app.domain.models.session import SessionStatus
from app.interfaces.schemas.event import AgentSSEEvent
from pydantic import BaseModel, Field


class CreateSessionResponse(BaseModel):
    """创建会话响应结构"""

    session_id: str  # 会话id


class ListSessionItem(BaseModel):
    """会话列表条目基础信息"""

    session_id: str = ""
    title: str = ""
    latest_message: str = ""
    latest_message_at: Optional[datetime] = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.PENDING
    unread_message_count: int = 0


class ListSessionResponse(BaseModel):
    """获取会话列表基础信息响应结构"""

    sessions: List[ListSessionItem]


class ChatRequest(BaseModel):
    """聊天请求结构"""

    message: Optional[str] = None  # 人类消息
    attachments: Optional[List[str]] = Field(
        default_factory=list
    )  # 附件列表(传递的是文件id列表)
    event_id: Optional[str] = None  # 最新事件id
    timestamp: Optional[int] = None  # 当前时间戳


class GetSessionResponse(BaseModel):
    """获取会话详情响应结构"""

    session_id: str
    title: Optional[str] = None
    status: SessionStatus
    events: List[AgentSSEEvent] = Field(default_factory=list)


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


class RejectTakeoverRequest(BaseModel):
    """拒绝接管请求结构"""

    decision: Literal["continue", "terminate"]


class RejectTakeoverResponse(BaseModel):
    """拒绝接管响应结构"""

    status: SessionStatus
    reason: str


class EndTakeoverRequest(BaseModel):
    """结束接管请求结构"""

    handoff_mode: Literal["continue", "complete"] = "complete"


class EndTakeoverResponse(BaseModel):
    """结束接管响应结构"""

    status: SessionStatus
    handoff_mode: str


class RenewTakeoverRequest(BaseModel):
    """续期接管请求结构"""

    takeover_id: str


class RenewTakeoverResponse(BaseModel):
    """续期接管响应结构"""

    status: SessionStatus
    request_status: str
    takeover_id: str
