import uuid
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

from .file import File
from .mailbox_envelope import CostAggregate, ResultReadyOutcome
from .message import SkillConfirmationAction
from .patch_apply_plan import GroupOutcome
from .plan import Plan, Step
from .search import SearchResultItem
from .tool_result import ToolResult
from app.domain.services.tools.tool_source_resolver import ToolSource


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
    seq: Optional[int] = None  # B3-core PR-1 §3.3: producer-side monotonic stamp via INCR session:seq:{sid}


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
    skill_confirmation_action: SkillConfirmationAction | None = None


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
    """工具事件 (R4 extended: internal artifact as dict + external wire metadata)."""

    type: Literal["tool"] = "tool"

    # --- 现有字段 (保留, pre-R4 兼容) ---
    tool_call_id: str
    tool_name: str                            # canonical category, R1 写入
    tool_content: Optional[ToolContent] = None
    function_name: str
    function_args: Dict[str, Any]
    function_result: Optional[ToolResult] = None    # R2 legacy shape, soft-coexistence
    status: ToolEventStatus = ToolEventStatus.CALLING

    # --- R4 新增 (all Optional, default=None/"", pre-R4 JSON 仍可 validate) ---
    # F2 fix: artifact 是 dict (R2 JSON serialized form), 不是 typed ToolArtifact.
    # TypeAdapter(Event).validate_python 在事件日志回放路径不会触发
    # ToolOutcome discriminated union dispatch, 未来 R2 加新 variant 不会在
    # 反序列化层硬抛 ValidationError. Projector 用 TOOL_ARTIFACT_ADAPTER.validate_python
    # 懒校验 + try/except 兜底消费.
    artifact: Optional[Dict[str, Any]] = None        # R2 ToolArtifact.model_dump(mode='json') output, INTERNAL ONLY
    tool_source: Optional[ToolSource] = None         # CS1, 从 artifact.tool_source 冗余上来
    activity_description: str = ""                   # B10 驱动
    display_icon: Optional[str] = None               # B10 驱动
    render_style: Optional[
        Literal["text", "code", "table", "image", "document"]
    ] = None                                         # B12 驱动
    media_type: Optional[str] = None                 # B12 驱动


class WaitEvent(BaseEvent):
    """等待事件，等待用户输入确认"""

    type: Literal["wait"] = "wait"
    pending_action: Literal["generate", "install"] | None = None


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


class ContextStatusEvent(BaseEvent):
    """上下文水位状态事件"""

    type: Literal["context_status"] = "context_status"
    used_tokens: int = 0
    context_window: int = 0
    usage_ratio: float = 0.0
    soft_threshold: float = 0.85
    hard_threshold: float = 0.95


class CompactionEvent(BaseEvent):
    """上下文压缩事件"""

    type: Literal["compaction"] = "compaction"
    level: int = 0
    tokens_before: int = 0
    tokens_after: int = 0
    messages_removed: int = 0
    usage_ratio_after: float = 0.0
    compaction_id: str | None = None  # B6: pointer to conversation_compactions row; None for legacy/Path-B events


class FinishingEvent(BaseEvent):
    """主回复完成，进入后台收尾阶段。前端收到后解锁输入框。"""

    type: Literal["finishing"] = "finishing"


class DoneEvent(BaseEvent):
    """结束事件类型"""

    type: Literal["done"] = "done"
    metrics: Optional[Dict[str, Any]] = None  # D5: execution metrics snapshot


class HealthStatus(str, Enum):
    """执行健康状态"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"          # idle timeout 触发，尝试恢复中
    TERMINATING = "terminating"    # 总超时或恢复失败，正在终止
    TERMINATED = "terminated"      # 已强制终止


class HealthEvent(BaseEvent):
    """执行健康状态事件"""

    type: Literal["health"] = "health"
    status: HealthStatus
    reason: str                         # 用户友好原因
    last_node: Optional[str] = None     # 最后活跃 graph node
    idle_seconds: Optional[float] = None
    tool_failures: int = 0
    action: str = "monitoring"          # monitoring/soft_recovery/hard_terminate
    metrics: Optional[Dict[str, Any]] = None


class ToolConfirmationEvent(BaseEvent):
    """危险工具确认请求事件"""

    type: Literal["tool_confirmation"] = "tool_confirmation"
    tool_call_id: str
    tool_name: str
    tool_args: Dict[str, Any]
    risk_level: str
    risk_reason: str
    matched_patterns: List[str]
    suggested_alternative: Optional[str] = None
    approval_options: List[str] = Field(default=["once", "session", "always", "deny"])
    timeout_seconds: int


class SandboxStateChangedEvent(BaseEvent):
    """Sandbox 绑定状态变更事件（PR2 §10.1）"""

    type: Literal["sandbox_state_changed"] = "sandbox_state_changed"
    old_state: str  # SandboxBindingState.value
    new_state: str  # SandboxBindingState.value
    generation: int
    sandbox_id: Optional[str] = None
    reason: Optional[str] = None  # DestroyReason.value or free-text


# A4-0: control-mode subset of SessionStatus values. Typed as str/Literal (NOT
# SessionStatus) to avoid an import cycle — session.py imports event.py.
ModeLiteral = Literal["running", "waiting", "takeover_pending", "takeover"]


class SessionModeChangedEvent(BaseEvent):
    """A4-0: unified control-mode-changed signal. Additive — emitted alongside
    WaitEvent/ControlEvent. ``to`` is authoritative; ``from_mode``/``mode_revision``
    are best-effort context for idempotent client reconciliation (LWW by
    ``mode_revision``)."""

    type: Literal["session_mode_changed"] = "session_mode_changed"
    to: ModeLiteral  # new control mode (authoritative), str value
    from_mode: Optional[ModeLiteral] = None  # prior mode if cheaply known
    reason: str  # server-fixed constant per emit site (INV-6)
    mode_revision: Optional[int] = None  # captured in the status-write txn


class ExecutionStatePayload(BaseModel):
    """B3-core supervisor execution state snapshot (spec v3 §3.3)."""

    execution_mode: Literal["foreground", "background"]
    execution_phase: Literal[
        "running", "recovering", "idle", "suspended", "terminating", "terminated"
    ]
    background_reason: Optional[Literal["explicit", "auto_degrade"]] = None
    expires_at: Optional[datetime] = None
    retry_budget_remaining: int
    suspended_reason: Optional[
        Literal["bg_idle_timeout", "server_restart"]
    ] = None
    terminal_reason: Optional[
        Literal[
            "natural",
            "user_cancel",
            "server_restart",
            "resume_state_lost",
            "watchdog_timeout",
        ]
    ] = None
    transition_reason: str = ""


class ExecutionStateChangedEvent(BaseEvent):
    """B3-core supervisor: execution mode/phase transition (spec v3 §3.3, T1-T11)."""

    type: Literal["execution_state_changed"] = "execution_state_changed"
    payload: ExecutionStatePayload


class OwnerConflictPayload(BaseModel):
    """B3-core supervisor multi-tab CAS lease conflict (spec v3 §3.3, §4.7)."""

    current_owner_connection_id: str
    conflicting_connection_id: str
    session_id: str
    suggested_action: Literal["wait_lease_expire", "request_takeover"] = "wait_lease_expire"


class OwnerConflictEvent(BaseEvent):
    """B3-core supervisor: Tab2 attempted to claim CAS lease but Tab1 holds it."""

    type: Literal["owner_conflict"] = "owner_conflict"
    payload: OwnerConflictPayload


# ---------------------------------------------------------------------------
# C2 PR-8 §13 — Coordinator SSE events
#
# Five new events surface the coordinator/child lineage to the frontend so a
# parent session can render the fan-out timeline. ``CoordinatorLineageMixin``
# carries the optional lineage tag fields; every coordinator event composes it
# alongside :class:`BaseEvent`. Lineage fields are all ``Optional[str] = None``
# so an event emitted from the root session still validates.
# ---------------------------------------------------------------------------


class CoordinatorLineageMixin(BaseModel):
    """[C2 PR-8 §13.2] Optional lineage tagging.

    All fields default to ``None`` so the mixin is safe to compose into events
    emitted from the root session (no coordinator context yet).
    """

    root_session_id: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    work_unit_id: Optional[str] = None


class CoordinatorDispatchEvent(BaseEvent, CoordinatorLineageMixin):
    """[C2 PR-8 §13.3] Coordinator about to spawn ``work_unit_count`` children."""

    type: Literal["coordinator_dispatch"] = "coordinator_dispatch"
    step_id: str
    work_unit_count: int
    work_unit_ids: List[str]
    phases: List[Literal["exploration", "write"]]


class CoordinatorWorkerSpawnedEvent(BaseEvent, CoordinatorLineageMixin):
    """[C2 PR-8 §13.3] A child session was spawned for a work unit."""

    type: Literal["coordinator_worker_spawned"] = "coordinator_worker_spawned"
    objective: str
    phase: Literal["exploration", "write"]
    allowed_tools: List[str]
    write_lease_count: int


class CoordinatorReduceEvent(BaseEvent, CoordinatorLineageMixin):
    """[C2 PR-8 §13.3] Reducer emitted the group-level outcome."""

    type: Literal["coordinator_reduce"] = "coordinator_reduce"
    group_outcome: GroupOutcome
    per_worker_outcomes: Dict[str, ResultReadyOutcome]
    diagnostics_summary: str
    conflict_paths: List[str] = Field(default_factory=list)
    cost_total: CostAggregate


class CoordinatorApplyEvent(BaseEvent, CoordinatorLineageMixin):
    """[C2 PR-8 §13.3] Patch-apply phase progress."""

    type: Literal["coordinator_apply"] = "coordinator_apply"
    apply_status: str  # ApplyStatus value
    file_count: int = 0
    total_bytes: int = 0
    failed_at_path: Optional[str] = None
    rollback_status: Optional[str] = None


class CoordinatorSiblingCancelEvent(BaseEvent, CoordinatorLineageMixin):
    """[C2 PR-8 §13.3] Fail-fast / authorization-gate sibling cancellation."""

    type: Literal["coordinator_sibling_cancel"] = "coordinator_sibling_cancel"
    triggered_by_work_unit_id: str
    triggered_by_outcome: ResultReadyOutcome
    cancelled_work_unit_ids: List[str]
    reason: str


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
        ContextStatusEvent,
        CompactionEvent,
        FinishingEvent,
        HealthEvent,
        ToolConfirmationEvent,
        SandboxStateChangedEvent,
        SessionModeChangedEvent,  # A4-0
        ExecutionStateChangedEvent,  # B3-core PR-1
        OwnerConflictEvent,           # B3-core PR-1
        CoordinatorDispatchEvent,     # C2 PR-8
        CoordinatorWorkerSpawnedEvent,  # C2 PR-8
        CoordinatorReduceEvent,       # C2 PR-8
        CoordinatorApplyEvent,        # C2 PR-8
        CoordinatorSiblingCancelEvent,  # C2 PR-8
        DoneEvent,
    ],
    Field(discriminator="type"),
]
