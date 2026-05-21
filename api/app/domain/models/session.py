import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .event import Event, PlanEvent
from .file import File
from .memory import Memory
from .plan import Plan


class SessionStatus(str, Enum):
    """会话状态类型枚举"""

    PENDING = "pending"  # 等待任务
    RUNNING = "running"  # 运行中
    TAKEOVER_PENDING = "takeover_pending"  # 请求接管待用户决策
    TAKEOVER = "takeover"  # 用户接管中
    WAITING = "waiting"  # 等待人类响应
    FINISHING = "finishing"  # 后处理中（主回复已完成）
    COMPLETED = "completed"  # 已完成
    TIMED_OUT = "timed_out"  # watchdog 超时终止


# ── Sandbox Binding (K8s-style terminal-state-aware lifecycle) ────────── #


class SandboxBindingState(str, Enum):
    """沙箱绑定状态。

    状态机见 docs/superpowers/specs/2026-04-15-sandbox-lifecycle-design.md §6。
    DESTROYED 是 terminal immutable（I1）。
    """

    UNBOUND = "unbound"  # 未绑定沙箱
    CREATING = "creating"  # 沙箱创建中
    ACTIVE = "active"  # 活跃
    SUSPENDED = "suspended"  # 已挂起（容器不销毁，可 resume）
    DESTROYING = "destroying"  # 销毁中（两阶段 quiesce barrier）
    DESTROYED = "destroyed"  # 已销毁（terminal，不可逆）


class DestroyReason(str, Enum):
    """沙箱销毁原因。"""

    SESSION_DELETE = "session_delete"  # 用户主动删除会话
    WATCHDOG_TIMEOUT = "watchdog_timeout"  # 超时销毁
    RECONCILE_ORPHAN = "reconcile_orphan"  # 容器被外部 kill，reconcile 标记
    # C3 PR-1 — mailbox-driven terminal transitions (spec §7.2)
    SUBAGENT_TERMINAL_RESULT = "subagent_terminal_result"  # RESULT_READY observed
    CANCEL_ACK_OBSERVED = "cancel_ack_observed"  # child confirmed cooperative cancel
    ORPHAN_TIMEOUT = "orphan_timeout"  # supervisor stale detection
    FORCE_TERMINATE = "force_terminate"  # cascade cancel policy=TERMINATE


class SandboxBinding(BaseModel):
    """Terminal-state-aware sandbox binding.

    frozen 保证任何修改必须经由 SandboxLifecycleService 产生新实例
    （service 单写者，I3）。
    """

    model_config = ConfigDict(frozen=True)

    id: Optional[str] = None
    state: SandboxBindingState = SandboxBindingState.UNBOUND
    generation: int = 0
    created_at: Optional[datetime] = None
    destroyed_at: Optional[datetime] = None
    destroy_reason: Optional[DestroyReason] = None

    def is_terminal(self) -> bool:
        return self.state == SandboxBindingState.DESTROYED

    def is_suspended(self) -> bool:
        return self.state == SandboxBindingState.SUSPENDED

    def can_acquire(self) -> bool:
        return self.state == SandboxBindingState.ACTIVE


class Session(BaseModel):
    """会话领域模型"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))  # 会话id
    parent_session_id: Optional[str] = None  # C1a canonical lineage field — sole survivor after PR-4 contract drop.
    worker_type: Literal["root", "subagent"] = "root"  # C1a identity axis; CHECK ck_sessions_worker_type_parent_invariant keeps this in sync with parent_session_id.
    # C3 PR-1 — control plane discriminator (spec §11.2). 'legacy' = SSE-only
    # path; 'mailbox' = MailboxSupervisor manages lifecycle; None = pre-C3 rows
    # (treat as 'legacy' via consumer-side coalesce). Narrowed to a Literal
    # in codex round-11 P2 review so typo'd values are caught at the domain
    # boundary and never reach the DB CHECK constraint
    # (ck_sessions_subagent_control_plane_valid).
    subagent_control_plane: Optional[Literal["legacy", "mailbox"]] = None
    tool_filter_preset: Optional[Literal["subagent_research"]] = (
        None  # T12 pod-restart resilience：持久化 tool_filter 预设名；NULL = 不限制
    )
    sandbox_id: Optional[str] = None  # 沙箱id（仅 infrastructure ORM 兼容层使用）
    sandbox_binding: SandboxBinding = Field(
        default_factory=SandboxBinding
    )  # 沙箱绑定（领域层唯一访问点，I8）
    task_id: Optional[str] = None  # 任务id
    title: str = ""  # 标题
    unread_message_count: int = 0  # 未读消息数
    latest_message: str = ""  # 最新消息
    latest_message_at: Optional[datetime] = None  # 最新消息时间
    events: List[Event] = Field(default_factory=list)  # 事件列表
    files: List[File] = Field(default_factory=list)  # 文件列表
    memories: Dict[str, Memory] = Field(default_factory=dict)  # 记忆
    status: SessionStatus = SessionStatus.PENDING  # 状态
    user_id: Optional[str] = None  # 会话所属用户ID
    completed_at: Optional[datetime] = None  # 完成时间
    execution_mode: Literal["foreground", "background"] = "foreground"
    background_reason: Literal["explicit", "auto_degrade"] | None = None
    expires_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    execution_phase: Literal[
        "running", "recovering", "idle", "suspended", "terminating", "terminated"
    ] = "running"
    retry_budget_remaining: int = 3
    terminal_reason: (
        Literal[
            "natural",
            "user_cancel",
            "server_restart",
            "resume_state_lost",
            "watchdog_timeout",
        ]
        | None
    ) = None
    suspended_reason: (
        Literal["bg_idle_timeout", "server_restart"] | None
    ) = None
    was_background: bool = False
    updated_at: datetime = Field(default_factory=datetime.now)  # 更新时间
    created_at: datetime = Field(default_factory=datetime.now)  # 创建时间

    def get_latest_plan(self) -> Optional[Plan]:
        """获取会话中的最新计划"""
        # 1.倒序遍历会话中所有事件消息
        for event in reversed(self.events):
            # 2.判断事件的类型是否为PlanEvent，如果是则提取计划后返回
            if isinstance(event, PlanEvent):
                return event.plan

        return None
