"""C4 通用 SubagentWorker 抽象 — DTO 契约（spec §3）。

domain 约束：仅依赖 pydantic + stdlib + 同层 domain models（mailbox_envelope）。
禁止 Web 框架与 ORM 依赖（由 PR-1 `test_c4_domain_file_is_pure` AST + 子串守门；
故此处刻意不写出被禁框架名的字面串，以免触发自身子串断言）。

最大局限（spec §11）：本契约是「投影 + 一个适配器 + 一致性证明」，不是 live
统一。三种 runtime（LOCAL in-process child / REMOTE A2A / SKILL 保留）仍各自
执行；没有生产代码调用 SubagentWorker。它们 **不** 共享权限 / lifecycle /
cancel / cost / artifacts / session 语义——不要据此 DTO 误判 false-unification。
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.models.mailbox_envelope import ArtifactRef, CostAggregate


class WorkerRuntimeType(str, Enum):
    """执行底座轴。"""

    LOCAL = "local"     # in-process subagent（C2 coordinator child + research 子）— 投影,不重接线
    REMOTE = "remote"   # A2A 远程 agent — 唯一落地适配器
    SKILL = "skill"     # 保留值；C4 无适配器（NON-GOAL）


class WorkerLifecycleState(str, Enum):
    PENDING = "pending"               # 保留；C4 无 producer（INV-C4-4）
    RUNNING = "running"               # 保留；C4 无 producer（INV-C4-4）
    WAITING_INPUT = "waiting_input"   # 非终态阻塞（research WAITING / A2A input-required）
    TERMINAL = "terminal"


class WorkerTerminalOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NEEDS_AUTHORIZATION = "needs_authorization"
    UNKNOWN = "unknown"               # A2A 无法分类的终态（§5 解析契约产出）


class WorkerSpec(BaseModel):
    """统一任务输入（spec §3.3）— 诚实、最小。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_runtime_type: WorkerRuntimeType
    parent_session_id: str
    child_session_id: Optional[str] = None             # A2A 无 child session ⇒ optional
    objective: str                                     # 任务 / query / prompt
    # 不透明权限描述符；C4 reserved，恒 None（C4 无 WorkerSpec 生产者）。
    # 不加 validator 强制 None，以便 C4.1 启用。C4 不做任何权限「继承」语义。
    permission_scope_descriptor: Optional[str] = None
    remote_target: Optional[str] = None                # A2A agent id（仅 REMOTE）
    expected_result_schema: Optional[str] = None

    @model_validator(mode="after")
    def _validate_runtime_field_matrix(self) -> "WorkerSpec":
        """INV-C4-2（spec §3.3）：

        | runtime | remote_target | child_session_id |
        |---------|---------------|------------------|
        | LOCAL   | 必须 None     | 必填             |
        | REMOTE  | 必填          | optional         |
        | SKILL   | 必须 None     | optional         |
        """
        rt = self.worker_runtime_type
        if rt == WorkerRuntimeType.REMOTE:
            if self.remote_target is None:
                raise ValueError("REMOTE worker requires remote_target")
        else:  # LOCAL or SKILL
            if self.remote_target is not None:
                raise ValueError(
                    f"{rt.value} worker forbids remote_target "
                    "(only REMOTE may carry one)"
                )
        if rt == WorkerRuntimeType.LOCAL and self.child_session_id is None:
            raise ValueError("LOCAL worker requires child_session_id")
        return self


class SubagentRunResult(BaseModel):
    """统一结果（spec §3.4）— 无损 + 诚实未知。

    复用 mailbox_envelope.ArtifactRef / CostAggregate（Actus 内部一致性）。
    cost 未知 ≠ 0：A2A cost_summary=None, cost_authoritative=False。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_runtime_type: WorkerRuntimeType
    lifecycle_state: WorkerLifecycleState
    terminal_outcome: Optional[WorkerTerminalOutcome] = None
    summary: str = ""
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    cost_summary: Optional[CostAggregate] = None           # None = 未知
    cost_authoritative: bool = False                        # A2A cost = 未知,不是 0
    duration_seconds: Optional[float] = None
    duration_source: Literal["observed_local", "remote_reported", "unavailable"] = "unavailable"
    error_summary: Optional[str] = None
    parent_session_id: Optional[str] = None
    child_session_id: Optional[str] = None
    source_ref: Optional[str] = None    # 源标识（coordinator work_unit_id / research child_id）；仅追溯

    @model_validator(mode="after")
    def _validate_terminal_outcome_iff_terminal(self) -> "SubagentRunResult":
        """INV-C4-1 / INV-C4-3（spec §3.2/§3.4）。"""
        is_terminal = self.lifecycle_state == WorkerLifecycleState.TERMINAL
        has_outcome = self.terminal_outcome is not None
        if is_terminal and not has_outcome:
            raise ValueError("lifecycle_state=TERMINAL requires terminal_outcome")
        if has_outcome and not is_terminal:
            raise ValueError(
                "terminal_outcome may only be set when lifecycle_state=TERMINAL"
            )
        return self
