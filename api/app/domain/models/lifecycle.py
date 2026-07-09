"""C7 生命周期统一 — 封闭词表 + 派生表 + 受控 schema（spec §2/§3/§5/§9）。

Leaf module：仅依赖 pydantic/enum/typing/dataclasses。
禁止 import 本包其他事件模块——``LifecycleEvent`` 类本体定义在 ``event.py``
（那里 import 本模块），避免 event.py ↔ lifecycle.py 循环 import。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple

from pydantic import BaseModel, ConfigDict


class LifecycleType(str, Enum):
    """统一生命周期单元类型（spec §2）。"""

    TASK = "task"          # 顶层任务（session 语义面；root session only，spec §4.4）
    PLAN = "plan"          # 规划
    STEP = "step"          # 执行步骤
    TOOL = "tool"          # 工具调用
    SUBAGENT = "subagent"  # 子代理（coordinator child + research child；research 为 best-effort 管线见 spec §4.5，R15 用户裁决重纳入）


class LifecycleState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class LifecycleEventKind(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRIED = "retried"


TERMINAL_STATES: FrozenSet[LifecycleState] = frozenset(
    {LifecycleState.COMPLETED, LifecycleState.FAILED, LifecycleState.CANCELLED}
)


def is_terminal(state: LifecycleState) -> bool:
    """单一终态谓词（spec §2；呼应 CC isTerminalTaskStatus / openclaw sticky）。"""
    return state in TERMINAL_STATES


_K = LifecycleEventKind
_T = LifecycleType
_S = LifecycleState

# per-type 事件子集单一真相（spec §2；INV-C7-1）。
# PLAN 无 FAILED：全仓无 Plan.status=FAILED 写入点（R5#P1b，失败由 task.failed 覆盖）。
# STEP failed 经 COMPLETED+success=False 真实生产路径（R5#P1）。
# SUBAGENT 无 PENDING/PROGRESS：pending 相位 wire 不可观测、per-child progress 不存在（F14/F16）。
SUPPORTED_EVENTS: Dict[LifecycleType, FrozenSet[LifecycleEventKind]] = {
    _T.TASK: frozenset({_K.STARTED, _K.PROGRESS, _K.COMPLETED, _K.FAILED, _K.CANCELLED, _K.RETRIED}),
    _T.PLAN: frozenset({_K.STARTED, _K.PROGRESS, _K.COMPLETED}),
    _T.STEP: frozenset({_K.STARTED, _K.COMPLETED, _K.FAILED}),
    _T.TOOL: frozenset({_K.STARTED, _K.PROGRESS, _K.COMPLETED, _K.FAILED, _K.CANCELLED}),
    _T.SUBAGENT: frozenset({_K.STARTED, _K.COMPLETED, _K.FAILED, _K.CANCELLED}),
}

# state 派生表（R10#A4：state 是 (lifecycle_type, event) 的纯函数，调用方不得传入）。
# 与 spec §4 五张映射表逐行对齐；INV-C7-1 测试穷举锁定。
STATE_FOR: Dict[Tuple[LifecycleType, LifecycleEventKind], LifecycleState] = {
    (_T.PLAN, _K.STARTED): _S.PENDING,
    (_T.PLAN, _K.PROGRESS): _S.RUNNING,
    (_T.PLAN, _K.COMPLETED): _S.COMPLETED,
    (_T.STEP, _K.STARTED): _S.RUNNING,
    (_T.STEP, _K.COMPLETED): _S.COMPLETED,
    (_T.STEP, _K.FAILED): _S.FAILED,
    (_T.TOOL, _K.STARTED): _S.PENDING,
    (_T.TOOL, _K.PROGRESS): _S.RUNNING,
    (_T.TOOL, _K.COMPLETED): _S.COMPLETED,
    (_T.TOOL, _K.FAILED): _S.FAILED,
    (_T.TOOL, _K.CANCELLED): _S.CANCELLED,
    (_T.TASK, _K.STARTED): _S.RUNNING,
    (_T.TASK, _K.PROGRESS): _S.RUNNING,
    (_T.TASK, _K.COMPLETED): _S.COMPLETED,
    (_T.TASK, _K.FAILED): _S.FAILED,
    (_T.TASK, _K.CANCELLED): _S.CANCELLED,
    (_T.TASK, _K.RETRIED): _S.RUNNING,
    (_T.SUBAGENT, _K.STARTED): _S.RUNNING,
    (_T.SUBAGENT, _K.COMPLETED): _S.COMPLETED,
    (_T.SUBAGENT, _K.FAILED): _S.FAILED,
    (_T.SUBAGENT, _K.CANCELLED): _S.CANCELLED,
}

# reason 受控词表（spec §3.1/§4/§5；自由文本禁止——诊断细节留在 source 事件）。
# 其中 3 码为 spec 描述但未显式命名、plan 定名（R1 review P3-2 记录）：
#   plan_updated（§4.1「update 原因进 reason/detail」）/ runner_error（§4.4「受控
#   错误 code（非异常文本）」）/ sibling_cancel（§4.5 sibling-cancel 行）。
REASON_CODES: FrozenSet[str] = frozenset({
    "plan_updated", "step_failed",
    "tool_error", "tool_timeout", "denied", "passthrough",
    "finishing", "waiting_confirmation", "watchdog_timeout",
    "user_cancel", "retry_from_suspend", "runner_error", "postprocess_failed",
    "worker_failed", "needs_authorization", "unknown_terminal_outcome",
    "waiting_unsupported", "sibling_cancel",
})

# R10#A3 — epoch 持久化来源：epoch = RETRY_BUDGET_INITIAL - retry_budget_remaining。
# 必须与 Session.retry_budget_remaining 字段默认值一致（session.py:146），
# 由 test_lifecycle_contract.py 锁定。禁止内存计数（pod 重启归零 →
# epoch 回退 → reducer 规则 1 永久丢弃该 unit 后续事件）。
RETRY_BUDGET_INITIAL: int = 3


class LifecycleContractError(ValueError):
    """(lifecycle_type, event) ∉ SUPPORTED_EVENTS / epoch 违约 / reason 越界时抛出。"""


class LifecycleDetailV1(BaseModel):
    """受控语义载荷（spec §3.1，R4#1）：extra=forbid，字段按白名单。

    无 ``attempt`` 字段（R12#P2：与顶层 epoch 冗余，防双源发散）。

    Documented deviation（R1 review P3-1）：白名单为全局扁平 5 字段并集，
    非 spec 措辞暗示的逐 (lifecycle_type, event) 对粒度——构造集中于
    helper/投影器 + extra=forbid 已达成 §3.1 敏感信息禁入的核心目标；
    逐对收紧留给后续迭代。
    """

    model_config = ConfigDict(extra="forbid")

    trigger: Optional[str] = None
    previous_state: Optional[str] = None
    retry_budget_remaining: Optional[int] = None
    original_outcome: Optional[str] = None
    note: Optional[str] = None


class LifecycleCorrelationV1(BaseModel):
    """subagent/coordinator 关联字段（spec §3.1，R6#P2c）：extra=forbid。"""

    model_config = ConfigDict(extra="forbid")

    work_unit_id: Optional[str] = None
    coordinator_run_id: Optional[str] = None
    coordinator_attempt_ix: Optional[int] = None


@dataclass(frozen=True)
class RetryLifecycleContext:
    """retry_from_suspend → 新 task runner 的类型化重试上下文（spec §5，R3#7）。

    构造/透传不受 flag 门控（R7#P3c：它不是 LifecycleEvent，不违反 INV-C7-3
    零构造承诺；只有 flag-gated 的 emit 点消费它）。
    """

    retry_budget_remaining: int
    trigger: str = "user"
    previous_state: str = "suspended"
