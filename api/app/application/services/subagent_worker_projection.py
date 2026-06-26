"""C4 LOCAL/research 结果投影 helper（spec §6）。

把已发布的 LOCAL（coordinator child）/ research 结果投影成统一
SubagentRunResult。放 application/：domain 禁 import application/interfaces；
application 可 import domain DTO + 既有 application/interface 形状。
application→interfaces import（ChildOutcome）已是既有先例
（subagent_research_service.py:55-57 自身就这么 import）。

投影是「受限无损」（spec §6.1）：在通用 status/summary/cost/artifacts 字面上
无损；patch_manifest / needs_authorization_details 结构化全文有意丢弃（仅
needs_authorization reason+requested_tool 入 error_summary）。
"""
from __future__ import annotations

from typing import Optional

from app.domain.models.mailbox_envelope import ResultReadyOutcome, ResultReadyPayload
from app.domain.models.needs_authorization_details import NeedsAuthorizationDetails
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)
from app.interfaces.schemas.subagent import ChildOutcome


# spec §6.1 — ResultReadyOutcome → WorkerTerminalOutcome（1:1，全 5 值）
_LOCAL_OUTCOME_MAP: dict[ResultReadyOutcome, WorkerTerminalOutcome] = {
    ResultReadyOutcome.SUCCESS: WorkerTerminalOutcome.SUCCESS,
    ResultReadyOutcome.FAILED: WorkerTerminalOutcome.FAILED,
    ResultReadyOutcome.CANCELLED: WorkerTerminalOutcome.CANCELLED,
    ResultReadyOutcome.TIMED_OUT: WorkerTerminalOutcome.TIMED_OUT,
    ResultReadyOutcome.NEEDS_AUTHORIZATION: WorkerTerminalOutcome.NEEDS_AUTHORIZATION,
}


# spec §6.2 — ChildOutcome →（lifecycle_state, terminal_outcome）
_RESEARCH_OUTCOME_MAP: dict[
    ChildOutcome, tuple[WorkerLifecycleState, Optional[WorkerTerminalOutcome]]
] = {
    ChildOutcome.COMPLETED: (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.SUCCESS),
    ChildOutcome.FAILED: (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.FAILED),
    ChildOutcome.TIMED_OUT: (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.TIMED_OUT),
    ChildOutcome.CANCELLED: (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.CANCELLED),
    ChildOutcome.WAITING: (WorkerLifecycleState.WAITING_INPUT, None),
}


def _format_needs_auth(details: Optional[NeedsAuthorizationDetails]) -> str:
    """NEEDS_AUTHORIZATION reason（inline 闭合 Literal）+ 可选 requested_tool
    入 error_summary。不取 free-text rationale（MinIO rationale_ref，绝不 fetch）。
    """
    if details is None:
        return "needs_authorization"
    base = f"needs_authorization: {details.reason}"
    if details.requested_tool:
        base += f" (tool={details.requested_tool})"
    return base


def project_local_result(
    payload: ResultReadyPayload,
    *,
    parent_session_id: str,
    child_session_id: str,
    work_unit_id: str,
) -> SubagentRunResult:
    """投影 coordinator child 的 ResultReadyPayload → SubagentRunResult（LOCAL）。

    投影源 = ResultReadyPayload（wire / RESULT_READY 载荷，frozen + AST-gated，
    有 artifacts）；in-graph WorkerResult（派生有损 carrier，无 artifacts）不作源。
    """
    outcome = _LOCAL_OUTCOME_MAP[payload.outcome]
    error_summary = None
    if payload.outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION:
        error_summary = _format_needs_auth(payload.needs_authorization_details)
    return SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=outcome,
        summary=payload.summary,
        artifacts=list(payload.artifacts),         # 透传 payload 自带 ArtifactRef
        cost_summary=payload.cost_summary,
        cost_authoritative=True,                   # LOCAL cost 权威
        duration_source="unavailable",             # LOCAL 结果对象不带 duration
        error_summary=error_summary,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        source_ref=work_unit_id,
    )


def project_research_result(
    *,
    child_id: str,
    outcome: ChildOutcome,
    final_answer: Optional[str],
    transcript_tokens: int,
    error_summary: Optional[str],
    parent_session_id: str,
    child_session_id: str,
) -> SubagentRunResult:
    """投影 research 子（同进程 subagent）的解构结果 → SubagentRunResult（LOCAL）。

    research 子也是 LOCAL 执行底座（同进程）。``transcript_tokens`` 为签名保真
    而接收，但 **有意不** 折入 cost（transcript 总量 ≠ output tokens，会误导；
    spec §6.2 R1#P1-5）——cost_summary=None, cost_authoritative=False。
    """
    lifecycle, terminal = _RESEARCH_OUTCOME_MAP[outcome]
    return SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=lifecycle,
        terminal_outcome=terminal,
        summary=final_answer or "",
        cost_summary=None,
        cost_authoritative=False,
        duration_source="unavailable",
        error_summary=error_summary,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        source_ref=child_id,
    )
