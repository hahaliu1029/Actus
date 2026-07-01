"""C4.1a subagent_runs 仓库实现（spec §4 / §3.3）。

持 session_factory（非 request-scoped）：两个消费者（MailboxSupervisor /
run_research）都是长驻，不适用 request-scoped Depends(get_db_session)。
每 call 开独立 AsyncSession；record 自持写事务，list 只读且在 context 内重建
（production session_factory 是 expire_on_commit=True → 必须在 session 关闭前
构建返回对象，避免 lazy-load）。
"""
from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.models.mailbox_envelope import ArtifactRef, CostAggregate
from app.domain.models.subagent_run_record import SubagentRunRecord
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)
from app.domain.repositories.subagent_run_repository import SubagentRunRepository
from app.infrastructure.models.subagent_run_orm import SubagentRunModel

# 写侧 cap（REMOTE-ready 防御；LOCAL 已受上游约束）。
_MAX_SUBAGENT_RUN_SUMMARY_CHARS = 16_384
_MAX_SUBAGENT_RUN_ARTIFACTS = 64
_MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS = 4_096


def _cap(value: Optional[str], limit: int) -> Optional[str]:
    if value is None:
        return None
    return value[:limit]


class DbSubagentRunRepository(SubagentRunRepository):
    """PostgreSQL 实现（持 session_factory）。"""

    def __init__(self, session_factory: async_sessionmaker) -> None:
        self._session_factory = session_factory

    async def record(self, result: SubagentRunResult) -> None:
        stmt = (
            pg_insert(SubagentRunModel)
            .values(**self._to_values(result))
            .on_conflict_do_nothing(constraint="uq_subagent_runs_child_session_id")
        )
        async with self._session_factory() as s:
            async with s.begin():
                await s.execute(stmt)

    async def list_by_parent_session(
        self, parent_session_id: str
    ) -> list[SubagentRunRecord]:
        stmt = (
            select(SubagentRunModel)
            .where(SubagentRunModel.parent_session_id == parent_session_id)
            .order_by(
                SubagentRunModel.created_at.asc(), SubagentRunModel.id.asc(),
            )
        )
        async with self._session_factory() as s:
            result = await s.execute(stmt)
            rows = result.scalars().all()
            # 在 session context 内构建，避免 expire_on_commit lazy-load。
            return [self._to_record(row) for row in rows]

    @staticmethod
    def _to_values(result: SubagentRunResult) -> dict:
        cost = result.cost_summary
        artifacts = [
            {
                "artifact_type": a.artifact_type,
                "ref": _cap(a.ref, _MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS),
                "description": _cap(
                    a.description, _MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS,
                ),
            }
            for a in result.artifacts[:_MAX_SUBAGENT_RUN_ARTIFACTS]
        ]
        return {
            "id": str(uuid.uuid4()),
            "runtime": result.worker_runtime_type.value,
            "lifecycle_state": result.lifecycle_state.value,
            "terminal_outcome": (
                result.terminal_outcome.value
                if result.terminal_outcome is not None
                else None
            ),
            "summary": _cap(result.summary, _MAX_SUBAGENT_RUN_SUMMARY_CHARS) or "",
            "error_summary": _cap(
                result.error_summary, _MAX_SUBAGENT_RUN_SUMMARY_CHARS,
            ),
            "parent_session_id": result.parent_session_id,
            "child_session_id": result.child_session_id,
            "source_ref": result.source_ref,
            "cost_authoritative": result.cost_authoritative,
            "cost_total_input_tokens": (
                cost.total_input_tokens if cost is not None else None
            ),
            "cost_total_output_tokens": (
                cost.total_output_tokens if cost is not None else None
            ),
            "cost_total_usd": cost.total_usd if cost is not None else None,
            "cost_tool_call_count": (
                cost.tool_call_count if cost is not None else None
            ),
            "duration_seconds": result.duration_seconds,
            "duration_source": result.duration_source,
            "artifacts": artifacts,
        }

    @staticmethod
    def _to_record(row: SubagentRunModel) -> SubagentRunRecord:
        has_cost = (
            row.cost_total_input_tokens is not None
            and row.cost_total_output_tokens is not None
            and row.cost_total_usd is not None
            and row.cost_tool_call_count is not None
        )
        cost_summary = (
            CostAggregate(
                total_input_tokens=row.cost_total_input_tokens,
                total_output_tokens=row.cost_total_output_tokens,
                total_usd=row.cost_total_usd,
                tool_call_count=row.cost_tool_call_count,
            )
            if has_cost
            else None
        )
        run = SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType(row.runtime),
            lifecycle_state=WorkerLifecycleState(row.lifecycle_state),
            terminal_outcome=(
                WorkerTerminalOutcome(row.terminal_outcome)
                if row.terminal_outcome is not None
                else None
            ),
            summary=row.summary,
            artifacts=[ArtifactRef(**a) for a in (row.artifacts or [])],
            cost_summary=cost_summary,
            cost_authoritative=row.cost_authoritative,
            duration_seconds=row.duration_seconds,
            duration_source=row.duration_source,
            error_summary=row.error_summary,
            parent_session_id=row.parent_session_id,
            child_session_id=row.child_session_id,
            source_ref=row.source_ref,
        )
        return SubagentRunRecord(id=row.id, created_at=row.created_at, run=run)
