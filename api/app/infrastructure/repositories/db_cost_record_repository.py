"""B4 M0 Phase I: DbCostRecordRepository — SQLAlchemy impl of the cost record repo.

The ``insert`` path is idempotent on ``run_id`` (uq_cost_records_run_id) so
callback retries cannot double-bill a single LLM call. On conflict we do
nothing — the first row wins.

``find_by_session`` returns rows ordered by (created_at, step_ix) so the
aggregation service doesn't have to re-sort (though it does anyway for
safety).
"""

from __future__ import annotations

from typing import List

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models.cost_record import CostRecord, CostStatus
from app.infrastructure.models.cost_record_orm import CostRecordModel


class DbCostRecordRepository:
    """Postgres-backed CostRecordRepository."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def insert(self, record: CostRecord) -> None:
        stmt = (
            pg_insert(CostRecordModel.__table__)
            .values(
                id=record.id,
                session_id=record.session_id,
                user_id=record.user_id,
                run_id=record.run_id,
                node_name=record.node_name,
                step_ix=record.step_ix,
                attempt_ix=record.attempt_ix,
                model=record.model,
                provider=record.provider,
                input_tokens=record.input_tokens,
                output_tokens=record.output_tokens,
                cache_read_tokens=record.cache_read_tokens,
                cache_write_tokens=record.cache_write_tokens,
                reasoning_tokens=record.reasoning_tokens,
                total_usd=record.total_usd,
                pricing_version=record.pricing_version,
                cost_status=record.cost_status.value,
                created_at=record.created_at,
            )
            .on_conflict_do_nothing(index_elements=["run_id"])
        )
        await self._session.execute(stmt)
        # Commit is the UoW's job (agent_service / service_dependencies wrap
        # the insert in ``async with uow_factory() as uow:``). Committing here
        # double-commits and fragments the transaction.

    async def find_by_session(self, session_id: str) -> List[CostRecord]:
        stmt = (
            select(CostRecordModel)
            .where(CostRecordModel.session_id == session_id)
            .order_by(CostRecordModel.created_at, CostRecordModel.step_ix)
        )
        result = await self._session.execute(stmt)
        return [_orm_to_domain(row) for row in result.scalars().all()]


def _orm_to_domain(row: CostRecordModel) -> CostRecord:
    return CostRecord(
        id=row.id,
        session_id=row.session_id,
        user_id=row.user_id,
        run_id=row.run_id,
        node_name=row.node_name,
        step_ix=row.step_ix,
        attempt_ix=row.attempt_ix,
        model=row.model,
        provider=row.provider,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        cache_read_tokens=row.cache_read_tokens,
        cache_write_tokens=row.cache_write_tokens,
        reasoning_tokens=row.reasoning_tokens,
        total_usd=row.total_usd,
        pricing_version=row.pricing_version,
        cost_status=CostStatus(row.cost_status),
        created_at=row.created_at,
    )
