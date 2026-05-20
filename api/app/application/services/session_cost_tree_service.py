"""C1a: tree-aware cost rollup service.

Composes SessionRepository (lineage walk) + CostRecordRepository (batch fetch)
+ CostAggregationService.aggregate_rows (rubric reuse). Live aggregate under
READ COMMITTED - concurrent writes to descendants' cost_records show up as
partial when status mixed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.application.errors.exceptions import NotFoundError
from app.application.services.cost_aggregation_service import (
    CostAggregate,
    CostAggregationService,
)
from app.domain.services.subagent_limits import MAX_DESCENDANTS_PER_ROOT, MAX_SUBAGENT_DEPTH

if TYPE_CHECKING:
    from app.domain.repositories.cost_record_repository import CostRecordRepository
    from app.domain.repositories.session_repository import SessionRepository


@dataclass(frozen=True)
class CostTreeAggregate:
    session_id: str
    self_cost: CostAggregate
    descendants_cost: CostAggregate
    total_cost: CostAggregate
    descendant_ids: list[str]
    depth_reached: int
    max_depth_applied: int
    truncated: bool


class SessionCostTreeService:
    def __init__(
        self,
        *,
        session_repo: "SessionRepository",
        cost_repo: "CostRecordRepository",
        cost_aggregator: CostAggregationService,
    ) -> None:
        self._sessions = session_repo
        self._costs = cost_repo
        self._aggregator = cost_aggregator

    async def get_tree_aggregate(
        self,
        session_id: str,
        *,
        user_id: str,
        max_depth: int | None = None,
    ) -> CostTreeAggregate:
        effective_depth = min(
            max_depth if max_depth is not None else MAX_SUBAGENT_DEPTH,
            MAX_SUBAGENT_DEPTH,
        )

        self_session = await self._sessions.find_by_id_for_user(
            session_id, user_id=user_id
        )
        if self_session is None:
            raise NotFoundError(f"session {session_id} not found")

        descendants = await self._sessions.find_descendants(
            session_id,
            user_id=user_id,
            max_depth=effective_depth,
            limit=MAX_DESCENDANTS_PER_ROOT + 1,
        )
        truncated = len(descendants) > MAX_DESCENDANTS_PER_ROOT
        descendants = descendants[:MAX_DESCENDANTS_PER_ROOT]

        rows = await self._costs.find_by_sessions_for_user(
            [self_session.id, *[d.id for d in descendants]],
            user_id=user_id,
        )

        self_rows = [r for r in rows if r.session_id == self_session.id]
        desc_rows = [r for r in rows if r.session_id != self_session.id]

        return CostTreeAggregate(
            session_id=session_id,
            self_cost=self._aggregator.aggregate_rows(self_rows),
            descendants_cost=self._aggregator.aggregate_rows(desc_rows),
            total_cost=self._aggregator.aggregate_rows(rows),
            descendant_ids=[d.id for d in descendants],
            depth_reached=max((1 for _ in descendants), default=0),
            max_depth_applied=effective_depth,
            truncated=truncated,
        )
