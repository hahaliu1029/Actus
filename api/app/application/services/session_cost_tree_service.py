"""C1a: tree-aware cost rollup service.

Composes SessionRepository (lineage walk) + CostRecordRepository (batch fetch)
+ CostAggregationService.aggregate_rows (rubric reuse). Live aggregate under
READ COMMITTED - concurrent writes to descendants' cost_records show up as
partial when status mixed.

[C2 PR-6 §14.4] Augmented with ``cost_source`` attribution: descendants are
bucketed by their ``tool_filter_preset`` (``coordinator_step`` → coordinator
child, ``subagent_research`` or NULL-with-worker_type=subagent → research
child) so the ``GET /cost/tree`` response can carry a frontend-renderable
``CostSource`` label without re-walking the tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from app.application.errors.exceptions import NotFoundError
from app.application.services.cost_aggregation_service import (
    CostAggregate,
    CostAggregationService,
)
from app.domain.models.cost_snapshot import CostSource, SessionCostSnapshot
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET
from app.domain.services.subagent_limits import MAX_DESCENDANTS_PER_ROOT, MAX_SUBAGENT_DEPTH

if TYPE_CHECKING:
    from app.domain.models.cost_record import CostRecord
    from app.domain.models.session import Session
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
    # [C2 PR-6 §14.4] Attribution label for the rolled-up cost. Derived from
    # ``SessionCostSnapshot`` over (direct, coordinator_child, research_child)
    # buckets. Defaults to ``CostSource.NONE`` so external callers (tests,
    # backfill scripts) that instantiate ``CostTreeAggregate`` without the
    # new field keep working — the production code path in
    # ``SessionCostTreeService.get_tree_aggregate`` always sets it explicitly.
    cost_source: CostSource = CostSource.NONE


class SessionCostTreeService:
    def __init__(
        self,
        *,
        session_repo: "SessionRepository",
        cost_repo: "CostRecordRepository",
        cost_aggregator: CostAggregationService,
        max_subagent_depth: int = MAX_SUBAGENT_DEPTH,
    ) -> None:
        self._sessions = session_repo
        self._costs = cost_repo
        self._aggregator = cost_aggregator
        # C2-full S3 (PR-3) — runtime ceiling. Defaulted to the constant so the
        # pre-existing ctor sites stay green; prod wires limits.max_subagent_depth
        # at service_dependencies.get_session_cost_tree_service.
        self._max_subagent_depth = max_subagent_depth

    async def get_tree_aggregate(
        self,
        session_id: str,
        *,
        user_id: str,
        max_depth: int | None = None,
    ) -> CostTreeAggregate:
        effective_depth = min(
            max_depth if max_depth is not None else self._max_subagent_depth,
            self._max_subagent_depth,
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

        # [C2 PR-6 §14.4] Bucket descendant cost rows by their session's
        # ``tool_filter_preset`` for the ``cost_source`` attribution label.
        # ``find_descendants`` already filtered to this user_id, so the map
        # is safe to build off the descendants list.
        descendants_map: dict[str, "Session"] = {d.id: d for d in descendants}
        cost_source = self._compute_cost_source(
            self_rows=self_rows,
            desc_rows=desc_rows,
            descendants_map=descendants_map,
        )

        return CostTreeAggregate(
            session_id=session_id,
            self_cost=self._aggregator.aggregate_rows(self_rows),
            descendants_cost=self._aggregator.aggregate_rows(desc_rows),
            total_cost=self._aggregator.aggregate_rows(rows),
            descendant_ids=[d.id for d in descendants],
            depth_reached=max(
                (d.depth - self_session.depth for d in descendants),
                default=0,
            ),
            max_depth_applied=effective_depth,
            truncated=truncated,
            cost_source=cost_source,
        )

    @staticmethod
    def _compute_cost_source(
        *,
        self_rows: list["CostRecord"],
        desc_rows: list["CostRecord"],
        descendants_map: dict[str, "Session"],
    ) -> CostSource:
        """[C2 PR-6 §14.4] Bucket cost rows into the three attribution
        dimensions and derive the ``CostSource`` label via
        ``SessionCostSnapshot``.

        Bucketing rule (spec §14.4):
        - ``tool_filter_preset == "coordinator_step"`` → coordinator bucket
        - ``tool_filter_preset == "subagent_research"`` → research bucket
        - ``tool_filter_preset is None AND worker_type == "subagent"`` →
          research bucket (legacy compat for pre-C2 subagents)
        - All other rows (including descendants we somehow lost the
          ``Session`` for) are dropped from the attribution view — they
          still count in ``total_cost`` but won't tilt the source label.

        Self-session rows go straight to the ``direct`` bucket regardless
        of the root session's own ``worker_type`` because the root, from
        the caller's perspective, *is* the direct cost.
        """
        coord_total = Decimal(0)
        research_total = Decimal(0)
        for row in desc_rows:
            session = descendants_map.get(row.session_id)
            if session is None:
                # Defensive: row referenced a descendant we don't have in
                # the map (shouldn't happen post-truncation since we only
                # query for self+kept descendant ids, but guard anyway).
                continue
            preset = session.tool_filter_preset
            if preset == COORDINATOR_STEP_PRESET:
                coord_total += row.total_usd
            elif preset == "subagent_research":
                research_total += row.total_usd
            elif preset is None and session.worker_type == "subagent":
                # Legacy pre-C2 subagent — pre-existing rows don't carry
                # a preset; bucket as research per spec §14.4.
                research_total += row.total_usd

        direct_total = sum(
            (r.total_usd for r in self_rows), start=Decimal(0)
        )

        snapshot = SessionCostSnapshot(
            direct_cost_usd=float(direct_total),
            coordinator_child_cost_usd=float(coord_total),
            research_child_cost_usd=float(research_total),
        )
        return snapshot.cost_source
