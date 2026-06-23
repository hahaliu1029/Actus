"""B4 M0 Phase I: GET /sessions/{session_id}/cost — per-session cost rollup endpoint.

Returns a ``CostAggregateResponse`` assembled by ``CostAggregationService``.
Scope-checked against the current user (session owner or admin) so one user
can't see another's spend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, Query

from app.domain.services.subagent_limits import MAX_SUBAGENT_DEPTH
from app.interfaces.dependencies import CurrentUser, rate_limit_read
from app.interfaces.schemas import Response
from app.interfaces.schemas.cost_schemas import CostAggregateResponse, CostTreeResponse
from app.interfaces.service_dependencies import (
    get_cost_aggregation_service,
    get_session_cost_tree_service,
    get_session_service,
    get_subagent_limits,
)
from core.config import SubagentLimitsConfig

if TYPE_CHECKING:
    from app.application.services.cost_aggregation_service import (
        CostAggregate,
        CostAggregationService,
    )
    from app.application.services.session_cost_tree_service import (
        SessionCostTreeService,
    )
    from app.application.services.session_service import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["会话成本"])


def _aggregate_to_response(agg: "CostAggregate") -> CostAggregateResponse:
    """C1a: application CostAggregate (dataclass) -> interface schema (Pydantic).

    Mirrors the field-by-field construction in ``get_session_cost``.
    """
    return CostAggregateResponse(
        total_usd=agg.total_usd,
        record_count=agg.record_count,
        by_node=agg.by_node,
        by_model=agg.by_model,
        by_provider=agg.by_provider,
        pricing_version=agg.pricing_version,
        cost_status=agg.cost_status.value,
        first_record_at=agg.first_record_at,
        last_record_at=agg.last_record_at,
        has_partial_records=agg.has_partial_records,
    )


@router.get(
    path="/{session_id}/cost",
    response_model=Response[CostAggregateResponse],
    summary="获取单会话 LLM 成本聚合",
    description=(
        "返回 total_usd / by_node / by_model / by_provider / "
        "cost_status（actual|estimated|partial|unknown）。"
    ),
    dependencies=[Depends(rate_limit_read)],
)
async def get_session_cost(
    session_id: str,
    current_user: CurrentUser,
    session_service: "SessionService" = Depends(get_session_service),
    cost_service: "CostAggregationService" = Depends(get_cost_aggregation_service),
) -> Response[CostAggregateResponse]:
    # Scope check — session_service.get_session raises a typed error if the
    # session doesn't exist or the caller isn't authorized.
    await session_service.get_session(
        session_id=session_id,
        user_id=current_user.id,
        is_admin=current_user.is_admin(),
    )

    aggregate = await cost_service.get_aggregate(session_id)
    return Response.success(data=_aggregate_to_response(aggregate))


@router.get(
    path="/{session_id}/cost/tree",
    response_model=Response[CostTreeResponse],
    summary="返回 session 树（self + descendants）的 LLM 成本聚合",
    description=(
        "Live aggregate under READ COMMITTED. truncated=true marks depth/descendants "
        "hitting the cap. Frontend should poll or refresh via SSE for fresh totals."
    ),
    dependencies=[Depends(rate_limit_read)],
)
async def get_session_cost_tree(
    session_id: str,
    current_user: CurrentUser,
    service: "SessionCostTreeService" = Depends(get_session_cost_tree_service),
    depth: int = Query(default=MAX_SUBAGENT_DEPTH, ge=1, le=10),
    limits: SubagentLimitsConfig = Depends(get_subagent_limits),
) -> Response[CostTreeResponse]:
    """C1a: GET /api/sessions/{session_id}/cost/tree — tree cost rollup."""
    effective_depth = min(depth, limits.max_subagent_depth)
    agg = await service.get_tree_aggregate(
        session_id,
        user_id=current_user.id,
        max_depth=effective_depth,
    )
    payload = CostTreeResponse(
        session_id=agg.session_id,
        self_cost=_aggregate_to_response(agg.self_cost),
        descendants_cost=_aggregate_to_response(agg.descendants_cost),
        total_cost=_aggregate_to_response(agg.total_cost),
        descendant_ids=agg.descendant_ids,
        depth_reached=agg.depth_reached,
        max_depth_applied=agg.max_depth_applied,
        truncated=agg.truncated or (effective_depth < depth),
        # [C2 PR-6 §14.4] Surface the attribution label the service derived
        # from the descendants' tool_filter_preset + worker_type buckets.
        # Pydantic v2 serialises the ``CostSource`` StrEnum to its wire
        # string value (e.g. ``"coordinator_subagent"``) automatically.
        cost_source=agg.cost_source,
    )
    return Response.success(data=payload)
