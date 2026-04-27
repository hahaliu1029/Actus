"""B4 M0 Phase I: GET /sessions/{session_id}/cost — per-session cost rollup endpoint.

Returns a ``CostAggregateResponse`` assembled by ``CostAggregationService``.
Scope-checked against the current user (session owner or admin) so one user
can't see another's spend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends

from app.interfaces.dependencies import CurrentUser, rate_limit_read
from app.interfaces.schemas import Response
from app.interfaces.schemas.cost_schemas import CostAggregateResponse
from app.interfaces.service_dependencies import (
    get_cost_aggregation_service,
    get_session_service,
)

if TYPE_CHECKING:
    from app.application.services.cost_aggregation_service import (
        CostAggregationService,
    )
    from app.application.services.session_service import SessionService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["会话成本"])


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
    payload = CostAggregateResponse(
        total_usd=aggregate.total_usd,
        record_count=aggregate.record_count,
        by_node=aggregate.by_node,
        by_model=aggregate.by_model,
        by_provider=aggregate.by_provider,
        pricing_version=aggregate.pricing_version,
        cost_status=aggregate.cost_status.value,
        first_record_at=aggregate.first_record_at,
        last_record_at=aggregate.last_record_at,
        has_partial_records=aggregate.has_partial_records,
    )
    return Response.success(data=payload)
