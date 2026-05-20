"""B4 M0 Phase H: CostAggregationService — session-scoped cost rollup.

Consumers: ``GET /api/v1/sessions/{id}/cost`` endpoint + M2 replay path.

The aggregate is computed on demand from raw CostRecord rows; we don't cache
(GET /cost is low-QPS, session-scoped, and the DB is already indexed on
session_id). If that changes, Redis-cache the rollup here keyed by
(session_id, record_count) with TTL matching the session's mailbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from app.domain.models.cost_record import CostStatus

if TYPE_CHECKING:
    from app.domain.models.cost_record import CostRecord
    from app.domain.repositories.cost_record_repository import CostRecordRepository


@dataclass(frozen=True)
class CostAggregate:
    """Single-session cost rollup returned to the UI."""

    total_usd: Decimal
    record_count: int
    by_node: dict[str, Decimal]
    by_model: dict[str, Decimal]
    by_provider: dict[str, Decimal]
    pricing_version: str
    cost_status: CostStatus
    first_record_at: Optional[datetime]
    last_record_at: Optional[datetime]
    has_partial_records: bool


def _aggregate_status(statuses: set[CostStatus]) -> CostStatus:
    """Issue 2A rubric — don't silently relabel mixed sessions as ``actual``."""
    if not statuses:
        return CostStatus.UNKNOWN
    if len(statuses) == 1:
        return next(iter(statuses))
    return CostStatus.PARTIAL


# Sentinel written by ``CostCallbackHandler._persist_safely`` when the real
# persist fails and a best-effort marker lands instead. Seeing this in
# ``by_node`` means the ledger is known-incomplete for this session and the
# aggregate must flip to ``partial`` regardless of other rows' shape.
_DEGRADED_MARKER_NODE: str = "persist_degraded"


class CostAggregationService:
    def __init__(self, repository: "CostRecordRepository") -> None:
        self._repo = repository

    @staticmethod
    def aggregate_rows(rows: list["CostRecord"]) -> CostAggregate:
        """C1a: pure rollup over already-fetched rows.

        Extracted from `get_aggregate` so `SessionCostTreeService` can re-use
        the partial/degraded/mixed status rubric without going through the
        repository again.
        """
        if not rows:
            return CostAggregate(
                total_usd=Decimal(0),
                record_count=0,
                by_node={},
                by_model={},
                by_provider={},
                pricing_version="",
                cost_status=CostStatus.UNKNOWN,
                first_record_at=None,
                last_record_at=None,
                has_partial_records=False,
            )

        # Tuple sort (Issue OV-3): (created_at, step_ix) — both tz-aware /
        # int so sort never explodes on mixed values.
        rows_sorted = sorted(rows, key=lambda r: (r.created_at, r.step_ix))

        total_usd = Decimal(0)
        by_node: dict[str, Decimal] = {}
        by_model: dict[str, Decimal] = {}
        by_provider: dict[str, Decimal] = {}
        statuses: set[CostStatus] = set()
        pricing_versions: set[str] = set()

        for r in rows_sorted:
            total_usd += r.total_usd
            by_node[r.node_name] = by_node.get(r.node_name, Decimal(0)) + r.total_usd
            by_model[r.model] = by_model.get(r.model, Decimal(0)) + r.total_usd
            by_provider[r.provider] = (
                by_provider.get(r.provider, Decimal(0)) + r.total_usd
            )
            statuses.add(r.cost_status)
            pricing_versions.add(r.pricing_version)

        status = _aggregate_status(statuses)
        # Degraded-marker override: any row written by the handler's
        # persist-failure fallback means "ledger known incomplete".
        # Force the aggregate to partial so a single-call session whose
        # sole row is the marker doesn't show as clean ``unknown``.
        has_degraded_marker = any(
            r.node_name == _DEGRADED_MARKER_NODE for r in rows_sorted
        )
        if has_degraded_marker:
            status = CostStatus.PARTIAL
        has_partial = status == CostStatus.PARTIAL

        pricing_version = (
            next(iter(pricing_versions))
            if len(pricing_versions) == 1
            else "mixed"
        )

        return CostAggregate(
            total_usd=total_usd,
            record_count=len(rows_sorted),
            by_node=by_node,
            by_model=by_model,
            by_provider=by_provider,
            pricing_version=pricing_version,
            cost_status=status,
            first_record_at=rows_sorted[0].created_at,
            last_record_at=rows_sorted[-1].created_at,
            has_partial_records=has_partial,
        )

    async def get_aggregate(self, session_id: str) -> CostAggregate:
        rows: list["CostRecord"] = await self._repo.find_by_session(session_id)
        return self.aggregate_rows(rows)
