"""B4 M0 Phase H: CostAggregationService — session-scoped cost rollup.

Locks Issue 2A: when rows carry mixed ``cost_status`` values, the aggregate
must reflect that truthfully instead of silently downgrading to ``actual``.

Aggregate cost_status rubric:
    no rows                        → unknown
    all actual                     → actual
    all estimated                  → estimated
    all unknown                    → unknown
    any mix of actual/est/unknown  → partial
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import List
from uuid import uuid4

import pytest

from app.application.services.cost_aggregation_service import (
    CostAggregate,
    CostAggregationService,
)
from app.domain.models.cost_record import CostRecord, CostStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _InMemoryCostRepo:
    """Fake repository scoped to a single test — returns pre-seeded rows."""

    def __init__(self, rows: List[CostRecord]) -> None:
        self._rows = list(rows)

    async def find_by_session(self, session_id: str) -> List[CostRecord]:
        return [r for r in self._rows if r.session_id == session_id]


def _make_record(
    session_id: str = "sess-1",
    *,
    created_at: datetime | None = None,
    node: str = "planner",
    model: str = "gpt-4o",
    provider: str = "openai",
    total_usd: str = "0.001",
    input_tokens: int = 100,
    output_tokens: int = 50,
    status: CostStatus = CostStatus.ACTUAL,
    step_ix: int = 0,
    pricing_version: str = "v1",
) -> CostRecord:
    return CostRecord(
        id=str(uuid4()),
        session_id=session_id,
        user_id="user-1",
        run_id=str(uuid4()),
        node_name=node,
        step_ix=step_ix,
        attempt_ix=0,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        total_usd=Decimal(total_usd),
        pricing_version=pricing_version,
        cost_status=status,
        created_at=created_at or datetime.now(timezone.utc),
    )


class TestEmptySession:
    async def test_empty_session_returns_unknown_zero_rollup(self) -> None:
        svc = CostAggregationService(repository=_InMemoryCostRepo([]))
        agg = await svc.get_aggregate("sess-empty")

        assert agg.record_count == 0
        assert agg.total_usd == Decimal(0)
        assert agg.cost_status == CostStatus.UNKNOWN
        assert agg.by_node == {}
        assert agg.by_model == {}
        assert agg.by_provider == {}
        assert agg.first_record_at is None
        assert agg.last_record_at is None
        assert agg.has_partial_records is False


class TestAllActualSession:
    async def test_all_actual_rows_sum_cleanly(self) -> None:
        t0 = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
        rows = [
            _make_record(created_at=t0, total_usd="0.001", node="planner", model="gpt-4o"),
            _make_record(
                created_at=t0 + timedelta(seconds=30),
                total_usd="0.002",
                node="executor",
                model="gpt-4o",
            ),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.record_count == 2
        assert agg.total_usd == Decimal("0.003")
        assert agg.cost_status == CostStatus.ACTUAL
        assert agg.by_node == {"planner": Decimal("0.001"), "executor": Decimal("0.002")}
        assert agg.by_model == {"gpt-4o": Decimal("0.003")}
        assert agg.by_provider == {"openai": Decimal("0.003")}
        assert agg.first_record_at == t0
        assert agg.last_record_at == t0 + timedelta(seconds=30)
        assert agg.has_partial_records is False


class TestMixedRowsMarkedPartial:
    async def test_actual_plus_estimated_yields_partial(self) -> None:
        rows = [
            _make_record(total_usd="0.01", status=CostStatus.ACTUAL),
            _make_record(total_usd="0", status=CostStatus.ESTIMATED),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.cost_status == CostStatus.PARTIAL
        assert agg.has_partial_records is True

    async def test_actual_plus_unknown_yields_partial(self) -> None:
        rows = [
            _make_record(total_usd="0.01", status=CostStatus.ACTUAL),
            _make_record(total_usd="0", status=CostStatus.UNKNOWN),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.cost_status == CostStatus.PARTIAL
        assert agg.has_partial_records is True


class TestEstimatedOnlyStaysEstimated:
    async def test_all_estimated_does_not_downgrade_to_actual(self) -> None:
        rows = [
            _make_record(total_usd="0", status=CostStatus.ESTIMATED),
            _make_record(total_usd="0", status=CostStatus.ESTIMATED),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.cost_status == CostStatus.ESTIMATED
        assert agg.has_partial_records is False


class TestAllUnknownStaysUnknown:
    async def test_all_unknown_does_not_get_marked_partial(self) -> None:
        rows = [
            _make_record(status=CostStatus.UNKNOWN),
            _make_record(status=CostStatus.UNKNOWN),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.cost_status == CostStatus.UNKNOWN


class TestBreakdownsGroupCorrectly:
    async def test_by_node_by_model_by_provider(self) -> None:
        rows = [
            _make_record(node="planner", model="gpt-4o", provider="openai", total_usd="0.01"),
            _make_record(node="executor", model="gpt-4o", provider="openai", total_usd="0.02"),
            _make_record(
                node="executor",
                model="deepseek-chat",
                provider="deepseek",
                total_usd="0.001",
            ),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.by_node == {
            "planner": Decimal("0.01"),
            "executor": Decimal("0.021"),
        }
        assert agg.by_model == {
            "gpt-4o": Decimal("0.03"),
            "deepseek-chat": Decimal("0.001"),
        }
        assert agg.by_provider == {
            "openai": Decimal("0.03"),
            "deepseek": Decimal("0.001"),
        }


class TestPricingVersionRollup:
    async def test_single_version_exposed(self) -> None:
        rows = [_make_record(pricing_version="v1") for _ in range(3)]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")
        assert agg.pricing_version == "v1"

    async def test_multiple_versions_shown_as_mixed(self) -> None:
        rows = [
            _make_record(pricing_version="v1"),
            _make_record(pricing_version="v2"),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.pricing_version == "mixed", (
            "Sessions spanning more than one pricing_version must flag it "
            "explicitly instead of reporting just one."
        )


class TestTzAwareOrderingDoesNotExplode:
    async def test_sort_with_same_timestamp_is_stable(self) -> None:
        t = datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)
        rows = [
            _make_record(created_at=t, step_ix=2),
            _make_record(created_at=t, step_ix=0),
            _make_record(created_at=t, step_ix=1),
        ]
        svc = CostAggregationService(repository=_InMemoryCostRepo(rows))
        agg = await svc.get_aggregate("sess-1")

        assert agg.first_record_at == t
        assert agg.last_record_at == t
        assert agg.record_count == 3


def test_cost_aggregate_dataclass_shape() -> None:
    expected = {
        "total_usd",
        "record_count",
        "by_node",
        "by_model",
        "by_provider",
        "pricing_version",
        "cost_status",
        "first_record_at",
        "last_record_at",
        "has_partial_records",
    }
    fields = {f.name for f in CostAggregate.__dataclass_fields__.values()}
    missing = expected - fields
    assert not missing, f"CostAggregate missing fields: {missing}"
