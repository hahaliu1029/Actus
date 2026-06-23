"""C1a: SessionCostTreeService rollup composition tests."""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.application.errors.exceptions import NotFoundError
from app.application.services.cost_aggregation_service import CostAggregationService
from app.application.services.session_cost_tree_service import (
    CostTreeAggregate,
    SessionCostTreeService,
)
from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.models.session import Session


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, *, self_session: Session | None, descendants: list[Session]):
        self._self = self_session
        self._desc = descendants

    async def find_by_id_for_user(self, session_id, *, user_id):
        if self._self and self._self.id == session_id and self._self.user_id == user_id:
            return self._self
        return None

    async def find_descendants(self, ancestor_id, *, user_id, max_depth, limit):
        del ancestor_id, user_id, max_depth  # silence Pylance unused-param warnings
        return self._desc[: limit]


def _row(session_id: str, usd: str, *, status: CostStatus = CostStatus.ACTUAL) -> CostRecord:
    """Real CostRecord requires all 18 fields (id, session_id, user_id, run_id, node_name,
    step_ix, attempt_ix, model, provider, input_tokens, output_tokens,
    cache_read_tokens, cache_write_tokens, reasoning_tokens, total_usd,
    pricing_version, cost_status, created_at). created_at MUST be tz-aware.
    """
    import uuid as _uuid
    return CostRecord(
        id=_uuid.uuid4().hex,
        session_id=session_id,
        user_id="u1",
        run_id=f"run-{session_id}-{usd}",
        node_name="n",
        step_ix=0,
        attempt_ix=0,
        model="m",
        provider="p",
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        total_usd=Decimal(usd),
        pricing_version="v1",
        cost_status=status,
        created_at=datetime.now(tz=timezone.utc),
    )


class _CostRepo:
    def __init__(self, rows: list[CostRecord]):
        self._rows = rows
        self.calls: list[tuple[list[str], str]] = []

    async def find_by_sessions_for_user(self, session_ids, *, user_id):
        self.calls.append((session_ids, user_id))
        return [r for r in self._rows if r.session_id in session_ids]


@pytest.mark.anyio
async def test_returns_404_for_missing_or_foreign_session():
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=None, descendants=[]),
        cost_repo=_CostRepo([]),
        cost_aggregator=CostAggregationService(_CostRepo([])),
    )
    with pytest.raises(NotFoundError):
        await svc.get_tree_aggregate("missing", user_id="u1", max_depth=1)


@pytest.mark.anyio
async def test_self_plus_descendants_rollup_reuses_aggregate_rows():
    self_s = Session(id="root", user_id="u1", worker_type="root")
    descendant = Session(
        id="child", user_id="u1", worker_type="subagent", parent_session_id="root"
    )
    rows = [
        _row("root", "1.00"),
        _row("child", "0.50"),
    ]
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=self_s, descendants=[descendant]),
        cost_repo=_CostRepo(rows),
        cost_aggregator=CostAggregationService(_CostRepo([])),
    )
    agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)
    assert isinstance(agg, CostTreeAggregate)
    assert agg.session_id == "root"
    assert agg.self_cost.total_usd == Decimal("1.00")
    assert agg.descendants_cost.total_usd == Decimal("0.50")
    assert agg.total_cost.total_usd == Decimal("1.50")
    assert agg.descendant_ids == ["child"]
    assert agg.truncated is False


@pytest.mark.anyio
async def test_partial_status_propagates_into_total():
    """If a descendant row has partial status, total_cost must NOT silently downgrade to actual."""
    self_s = Session(id="root", user_id="u1", worker_type="root")
    descendant = Session(
        id="child", user_id="u1", worker_type="subagent", parent_session_id="root"
    )
    rows = [
        _row("root", "1.00", status=CostStatus.ACTUAL),
        _row("child", "0.50", status=CostStatus.PARTIAL),
    ]
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=self_s, descendants=[descendant]),
        cost_repo=_CostRepo(rows),
        cost_aggregator=CostAggregationService(_CostRepo([])),
    )
    agg = await svc.get_tree_aggregate("root", user_id="u1", max_depth=1)
    assert agg.total_cost.cost_status == CostStatus.PARTIAL


@pytest.mark.anyio
async def test_cost_tree_clamp_and_depth_reached_honor_injected_ceiling():
    """S3 PR-3: with max_subagent_depth=2 injected, the walk clamps to 2 and
    depth_reached is computed from the persisted depth relative to the root."""
    self_s = Session(id="root", user_id="u1", worker_type="root", depth=0)
    c1 = Session(
        id="c1", user_id="u1", worker_type="subagent",
        parent_session_id="root", depth=1, root_session_id="root",
    )
    c2 = Session(
        id="c2", user_id="u1", worker_type="subagent",
        parent_session_id="c1", depth=2, root_session_id="root",
    )
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=self_s, descendants=[c1, c2]),
        cost_repo=_CostRepo([]),
        cost_aggregator=CostAggregationService(_CostRepo([])),
        max_subagent_depth=2,
    )
    agg = await svc.get_tree_aggregate("root", user_id="u1")  # max_depth=None → ceiling
    assert agg.max_depth_applied == 2
    assert agg.depth_reached == 2  # max(1-0, 2-0)


@pytest.mark.anyio
async def test_cost_tree_default_parity_clamps_to_1():
    """INV-E: default ceiling=1 → effective_depth 1; depth_reached == 1 when a
    descendant exists (identical to the legacy hardcoded behavior)."""
    self_s = Session(id="root", user_id="u1", worker_type="root", depth=0)
    c1 = Session(
        id="c1", user_id="u1", worker_type="subagent",
        parent_session_id="root", depth=1, root_session_id="root",
    )
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=self_s, descendants=[c1]),
        cost_repo=_CostRepo([]),
        cost_aggregator=CostAggregationService(_CostRepo([])),
    )  # default max_subagent_depth = MAX_SUBAGENT_DEPTH = 1
    agg = await svc.get_tree_aggregate("root", user_id="u1")
    assert agg.max_depth_applied == 1
    assert agg.depth_reached == 1


@pytest.mark.anyio
async def test_cost_tree_depth_reached_zero_with_no_descendants():
    """INV-E: no descendants → depth_reached 0 (unchanged)."""
    self_s = Session(id="root", user_id="u1", worker_type="root", depth=0)
    svc = SessionCostTreeService(
        session_repo=_SessionRepo(self_session=self_s, descendants=[]),
        cost_repo=_CostRepo([]),
        cost_aggregator=CostAggregationService(_CostRepo([])),
    )
    agg = await svc.get_tree_aggregate("root", user_id="u1")
    assert agg.depth_reached == 0


def test_di_factory_threads_runtime_limit_into_cost_tree():
    """S3 PR-3 / R2-B1: the DI factory passes limits.max_subagent_depth into
    the service ctor (calling the factory directly bypasses FastAPI Depends)."""
    from app.interfaces.service_dependencies import get_session_cost_tree_service
    from core.config import SubagentLimitsConfig

    svc = get_session_cost_tree_service(
        session_repo=object(),
        cost_repo=object(),
        cost_aggregator=object(),
        limits=SubagentLimitsConfig(max_subagent_depth=2),
    )
    assert svc._max_subagent_depth == 2
