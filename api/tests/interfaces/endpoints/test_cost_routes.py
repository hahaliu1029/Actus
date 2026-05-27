"""[C2 PR-6 §14.4] GET /api/sessions/{id}/cost/tree response surface tests.

Locks the HTTP-layer contract that the route hands the
``SessionCostTreeService.cost_source`` attribution label out to the
frontend. The service-level bucketing rule is covered by
``tests/application/services/test_session_cost_tree_cost_source.py``;
this file pins the API surface — schema field name, serialised wire
string, and end-to-end wiring through the route handler.

A unit-style FastAPI TestClient with ``app.dependency_overrides``
replaces auth + rate-limit + the cost-tree service so the test runs
without postgres / redis / lifespan setup.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.services.cost_aggregation_service import CostAggregate
from app.application.services.session_cost_tree_service import CostTreeAggregate
from app.domain.models.cost_record import CostStatus
from app.domain.models.cost_snapshot import CostSource
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies import rate_limit_read
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.endpoints.cost_routes import router as cost_router
from app.interfaces.service_dependencies import get_session_cost_tree_service


# ── Helpers ──────────────────────────────────────────────────────────────────


def _aggregate(total: str = "0") -> CostAggregate:
    """Minimal ``CostAggregate`` with a single ``total_usd`` value.

    All other fields default to empty / zero — the test only asserts on
    the ``cost_source`` wire field, not on the underlying aggregate
    payload (covered by the integration test).
    """
    return CostAggregate(
        total_usd=Decimal(total),
        record_count=0,
        by_node={},
        by_model={},
        by_provider={},
        pricing_version="v1",
        cost_status=CostStatus.ACTUAL,
        first_record_at=None,
        last_record_at=None,
        has_partial_records=False,
    )


def _tree_aggregate(*, cost_source: CostSource) -> CostTreeAggregate:
    """Build a ``CostTreeAggregate`` carrying the requested attribution
    label so the route handler has something concrete to serialise.
    """
    return CostTreeAggregate(
        session_id="root-session",
        self_cost=_aggregate("1.00"),
        descendants_cost=_aggregate("0.50"),
        total_cost=_aggregate("1.50"),
        descendant_ids=["child-1"],
        depth_reached=1,
        max_depth_applied=3,
        truncated=False,
        cost_source=cost_source,
    )


class _StubCostTreeService:
    """Returns a pre-baked ``CostTreeAggregate``; ignores the input."""

    def __init__(self, agg: CostTreeAggregate) -> None:
        self._agg = agg

    async def get_tree_aggregate(
        self, session_id: str, *, user_id: str, max_depth: int | None = None
    ) -> CostTreeAggregate:
        del session_id, user_id, max_depth
        return self._agg


def _stub_user() -> User:
    """Auth-bypass user — only used to satisfy ``CurrentUser`` typing.

    The route does NOT call ``session_service.get_session`` (that's the
    sibling ``/cost`` endpoint) — it forwards ``current_user.id`` straight
    into ``SessionCostTreeService.get_tree_aggregate`` for the scope
    check, so a synthetic User is fine here.
    """
    return User(
        id="test-user-1",
        username="tester",
        email="tester@example.com",
        password_hash="x",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def _build_client(
    *, cost_source: CostSource = CostSource.NONE,
) -> TestClient:
    """Compose a minimal FastAPI app and override the three deps the
    ``/cost/tree`` route reaches for.
    """
    app = FastAPI()
    app.include_router(cost_router, prefix="/api")

    agg = _tree_aggregate(cost_source=cost_source)
    stub_service = _StubCostTreeService(agg)

    async def _override_user() -> User:
        return _stub_user()

    async def _override_rate_limit() -> None:
        return None

    def _override_service() -> _StubCostTreeService:
        return stub_service

    app.dependency_overrides[get_current_user] = _override_user
    app.dependency_overrides[rate_limit_read] = _override_rate_limit
    app.dependency_overrides[get_session_cost_tree_service] = _override_service

    return TestClient(app)


# ── Tests ────────────────────────────────────────────────────────────────────


def test_cost_tree_response_includes_cost_source_field() -> None:
    """[P1-2] The /cost/tree wire payload must surface the ``cost_source``
    key — without this the frontend can't render attribution breakdowns
    and the §14.4 contract leaks only inside the backend.
    """
    client = _build_client(cost_source=CostSource.DIRECT)
    response = client.get("/api/sessions/root-session/cost/tree")

    assert response.status_code == 200, response.text
    body = response.json()
    assert "data" in body, (
        f"Response.success wraps payload in {{code, msg, data}}; got {body!r}"
    )
    data = body["data"]
    assert "cost_source" in data, (
        f"/cost/tree payload missing cost_source key; got keys={list(data)!r}"
    )


@pytest.mark.parametrize(
    "label,expected_wire",
    [
        (CostSource.NONE, "none"),
        (CostSource.DIRECT, "direct"),
        (CostSource.COORDINATOR_SUBAGENT, "coordinator_subagent"),
        (CostSource.RESEARCH_SUBAGENT, "research_subagent"),
        (CostSource.MIXED, "mixed"),
    ],
)
def test_cost_tree_response_cost_source_reflects_aggregate(
    label: CostSource, expected_wire: str,
) -> None:
    """End-to-end: whatever ``cost_source`` label the service returns on
    the aggregate must round-trip through the schema as the canonical
    wire string the frontend matches on.

    Pydantic v2 serialises ``StrEnum`` to its ``.value`` by default, so
    the schema needs no custom ``field_serializer`` — this test pins
    that default behaviour.
    """
    client = _build_client(cost_source=label)
    response = client.get("/api/sessions/root-session/cost/tree")

    assert response.status_code == 200, response.text
    payload = response.json()["data"]
    assert payload["cost_source"] == expected_wire, (
        f"cost_source wire string mismatch: got {payload['cost_source']!r}, "
        f"expected {expected_wire!r}"
    )


def test_cost_tree_response_cost_source_is_coordinator_subagent_when_set() -> None:
    """Spec-pointed canonical case: a ``CostTreeAggregate`` carrying
    ``CostSource.COORDINATOR_SUBAGENT`` must surface as the literal
    ``"coordinator_subagent"`` string (the value the frontend pattern-
    matches on for the coordinator-attribution badge).
    """
    client = _build_client(cost_source=CostSource.COORDINATOR_SUBAGENT)
    response = client.get("/api/sessions/root-session/cost/tree")

    assert response.status_code == 200
    data: dict[str, Any] = response.json()["data"]
    assert data["cost_source"] == "coordinator_subagent"
    # And the rest of the contract still surfaces (regression guard so
    # adding cost_source didn't accidentally drop a sibling field).
    assert data["session_id"] == "root-session"
    assert data["descendant_ids"] == ["child-1"]
    assert data["truncated"] is False
