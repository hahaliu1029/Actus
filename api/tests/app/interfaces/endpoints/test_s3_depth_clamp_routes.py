"""S3 PR-3: the /cost/tree and /children handlers clamp depth to the runtime
ceiling. Handlers are plain async functions — call them directly with fakes
(full FastAPI per-request DI resolution is integration/CI)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.config import SubagentLimitsConfig


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_cost_tree_route_clamps_depth_to_runtime_limit():
    from app.application.services.cost_aggregation_service import CostAggregationService
    from app.application.services.session_cost_tree_service import CostTreeAggregate
    from app.interfaces.endpoints.cost_routes import get_session_cost_tree

    zero = CostAggregationService.aggregate_rows([])
    captured: dict[str, int] = {}

    class _FakeService:
        async def get_tree_aggregate(self, session_id, *, user_id, max_depth):
            captured["max_depth"] = max_depth
            return CostTreeAggregate(
                session_id=session_id,
                self_cost=zero,
                descendants_cost=zero,
                total_cost=zero,
                descendant_ids=[],
                depth_reached=0,
                max_depth_applied=max_depth,
                truncated=False,
            )

    await get_session_cost_tree(
        session_id="root",
        current_user=SimpleNamespace(id="u1"),
        service=_FakeService(),
        depth=5,
        limits=SubagentLimitsConfig(max_subagent_depth=2),
    )
    assert captured["max_depth"] == 2  # clamped from 5


@pytest.mark.anyio
async def test_children_route_clamps_depth_to_runtime_limit():
    from app.domain.models.session import Session
    from app.interfaces.endpoints.session_routes import list_session_children

    captured: dict[str, int] = {}

    class _FakeRepo:
        async def find_by_id_for_user(self, session_id, *, user_id):
            return Session(id=session_id, user_id=user_id, worker_type="root")

        async def find_descendants(self, ancestor_id, *, user_id, max_depth, limit):
            captured["max_depth"] = max_depth
            return []

    await list_session_children(
        session_id="root",
        current_user=SimpleNamespace(id="u1"),
        repo=_FakeRepo(),
        depth=7,
        limits=SubagentLimitsConfig(max_subagent_depth=2),
    )
    assert captured["max_depth"] == 2  # clamped from 7


# ── In-process TestClient DI-wiring smoke (spec §8 / R2-B3) — no host DB ──────
# Proves FastAPI resolves the new Depends(get_subagent_limits) per-request AND
# the clamp honors an OVERRIDDEN runtime limit (not just the default). Mirrors
# tests/interfaces/endpoints/test_cost_routes.py (which runs without postgres).


def _stub_user():
    from app.domain.models.user import User, UserRole, UserStatus

    return User(
        id="u1",
        username="t",
        email="t@e.com",
        password_hash="x",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def test_cost_tree_endpoint_resolves_di_and_honors_override():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.application.services.cost_aggregation_service import CostAggregationService
    from app.application.services.session_cost_tree_service import CostTreeAggregate
    from app.interfaces.dependencies import rate_limit_read
    from app.interfaces.dependencies.auth import get_current_user
    from app.interfaces.endpoints.cost_routes import router as cost_router
    from app.interfaces.service_dependencies import (
        get_session_cost_tree_service,
        get_subagent_limits,
    )

    zero = CostAggregationService.aggregate_rows([])
    captured: dict[str, int] = {}

    class _StubService:
        async def get_tree_aggregate(self, session_id, *, user_id, max_depth):
            captured["max_depth"] = max_depth
            return CostTreeAggregate(
                session_id=session_id,
                self_cost=zero,
                descendants_cost=zero,
                total_cost=zero,
                descendant_ids=[],
                depth_reached=0,
                max_depth_applied=max_depth,
                truncated=False,
            )

    app = FastAPI()
    app.include_router(cost_router, prefix="/api")

    async def _user():
        return _stub_user()

    async def _rl():
        return None

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[rate_limit_read] = _rl
    app.dependency_overrides[get_session_cost_tree_service] = lambda: _StubService()
    app.dependency_overrides[get_subagent_limits] = lambda: SubagentLimitsConfig(
        max_subagent_depth=2
    )

    resp = TestClient(app).get("/api/sessions/root/cost/tree?depth=5")
    assert resp.status_code == 200, resp.text  # NOT 422 → Depends resolved
    assert captured["max_depth"] == 2  # clamp honored the OVERRIDDEN ceiling: min(5, 2)


def test_children_endpoint_resolves_di_and_honors_override():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from app.domain.models.session import Session
    from app.interfaces.dependencies import rate_limit_read
    from app.interfaces.dependencies.auth import get_current_user
    from app.interfaces.endpoints.session_routes import router as session_router
    from app.interfaces.service_dependencies import (
        get_session_repository,
        get_subagent_limits,
    )

    captured: dict[str, int] = {}

    class _StubRepo:
        async def find_by_id_for_user(self, session_id, *, user_id):
            return Session(id=session_id, user_id=user_id, worker_type="root")

        async def find_descendants(self, ancestor_id, *, user_id, max_depth, limit):
            captured["max_depth"] = max_depth
            return []

    app = FastAPI()
    app.include_router(session_router, prefix="/api")  # router has no router-level deps

    async def _user():
        return _stub_user()

    async def _rl():
        return None

    app.dependency_overrides[get_current_user] = _user
    app.dependency_overrides[rate_limit_read] = _rl
    app.dependency_overrides[get_session_repository] = lambda: _StubRepo()
    app.dependency_overrides[get_subagent_limits] = lambda: SubagentLimitsConfig(
        max_subagent_depth=2
    )

    resp = TestClient(app).get("/api/sessions/root/children?depth=7")
    assert resp.status_code == 200, resp.text  # NOT 422 → Depends resolved
    assert captured["max_depth"] == 2  # clamp honored the OVERRIDDEN ceiling: min(7, 2)
