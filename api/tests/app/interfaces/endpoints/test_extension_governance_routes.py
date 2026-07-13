"""D1a Task 20 §9.2 治理路由端点测试（fake service + ASGITransport）。

只验 HTTP 面：off 门（GET /governance 200 固定零 / 其余 409 governance_disabled）、
on 透传、409 族逐码断言（revision_conflict/invalid_state/missing_observation/
operation_pending）、CAS 必填字段 422、kind 非法 422、非 Admin 403。service 内部
语义已在 test_extension_governance_service.py 单测锁定。

双 client fixture 对齐 test_runtime_extension_routes.py：teardown 精确 pop 自己的
override key，禁用 clear()。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.application.services.extension_governance_service import (
    ExtensionGovernanceService,
    ItemOutcome,
    RefreshResult,
)
from app.domain.external.extension_admission import AuditPage
from app.domain.models.extension_governance import (
    InvalidStateTransitionError,
    MissingObservationError,
    OperationPendingError,
    RevisionConflictError,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.service_dependencies import get_extension_governance_service
from app.main import app

pytestmark = pytest.mark.anyio

_BASE = "/api/v2/extensions"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _admin_user() -> User:
    return User(id="u1", username="admin",
                role=UserRole.SUPER_ADMIN, status=UserStatus.ACTIVE)


def _normal_user() -> User:
    return User(id="u1", username="tester",
                role=UserRole.USER, status=UserStatus.ACTIVE)


def _make_service() -> MagicMock:
    svc = MagicMock(spec=ExtensionGovernanceService)
    svc.summary = AsyncMock(return_value={
        "mode": "enforce", "unpinned_count": 2,
        "missing_observation_count": 1, "quarantined_count": 0})
    svc.quarantine = AsyncMock(return_value=5)
    svc.reapprove = AsyncMock(return_value=6)
    svc.set_enabled = AsyncMock(return_value=7)
    svc.refresh_observation = AsyncMock(
        return_value=RefreshResult(outcome="refreshed", row_revision=8, surface_summary=None))
    svc.refresh_observations_batch = AsyncMock(return_value=[
        ItemOutcome(kind="mcp", ext_id="a", outcome="refreshed", row_revision=8)])
    svc.approve_pins = AsyncMock(return_value=[
        ItemOutcome(kind="mcp", ext_id="a", outcome="pinned")])
    svc.list_audit = AsyncMock(return_value=AuditPage(entries=[], next_cursor=None))
    return svc


@pytest.fixture
def client_on():
    fake = _make_service()
    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_extension_governance_service] = lambda: fake
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    client.fake = fake  # type: ignore[attr-defined]
    try:
        yield client
    finally:
        for dep in (get_current_user, get_extension_governance_service):
            app.dependency_overrides.pop(dep, None)


@pytest.fixture
def client_off():
    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_extension_governance_service] = lambda: None
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        for dep in (get_current_user, get_extension_governance_service):
            app.dependency_overrides.pop(dep, None)


@pytest.fixture
def client_user():
    fake = _make_service()
    app.dependency_overrides[get_current_user] = _normal_user
    app.dependency_overrides[get_extension_governance_service] = lambda: fake
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    try:
        yield client
    finally:
        for dep in (get_current_user, get_extension_governance_service):
            app.dependency_overrides.pop(dep, None)


# ------------------------------------------------- off 门 --------------------


async def test_off_governance_returns_literal_zeros(client_off):
    resp = await client_off.get(f"{_BASE}/governance")
    assert resp.status_code == 200
    # R4-11：字面量零 + 零 registry 读（service=None → 不触碰任何 port）
    assert resp.json() == {
        "mode": "off", "unpinned_count": 0,
        "missing_observation_count": 0, "quarantined_count": 0}


async def test_off_other_endpoints_409_governance_disabled(client_off):
    for method, path, body in (
        ("post", f"{_BASE}/mcp/a/refresh-observation", None),
        ("post", f"{_BASE}/refresh-observations", {"all": True}),
        ("post", f"{_BASE}/mcp/a/quarantine", {"expected_row_revision": 1}),
        ("post", f"{_BASE}/mcp/a/reapprove", {"expected_row_revision": 1}),
        ("post", f"{_BASE}/mcp/a/governance-disable", {"expected_row_revision": 1}),
        ("post", f"{_BASE}/mcp/a/governance-enable", {"expected_row_revision": 1}),
        ("post", f"{_BASE}/approve-pins", {"all": True}),
        ("get", f"{_BASE}/audit", None),
    ):
        kwargs = {"json": body} if method == "post" else {}
        resp = await getattr(client_off, method)(path, **kwargs)
        assert resp.status_code == 409, path
        assert resp.json().get("code") == "governance_disabled", path


# ------------------------------------------------- on 透传 -------------------


async def test_on_governance_summary_passthrough(client_on):
    resp = await client_on.get(f"{_BASE}/governance")
    assert resp.status_code == 200
    assert resp.json()["mode"] == "enforce"
    client_on.fake.summary.assert_awaited_once()


async def test_on_refresh_observation_passthrough(client_on):
    resp = await client_on.post(f"{_BASE}/mcp/srv/refresh-observation")
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "refreshed"
    assert body["row_revision"] == 8
    client_on.fake.refresh_observation.assert_awaited_once_with("mcp", "srv")


async def test_on_refresh_batch_passthrough(client_on):
    resp = await client_on.post(f"{_BASE}/refresh-observations", json={"all": True})
    assert resp.status_code == 200
    assert resp.json()["items"][0]["outcome"] == "refreshed"


async def test_on_quarantine_passthrough(client_on):
    resp = await client_on.post(
        f"{_BASE}/mcp/srv/quarantine", json={"expected_row_revision": 3, "note": "bad"})
    assert resp.status_code == 200
    assert resp.json()["row_revision"] == 5
    client_on.fake.quarantine.assert_awaited_once_with(
        "mcp", "srv", expected_row_revision=3, actor_id="u1", note="bad")


async def test_on_reapprove_passthrough(client_on):
    resp = await client_on.post(f"{_BASE}/mcp/srv/reapprove", json={"expected_row_revision": 3})
    assert resp.status_code == 200
    assert resp.json()["row_revision"] == 6


async def test_on_enable_disable_passthrough(client_on):
    resp = await client_on.post(
        f"{_BASE}/mcp/srv/governance-enable", json={"expected_row_revision": 3})
    assert resp.status_code == 200
    client_on.fake.set_enabled.assert_awaited_with(
        "mcp", "srv", enabled=True, expected_row_revision=3, actor_id="u1")
    resp = await client_on.post(
        f"{_BASE}/mcp/srv/governance-disable", json={"expected_row_revision": 3})
    assert resp.status_code == 200
    client_on.fake.set_enabled.assert_awaited_with(
        "mcp", "srv", enabled=False, expected_row_revision=3, actor_id="u1")


async def test_on_approve_pins_passthrough(client_on):
    resp = await client_on.post(f"{_BASE}/approve-pins", json={"all": True})
    assert resp.status_code == 200
    assert resp.json()["items"][0]["outcome"] == "pinned"


async def test_on_audit_passthrough(client_on):
    resp = await client_on.get(f"{_BASE}/audit", params={"event": "quarantined"})
    assert resp.status_code == 200
    assert resp.json() == {"entries": [], "next_cursor": None}
    client_on.fake.list_audit.assert_awaited_once()


# ------------------------------------------------- 409 族逐码 ----------------


@pytest.mark.parametrize("exc,code", [
    (RevisionConflictError("x"), "revision_conflict"),
    (InvalidStateTransitionError("x"), "invalid_state"),
    (MissingObservationError("x"), "missing_observation"),
    (OperationPendingError("x"), "operation_pending"),
])
async def test_on_governance_error_family_maps_409(client_on, exc, code):
    client_on.fake.reapprove = AsyncMock(side_effect=exc)
    resp = await client_on.post(f"{_BASE}/mcp/srv/reapprove", json={"expected_row_revision": 1})
    assert resp.status_code == 409
    assert resp.json()["code"] == code


# ------------------------------------------------- 422 / 403 ----------------


async def test_quarantine_missing_revision_422(client_on):
    resp = await client_on.post(f"{_BASE}/mcp/srv/quarantine", json={"note": "x"})
    assert resp.status_code == 422


async def test_invalid_kind_422(client_on):
    resp = await client_on.post(
        f"{_BASE}/bogus/srv/reapprove", json={"expected_row_revision": 1})
    assert resp.status_code == 422


async def test_refresh_batch_invalid_kind_422(client_on):
    """批量 /refresh-observations 非法 kind → 422 parse-time（与单项路由闭词表一致，
    不得 200 skipped_invalid_state 掩盖非法输入）。"""
    resp = await client_on.post(
        f"{_BASE}/refresh-observations",
        json={"items": [{"kind": "bogus", "ext_id": "a"}]})
    assert resp.status_code == 422
    client_on.fake.refresh_observations_batch.assert_not_awaited()


async def test_approve_pins_batch_invalid_kind_422(client_on):
    """批量 /approve-pins 非法 kind → 422 parse-time（同上闭词表一致性）。"""
    resp = await client_on.post(
        f"{_BASE}/approve-pins",
        json={"items": [{"kind": "bogus", "ext_id": "a"}]})
    assert resp.status_code == 422
    client_on.fake.approve_pins.assert_not_awaited()


async def test_non_admin_403(client_user):
    resp = await client_user.post(f"{_BASE}/mcp/srv/reapprove", json={"expected_row_revision": 1})
    assert resp.status_code == 403
