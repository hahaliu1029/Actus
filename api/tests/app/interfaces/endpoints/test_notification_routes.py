"""Integration tests for /v2/notifications routes.

Override ``get_memory_system_notification_repository`` with an AsyncMock
so we verify wiring + response shape without touching real Postgres.
DI-level commit-on-success path is exercised indirectly via
test_db_memory_system_notification_repository_integration.py.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from app.domain.models.memory_system_notification import (
    MemorySystemNotification,
)
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import (
    rate_limit_read,
    rate_limit_write,
)
from app.interfaces.service_dependencies import (
    get_memory_system_notification_repository,
)
from app.main import app

from tests.conftest import TEST_USER_ID_FIXED

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_user() -> User:
    return User(
        id=TEST_USER_ID_FIXED,
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


def _make_notification(
    *,
    nid: str = "n-1",
    event_type: str = "memory_gate_paused",
    payload: dict | None = None,
) -> MemorySystemNotification:
    now = datetime.now(tz=timezone.utc)
    return MemorySystemNotification(
        id=nid,
        user_id=str(TEST_USER_ID_FIXED),
        event_type=event_type,
        payload=payload or {"consecutive_failures": 3},
        created_at=now,
        expires_at=now + timedelta(days=30),
    )


@pytest.fixture
def mock_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.list_unread = AsyncMock(return_value=[])
    repo.count_unread = AsyncMock(return_value=0)
    repo.mark_read = AsyncMock(return_value=False)
    return repo


async def _noop_rate_limit() -> None:
    return None


@pytest.fixture
def client_app(mock_repo: AsyncMock):
    # DI factory in service_dependencies yields the repo and commits on
    # success; for tests we override with a plain callable that returns
    # the mock — no commit semantics to test here (integration test
    # covers that).
    app.dependency_overrides[get_memory_system_notification_repository] = (
        lambda: mock_repo
    )
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        yield app
    finally:
        app.dependency_overrides.pop(
            get_memory_system_notification_repository, None
        )
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(rate_limit_read, None)
        app.dependency_overrides.pop(rate_limit_write, None)


async def _request(
    client_app,
    method: str,
    url: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=client_app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.request(method, url, json=json, params=params)


# --- GET /unread ------------------------------------------------------------


async def test_list_unread_empty_returns_200(
    client_app, mock_repo: AsyncMock
) -> None:
    res = await _request(client_app, "GET", "/api/v2/notifications/unread")
    assert res.status_code == 200
    body = res.json()
    assert body["code"] == 200
    assert body["data"]["unread_count"] == 0
    assert body["data"]["items"] == []
    # limit defaults to 50
    mock_repo.list_unread.assert_awaited_once_with(
        TEST_USER_ID_FIXED, limit=50
    )


async def test_list_unread_passes_limit_query_param(
    client_app, mock_repo: AsyncMock
) -> None:
    await _request(
        client_app, "GET", "/api/v2/notifications/unread", params={"limit": 10}
    )
    mock_repo.list_unread.assert_awaited_once_with(TEST_USER_ID_FIXED, limit=10)


async def test_list_unread_rejects_limit_above_50(client_app) -> None:
    res = await _request(
        client_app, "GET", "/api/v2/notifications/unread", params={"limit": 51}
    )
    assert res.status_code == 422


async def test_list_unread_returns_items(
    client_app, mock_repo: AsyncMock
) -> None:
    mock_repo.list_unread.return_value = [
        _make_notification(nid="n-1"),
        _make_notification(nid="n-2", event_type="quota_exceeded", payload={}),
    ]
    mock_repo.count_unread.return_value = 2
    res = await _request(client_app, "GET", "/api/v2/notifications/unread")
    assert res.status_code == 200
    body = res.json()
    assert body["data"]["unread_count"] == 2
    assert [it["id"] for it in body["data"]["items"]] == ["n-1", "n-2"]
    assert body["data"]["items"][0]["event_type"] == "memory_gate_paused"
    assert body["data"]["items"][1]["event_type"] == "quota_exceeded"


async def test_list_unread_count_reflects_total_not_slice(
    client_app, mock_repo: AsyncMock
) -> None:
    """Regression: ``unread_count`` must be the **total** unread (via
    count_unread), not ``len(items)``. Otherwise the frontend badge
    silently caps at ``limit`` and looks "stuck" at e.g. 20 forever
    when the true backlog is 75."""
    mock_repo.list_unread.return_value = [
        _make_notification(nid=f"n-{i}") for i in range(20)
    ]
    mock_repo.count_unread.return_value = 75
    res = await _request(
        client_app, "GET", "/api/v2/notifications/unread", params={"limit": 20}
    )
    body = res.json()
    assert len(body["data"]["items"]) == 20
    assert body["data"]["unread_count"] == 75


# --- POST /{id}/mark-read ---------------------------------------------------


async def test_mark_read_true_on_first_success(
    client_app, mock_repo: AsyncMock
) -> None:
    mock_repo.mark_read.return_value = True
    res = await _request(
        client_app, "POST", "/api/v2/notifications/n-1/mark-read"
    )
    assert res.status_code == 200
    body = res.json()
    assert body["data"]["marked_read"] is True
    mock_repo.mark_read.assert_awaited_once_with(
        notification_id="n-1",
        user_id=TEST_USER_ID_FIXED,
    )


async def test_mark_read_false_on_repeat_or_missing(
    client_app, mock_repo: AsyncMock
) -> None:
    # Idempotent: repo returns False for {already read, not found, foreign
    # user}. Endpoint surfaces this as marked_read=false instead of 404/409
    # to keep UX simple (users double-clicking shouldn't trigger error toasts).
    mock_repo.mark_read.return_value = False
    res = await _request(
        client_app, "POST", "/api/v2/notifications/n-missing/mark-read"
    )
    assert res.status_code == 200
    assert res.json()["data"]["marked_read"] is False
