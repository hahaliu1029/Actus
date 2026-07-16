"""SPM Task 20 — GET /sessions/{id} 下发 sandbox_mode 部署常量.

沿用 ``test_session_events_route.py`` 的 ASGITransport + dependency_overrides
harness。sandbox_mode 值在路由组装处取自 ``get_settings().sandbox_provision_mode``；
第二个用例 monkeypatch 该来源证明字段非硬编码。
"""
from __future__ import annotations

from typing import Any

import httpx
import pytest
from app.domain.models.session import Session
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read
from app.interfaces.endpoints import session_routes
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_session_service,
)
from app.main import app

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


async def _noop_rate_limit() -> None:
    return None


class _FakeSessionService:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_session(self, **kwargs: Any) -> Session:
        return self._session


class _FakeAgentService:
    async def build_supervisor_snapshot(self, session: Session) -> None:  # pragma: no cover - foreground session never calls
        return None


async def _get(url: str, *, session: Session) -> httpx.Response:
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_session_service] = lambda: _FakeSessionService(session)
    app.dependency_overrides[get_agent_service] = lambda: _FakeAgentService()
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            return await client.get(url)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_read, None)


async def test_get_session_payload_includes_sandbox_mode_default() -> None:
    """GET 详情下发 sandbox_mode；PR-2 阶段部署常量恒为 'always'."""
    session = Session(id="s1", title="demo", user_id="test-user")

    response = await _get("/api/sessions/s1", session=session)
    body = response.json()

    assert response.status_code == 200
    assert body["code"] == 200
    assert body["data"]["sandbox_mode"] == "always"


async def test_get_session_sandbox_mode_sourced_from_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sandbox_mode 取自 get_settings().sandbox_provision_mode（非硬编码）。

    monkeypatch 路由模块的 ``get_settings`` 绕过 config ALLOWED 校验，
    直接证明字段随部署常量变化。
    """

    class _Stub:
        sandbox_provision_mode = "on_demand"

    monkeypatch.setattr(session_routes, "get_settings", lambda: _Stub())

    session = Session(id="s1", title="demo", user_id="test-user")

    response = await _get("/api/sessions/s1", session=session)
    body = response.json()

    assert response.status_code == 200
    assert body["data"]["sandbox_mode"] == "on_demand"
