"""B3-core PR-1: /sessions/{id}/events query param `since_seq`.

Spec v3 §3.3 — since_seq precedence + last_seq response field.
Anchor C-Wire-4 (in tests/integration/test_supervisor_wire_contract.py) covers
the integration variant; this file covers the route-level wiring with a fake
agent_service.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.domain.models.session import SessionStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read
from app.interfaces.service_dependencies import get_agent_service
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


class _FakeAgentService:
    def __init__(self, *, last_seq: int = 0) -> None:
        self.calls: list[dict[str, Any]] = []
        self._last_seq = last_seq

    async def get_events_since(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {
            "events": [],
            "session_status": SessionStatus.RUNNING,
            "has_more": False,
            "last_seq": self._last_seq,
            "supervisor_snapshot": None,
        }


async def _request(url: str, *, fake: _FakeAgentService) -> httpx.Response:
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: fake
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.get(url)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(rate_limit_read, None)


async def test_get_events_since_accepts_since_seq_query() -> None:
    """Endpoint accepts ?since_seq=N — passes through to agent_service + returns last_seq."""
    fake = _FakeAgentService(last_seq=5)
    resp = await _request("/api/sessions/s1/events?since_seq=5", fake=fake)
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["last_seq"] == 5
    assert body["supervisor_snapshot"] is None
    # Service received since_seq=5
    assert fake.calls[0]["since_seq"] == 5
    assert fake.calls[0]["since_event_id"] is None


async def test_get_events_since_seq_takes_precedence() -> None:
    """Both since and since_seq → endpoint forwards both; service decides precedence."""
    fake = _FakeAgentService(last_seq=7)
    resp = await _request(
        "/api/sessions/s1/events?since=foo-bar&since_seq=7", fake=fake
    )
    assert resp.status_code == 200
    call = fake.calls[0]
    assert call["since_seq"] == 7
    assert call["since_event_id"] == "foo-bar"


async def test_get_events_since_legacy_path_no_since_seq() -> None:
    """Legacy clients without since_seq still work — defaults to None + last_seq=0."""
    fake = _FakeAgentService(last_seq=0)
    resp = await _request("/api/sessions/s1/events", fake=fake)
    assert resp.status_code == 200
    body = resp.json()["data"]
    assert body["last_seq"] == 0
    assert "supervisor_snapshot" in body
    call = fake.calls[0]
    assert call["since_seq"] is None
    assert call["since_event_id"] is None
