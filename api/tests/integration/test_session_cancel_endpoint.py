from __future__ import annotations

from types import SimpleNamespace
import uuid
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def supervisor_route_overrides(app, redis_client, agent_service_with_redis):
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_chat, rate_limit_write

    async def _noop_rate_limit() -> None:
        return None

    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_redis, None)
        app.dependency_overrides.pop(rate_limit_write, None)
        app.dependency_overrides.pop(rate_limit_chat, None)


async def test_cancel_session_owner_returns_200_and_stamps_hot_hash(
    asgi_client,
    sample_session,
    sample_user_token,
    redis_client,
    agent_service_with_redis,
    supervisor_route_overrides,
    monkeypatch,
):
    stop_session = AsyncMock()
    monkeypatch.setattr(agent_service_with_redis, "stop_session", stop_session)

    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        json={"reason": "user_cancel"},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["data"] == {
        "status": "cancel_requested",
        "session_id": sample_session.id,
        "reason": "user_cancel",
    }
    stop_session.assert_awaited_once_with(
        session_id=sample_session.id,
        user_id=str(sample_session.user_id),
        is_admin=False,
    )
    hot_key = f"supervisor:hot:{sample_session.id}"
    assert await redis_client.hget(hot_key, "cancellation_pending") == "1"
    assert await redis_client.hget(hot_key, "pending_terminal_reason") == "user_cancel"


async def test_cancel_session_cross_user_returns_403(
    asgi_client,
    sample_session,
    other_user_token,
    supervisor_route_overrides,
):
    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        json={"reason": "user_cancel"},
        headers={"Authorization": f"Bearer {other_user_token}"},
    )

    assert resp.status_code == 403
    assert resp.json()["code"] == 403


async def test_cancel_session_admin_cross_owner_returns_403(
    app,
    asgi_client,
    sample_session,
    sample_user_token,
    supervisor_route_overrides,
):
    from app.interfaces.dependencies.auth import get_current_user

    async def _admin_user():
        return SimpleNamespace(
            id=f"admin-{uuid.uuid4()}",
            username="admin",
            is_admin=lambda: True,
        )

    app.dependency_overrides[get_current_user] = _admin_user
    try:
        resp = await asgi_client.post(
            f"/api/sessions/{sample_session.id}/cancel",
            json={"reason": "user_cancel"},
            headers={"Authorization": f"Bearer {sample_user_token}"},
        )
    finally:
        app.dependency_overrides.pop(get_current_user, None)

    assert resp.status_code == 403
    assert resp.json()["code"] == 403


async def test_legacy_stop_route_uses_cancel_hot_hash_contract(
    asgi_client,
    sample_session,
    sample_user_token,
    redis_client,
    agent_service_with_redis,
    supervisor_route_overrides,
    monkeypatch,
):
    stop_session = AsyncMock()
    monkeypatch.setattr(agent_service_with_redis, "stop_session", stop_session)

    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/stop",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 200
    stop_session.assert_awaited_once_with(
        session_id=sample_session.id,
        user_id=str(sample_session.user_id),
        is_admin=False,
    )
    hot_key = f"supervisor:hot:{sample_session.id}"
    assert await redis_client.hget(hot_key, "cancellation_pending") == "1"
    assert await redis_client.hget(hot_key, "pending_terminal_reason") == "user_cancel"


async def test_cancel_session_missing_returns_404(
    asgi_client,
    sample_user_token,
    supervisor_route_overrides,
):
    missing_session_id = f"missing-{uuid.uuid4()}"
    resp = await asgi_client.post(
        f"/api/sessions/{missing_session_id}/cancel",
        json={"reason": "user_cancel"},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 404
    assert resp.json()["code"] == 404


async def test_cancel_session_unauthenticated_returns_401(
    asgi_client,
    sample_session,
    supervisor_route_overrides,
):
    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        json={"reason": "user_cancel"},
    )

    assert resp.status_code == 401
    assert resp.json()["code"] == 401


async def test_cancel_session_empty_body_defaults_to_user_cancel(
    asgi_client,
    sample_session,
    sample_user_token,
    agent_service_with_redis,
    supervisor_route_overrides,
    monkeypatch,
):
    stop_session = AsyncMock()
    monkeypatch.setattr(agent_service_with_redis, "stop_session", stop_session)

    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 200
    assert resp.json()["data"]["reason"] == "user_cancel"


async def test_cancel_session_invalid_reason_returns_422(
    asgi_client,
    sample_session,
    sample_user_token,
    supervisor_route_overrides,
):
    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        json={"reason": "not_supported"},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 422


async def test_cancel_session_extra_field_returns_422(
    asgi_client,
    sample_session,
    sample_user_token,
    supervisor_route_overrides,
):
    resp = await asgi_client.post(
        f"/api/sessions/{sample_session.id}/cancel",
        json={"reason": "user_cancel", "unexpected": True},
        headers={"Authorization": f"Bearer {sample_user_token}"},
    )

    assert resp.status_code == 422
