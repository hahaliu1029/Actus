from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


async def test_multitab_tool_confirmation_conflict_skips_preflight_and_chat(
    asgi_client,
    sample_session,
    sample_user_token,
    agent_service_with_redis,
    app,
    redis_client,
    monkeypatch,
):
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_chat

    async def _noop_rate_limit() -> None:
        return None

    sid = sample_session.id
    owner = f"{sample_session.user_id}:tab1"
    tab2 = "tab2"
    await redis_client.set(f"supervisor:owner:{sid}", owner, ex=10)

    preflight = AsyncMock(return_value=object())
    chat = MagicMock()
    drive = MagicMock()
    monkeypatch.setattr(
        agent_service_with_redis,
        "preflight_resume_tool_confirmation",
        preflight,
    )
    monkeypatch.setattr(agent_service_with_redis, "chat", chat)
    monkeypatch.setattr(agent_service_with_redis, "drive_resume_tool_confirmation", drive)

    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit
    try:
        resp = await asgi_client.post(
            f"/api/sessions/{sid}/chat",
            json={
                "tool_confirmation": {
                    "action": "approve",
                    "scope": "once",
                    "tool_call_id": "tool-call-conflict",
                }
            },
            headers={
                "Authorization": f"Bearer {sample_user_token}",
                "X-Connection-Id": tab2,
            },
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)
        app.dependency_overrides.pop(rate_limit_chat, None)

    assert resp.status_code == 200
    assert "owner_conflict" in resp.text
    preflight.assert_not_awaited()
    chat.assert_not_called()
    drive.assert_not_called()


async def test_cross_user_chat_does_not_enter_subscriber_scope(
    asgi_client,
    sample_session,
    other_user_token,
    agent_service_with_redis,
    app,
    redis_client,
):
    from app.infrastructure.storage.redis import get_redis
    from app.interfaces.dependencies import rate_limit_chat

    async def _noop_rate_limit() -> None:
        return None

    sid = sample_session.id
    app.dependency_overrides[get_redis] = lambda: redis_client
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit
    try:
        resp = await asgi_client.post(
            f"/api/sessions/{sid}/chat",
            json={"message": "hello"},
            headers={
                "Authorization": f"Bearer {other_user_token}",
                "X-Connection-Id": "other-tab",
            },
        )
    finally:
        app.dependency_overrides.pop(get_redis, None)
        app.dependency_overrides.pop(rate_limit_chat, None)

    assert resp.status_code == 403
    assert await redis_client.exists(f"supervisor:owner:{sid}") == 0
    assert await redis_client.exists(f"supervisor:hot:{sid}") == 0


async def test_subscriber_scope_exit_failure_still_releases_connection_lease(
    sample_session,
    sample_user,
    agent_service_with_redis,
    monkeypatch,
):
    from app.interfaces.endpoints import session_routes
    from app.interfaces.schemas.session import ChatRequest

    class FakeLease:
        def __init__(self) -> None:
            self.released = False

        def start_heartbeat(self) -> None:
            return None

        async def release(self) -> None:
            self.released = True

    class FailingSubscriberScope:
        async def __aenter__(self):
            return SimpleNamespace(is_conflict=True, current_owner="owner-tab")

        async def __aexit__(self, exc_type, exc, tb):
            raise RuntimeError("subscriber cleanup failed")

    class FakeSupervisor:
        def subscriber_scope(self, *, session_id: str, connection_id: str):
            return FailingSubscriberScope()

    class FakeSessionService:
        async def get_session(self, **_kwargs):
            return object()

    lease = FakeLease()

    async def _fake_acquire_connection_limit(**_kwargs):
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire_connection_limit,
    )

    response = await session_routes.chat(
        session_id=sample_session.id,
        request=ChatRequest(message="hello"),
        fastapi_request=SimpleNamespace(headers={}),
        current_user=sample_user.to_domain(),
        agent_service=agent_service_with_redis,
        session_service=FakeSessionService(),
        supervisor=FakeSupervisor(),
        redis_client=SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="subscriber cleanup failed"):
        async for _event in response.body_iterator:
            pass

    assert lease.released is True
