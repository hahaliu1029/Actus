"""R5b-3 Codex round-3 LOW fix: endpoint 级验证 preflight 抛出的异常能被 FastAPI
exception handler 映射到明确 HTTP 状态码，且 preflight 失败路径不泄漏连接 lease。

覆盖：
- ``POST /sessions/{id}/chat`` with ``tool_confirmation`` → preflight 抛 ConflictError
  → response.status_code == 409，body 不是 SSE 帧（EventSourceResponse 未创建）
- preflight 抛 NotFoundError → 404 HTTP
- preflight 抛 ServiceUnavailableError（task 创建失败）→ 503 HTTP
- preflight 失败时 lease.release() 必须被调——否则连接配额泄漏

Pattern 复用自 ``test_session_sse_wire_format.py``：dependency_overrides +
monkeypatch session_routes.acquire_connection_limit + FastAPI TestClient。
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest

from app.application.errors.exceptions import (
    ConflictError,
    NotFoundError,
    ServiceUnavailableError,
)
from app.domain.models.session import Session, SessionStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies import rate_limit_chat
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.endpoints import session_routes
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_session_service,
    get_supervisor,
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


class _RecordingLease:
    """记录 start_heartbeat / release 调用。preflight 失败时必须 release。"""

    def __init__(self) -> None:
        self.started = False
        self.released = False

    def start_heartbeat(self) -> None:
        self.started = True

    async def release(self) -> None:
        self.released = True


class _PreflightFailingAgent:
    """mock ``AgentService.preflight_resume_tool_confirmation`` 抛指定异常。"""

    def __init__(self, *, exc: BaseException) -> None:
        self._exc = exc
        self.preflight_calls = 0

    async def preflight_resume_tool_confirmation(self, **kwargs: Any):
        self.preflight_calls += 1
        raise self._exc

    async def chat(self, **kwargs: Any):  # noqa: ASYNC101 — generator stub
        # 不应被调用；定义以满足接口形状
        if False:
            yield  # type: ignore[unreachable]


class _NoConflictScope:
    is_conflict = False
    current_owner = None


class _NoConflictSupervisor:
    @asynccontextmanager
    async def subscriber_scope(self, **kwargs: Any):
        yield _NoConflictScope()


class _AllowSessionService:
    async def get_session(self, **kwargs: Any) -> object:
        return object()


class _SuspendedBackgroundSessionService:
    async def get_session(self, **kwargs: Any) -> Session:
        return Session(
            id="s1",
            user_id="test-user",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="suspended",
        )


class _UnexpectedChatAgent:
    def __init__(self) -> None:
        self.chat_called = False

    async def chat(self, **kwargs: Any):  # noqa: ASYNC101 — generator stub
        self.chat_called = True
        if False:
            yield  # type: ignore[unreachable]


def _tool_confirmation_payload() -> dict:
    return {
        "message": None,
        "tool_confirmation": {
            "action": "approve",
            "scope": "session",
            "tool_call_id": "tc-endpoint-1",
        },
    }


async def _send_chat_with_confirmation(
    monkeypatch: pytest.MonkeyPatch,
    agent: _PreflightFailingAgent,
    lease: _RecordingLease,
) -> httpx.Response:
    async def _fake_acquire(**kwargs: Any) -> _RecordingLease:
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire,
    )
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: agent
    app.dependency_overrides[get_session_service] = lambda: _AllowSessionService()
    app.dependency_overrides[get_supervisor] = lambda: _NoConflictSupervisor()
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            return await client.post(
                "/api/sessions/s1/chat",
                json=_tool_confirmation_payload(),
            )
    finally:
        app.dependency_overrides.clear()


# ---------------- 409 Conflict ----------------


async def test_preflight_conflict_error_maps_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight 抛 ConflictError → response.status_code == 409；SSE 未建连。"""
    agent = _PreflightFailingAgent(exc=ConflictError("工具确认[tc-endpoint-1]已被处理"))
    lease = _RecordingLease()
    response = await _send_chat_with_confirmation(monkeypatch, agent, lease)

    assert response.status_code == 409
    # body 不是 SSE 帧（SSE 帧以 "event:" / "data:" 开头）
    assert "event:" not in response.text[:200]
    # 连接配额 lease 必须在 preflight 失败时 release
    assert lease.released, "lease.release() 未被调用——连接配额泄漏"
    assert agent.preflight_calls == 1


# ---------------- 404 Not Found ----------------


async def test_preflight_not_found_error_maps_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight 抛 NotFoundError（confirmation 已过期）→ 404。"""
    agent = _PreflightFailingAgent(
        exc=NotFoundError("工具确认请求[tc-endpoint-1]不存在或已过期"),
    )
    lease = _RecordingLease()
    response = await _send_chat_with_confirmation(monkeypatch, agent, lease)

    assert response.status_code == 404
    assert lease.released


# ---------------- 503 Service Unavailable（Codex MEDIUM 回归） ----------------


async def test_preflight_service_unavailable_maps_to_503(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preflight task 创建失败抛 ServiceUnavailableError → 503（不是裸 500）。"""
    agent = _PreflightFailingAgent(
        exc=ServiceUnavailableError("会话[s1]创建任务失败，请稍后重试"),
    )
    lease = _RecordingLease()
    response = await _send_chat_with_confirmation(monkeypatch, agent, lease)

    assert response.status_code == 503
    assert lease.released


# ---------------- Codex round-6 HIGH: winner cleanup 后的 late-duplicate HTTP 409 ----------------


async def test_preflight_late_duplicate_after_winner_cleanup_maps_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """I4 late-duplicate endpoint 合同：winner 已完成并 cleanup 后，再次
    ``POST /chat`` 同 confirmation_id → ConflictError (409)，不是 404。

    对应 agent_service 层的 ``test_preflight_late_duplicate_after_cleanup_returns_409``：
    preflight 查 grants 表命中历史 grant → 返 409 带 /events 重连指引。
    """
    agent = _PreflightFailingAgent(
        exc=ConflictError(
            "工具确认[tc-endpoint-1]已被处理完成（grant 已持久）；"
            "请通过 /events?since=<last_event_id> 重连 SSE 复播结果"
        ),
    )
    lease = _RecordingLease()
    response = await _send_chat_with_confirmation(monkeypatch, agent, lease)

    assert response.status_code == 409
    # body 应带明确的 reconnect 提示（让前端能走 /events SSE 复播）
    assert "/events" in response.text or "重连" in response.text
    # 不误报 SSE 200
    assert "event:" not in response.text[:200]
    assert lease.released


async def test_suspended_background_message_preflight_maps_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _UnexpectedChatAgent()
    lease = _RecordingLease()
    acquire_calls = 0

    async def _fake_acquire(**kwargs: Any) -> _RecordingLease:
        nonlocal acquire_calls
        acquire_calls += 1
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire,
    )
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: agent
    app.dependency_overrides[get_session_service] = (
        lambda: _SuspendedBackgroundSessionService()
    )
    app.dependency_overrides[get_supervisor] = lambda: _NoConflictSupervisor()
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/sessions/s1/chat",
                json={"message": "continue"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert "event:" not in response.text[:200]
    assert agent.chat_called is False
    assert acquire_calls == 0


async def test_suspended_background_attachments_only_preflight_maps_to_409(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _UnexpectedChatAgent()
    lease = _RecordingLease()
    acquire_calls = 0

    async def _fake_acquire(**kwargs: Any) -> _RecordingLease:
        nonlocal acquire_calls
        acquire_calls += 1
        return lease

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire,
    )
    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: agent
    app.dependency_overrides[get_session_service] = (
        lambda: _SuspendedBackgroundSessionService()
    )
    app.dependency_overrides[get_supervisor] = lambda: _NoConflictSupervisor()
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/sessions/s1/chat",
                json={"attachments": ["file-1"]},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 409
    assert "event:" not in response.text[:200]
    assert agent.chat_called is False
    assert acquire_calls == 0
