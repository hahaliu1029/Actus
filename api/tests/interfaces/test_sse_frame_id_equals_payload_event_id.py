"""CS3 wire contract: SSE frame `id:` field == payload `event_id`.

N2 landing 后摘 xfail. chat endpoint 每一帧 SSE 必须满足:
    frame.id == json.loads(frame.data)["event_id"]

CS3 ADR 硬契约 (docs/adr/CS3-tool-event-envelope-v1.md).

Spec: docs/superpowers/specs/2026-04-17-n2-sse-transport-repair-design.md §2, §7
SSE endpoint: POST /api/sessions/{session_id}/chat
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any, AsyncGenerator

import httpx
import pytest

from app.domain.models.event import BaseEvent, MessageEvent, ToolEvent, ToolEventStatus
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


class _FakeLease:
    def start_heartbeat(self) -> None:
        pass

    async def release(self) -> None:
        pass


class _TwoFrameAgentService:
    """yield 两帧: 一条 message + 一条 ToolEvent. 两者的 event.id 走 EventMapper 后
    payload.event_id 相等, 验证 invariant 对多种 event 类型成立."""

    async def chat(self, **kwargs: Any) -> AsyncGenerator[BaseEvent, None]:
        msg = MessageEvent(
            role="assistant",
            message="hello",
        )
        msg.id = "1000-0"
        yield msg

        tool = ToolEvent(
            tool_call_id="c_test",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
            function_result=None,
            artifact={
                "tool_call_id": "c_test",
                "tool_name": "shell_execute",
                "tool_source": {
                    "source": "native",
                    "category": "shell",
                    "canonical_name": "shell_execute",
                },
                "outcome": {"variant": "allow_success", "content": "ok", "data": None},
            },
        )
        tool.id = "1000-1"
        yield tool


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


def _parse_sse_frames(body: str) -> list[dict[str, str]]:
    frames: list[dict[str, str]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        frame: dict[str, str] = {}
        for line in block.split("\n"):
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            frame[key.strip()] = value.lstrip(" ").rstrip("\r")
        if frame.get("data"):
            frames.append(frame)
    return frames


async def test_chat_sse_frame_id_equals_payload_event_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/sessions/{id}/chat 的每一帧都满足 frame.id == payload.event_id."""

    async def _fake_acquire_connection_limit(**kwargs: Any) -> _FakeLease:
        return _FakeLease()

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire_connection_limit,
    )

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: _TwoFrameAgentService()
    app.dependency_overrides[get_session_service] = lambda: _AllowSessionService()
    app.dependency_overrides[get_supervisor] = lambda: _NoConflictSupervisor()
    app.dependency_overrides[rate_limit_chat] = _noop_rate_limit

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/sessions/test-session/chat",
                json={"message": "hi"},
            )
            assert response.status_code == 200
            frames = _parse_sse_frames(response.text)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_agent_service, None)
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_supervisor, None)
        app.dependency_overrides.pop(rate_limit_chat, None)

    assert frames, "expected at least one SSE frame in body"
    for frame in frames:
        assert "id" in frame, f"frame missing id: {frame}"
        payload = json.loads(frame["data"])
        assert frame["id"] == payload.get("event_id"), (
            f"frame.id={frame['id']} != payload.event_id={payload.get('event_id')}"
        )
