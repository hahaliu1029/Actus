"""R4 I-R4.6 end-to-end: POST /chat SSE endpoint 的 tool event wire 必须用短字段名.

Target: session_routes.py:222-272 (chat endpoint), 其 to_sse_data_json() callsite
必须保持 by_alias=True wire.

Pattern: _FakeAgentService + dependency_overrides. 不启 sandbox / LangGraph,
只让 agent_service.chat() async generator yield 一条 R4 ToolEvent, 走完
EventMapper → ToolSSEEvent → to_sse_data_json → ServerSentEvent 的完整 wire 链.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import json
from typing import Any, AsyncGenerator

import httpx
import pytest

from app.domain.models.event import BaseEvent, ToolEvent, ToolEventStatus
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
    """简化 rate_limit lease, 直接 no-op."""

    def start_heartbeat(self) -> None:
        pass

    async def release(self) -> None:
        pass


class _ChatAgentService:
    """重现 agent_service.chat() async generator 接口, yield 一条 R4 ToolEvent."""

    async def chat(
        self,
        **kwargs: Any,
    ) -> AsyncGenerator[BaseEvent, None]:
        yield ToolEvent(
            tool_call_id="c_sse_wire_test",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
            function_result=None,
            artifact={
                "tool_call_id": "c_sse_wire_test",
                "tool_name": "shell_execute",
                "tool_source": {
                    "source": "native",
                    "category": "shell",
                    "canonical_name": "shell_execute",
                },
                "outcome": {"variant": "allow_success", "content": "ok", "data": None},
            },
        )


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


async def test_post_chat_sse_wire_uses_short_field_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """E2E: POST /chat → agent_service.chat → ServerSentEvent(data=...) SSE body
    must contain tool frame with wire short names name/function/args.

    acquire_connection_limit is a direct module-level function call in
    session_routes.py:236 (NOT Depends()), so we use monkeypatch.setattr
    on the module symbol. See test_session_takeover_ws_route.py:141 for same pattern.
    """

    async def _fake_acquire_connection_limit(**kwargs: Any) -> _FakeLease:
        return _FakeLease()

    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire_connection_limit,
    )

    app.dependency_overrides[get_current_user] = _fake_user
    app.dependency_overrides[get_agent_service] = lambda: _ChatAgentService()
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
            body = response.text

            tool_data: dict | None = None
            for frame in body.split("\n\n"):
                if "event: tool" not in frame:
                    continue
                for line in frame.split("\n"):
                    if line.startswith("data: "):
                        tool_data = json.loads(line[6:])
                        break
                if tool_data is not None:
                    break

            assert tool_data is not None, f"no tool event in SSE body: {body[:500]!r}"

            assert "name" in tool_data, f"wire missing 'name': {tool_data!r}"
            assert "function" in tool_data
            assert "args" in tool_data
            assert tool_data["name"] == "shell"
            assert tool_data["function"] == "shell_execute"
            assert tool_data["args"] == {"command": "ls"}

            assert "tool_name" not in tool_data
            assert "function_name" not in tool_data
            assert "function_args" not in tool_data

            assert tool_data.get("envelope_version") == 1
    finally:
        app.dependency_overrides.clear()
