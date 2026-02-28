from __future__ import annotations

import asyncio
from typing import Any

import pytest
from app.application.errors.exceptions import ConflictError
from app.domain.models.tool_result import ToolResult
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.endpoints import session_routes
from app.interfaces.service_dependencies import get_agent_service, get_session_service
from app.main import app
from fastapi.testclient import TestClient


class _FakeLease:
    def __init__(self) -> None:
        self.heartbeat_started = False
        self.released = False

    def start_heartbeat(self) -> None:
        self.heartbeat_started = True

    async def release(self) -> None:
        self.released = True


class _FakeSandbox:
    def __init__(self) -> None:
        self.output = ""
        self.write_calls: list[dict[str, Any]] = []
        self.read_calls: list[dict[str, Any]] = []
        self.resize_calls: list[dict[str, Any]] = []

    async def write_shell_input(
        self,
        *,
        session_id: str,
        input_text: str,
        press_enter: bool,
    ) -> ToolResult:
        self.write_calls.append(
            {
                "session_id": session_id,
                "input_text": input_text,
                "press_enter": press_enter,
            }
        )
        self.output += input_text
        return ToolResult(success=True, message="", data={"status": "success"})

    async def read_shell_output(self, *, session_id: str, console: bool) -> ToolResult:
        self.read_calls.append({"session_id": session_id, "console": console})
        return ToolResult(success=True, message="", data={"output": self.output})

    async def resize_shell_session(
        self,
        *,
        session_id: str,
        cols: int,
        rows: int,
    ) -> ToolResult:
        self.resize_calls.append(
            {
                "session_id": session_id,
                "cols": cols,
                "rows": rows,
            }
        )
        return ToolResult(success=True, message="", data={"status": "success"})


class _FakeSessionService:
    def __init__(self, sandbox: _FakeSandbox) -> None:
        self.sandbox = sandbox
        self.calls: list[dict[str, Any]] = []

    async def ensure_takeover_shell_session(
        self,
        session_id: str,
        takeover_id: str,
        user_id: str,
        is_admin: bool = False,
    ):
        self.calls.append(
            {
                "session_id": session_id,
                "takeover_id": takeover_id,
                "user_id": user_id,
                "is_admin": is_admin,
            }
        )
        return self.sandbox, f"takeover_{session_id}_{takeover_id}"


class _FakeSandboxWithWsUrl(_FakeSandbox):
    def __init__(self, shell_ws_url: str) -> None:
        super().__init__()
        self.shell_ws_url = shell_ws_url


class _FakeAgentService:
    def __init__(self, conflict_on_call: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.conflict_on_call = conflict_on_call

    async def assert_takeover_shell_access(self, **kwargs) -> None:
        self.calls.append(kwargs)
        if self.conflict_on_call and len(self.calls) >= self.conflict_on_call:
            raise ConflictError("接管租约已失效或不匹配")


def _fake_user() -> User:
    return User(
        id="test-user",
        username="tester",
        role=UserRole.USER,
        status=UserStatus.ACTIVE,
    )


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _install_ws_overrides(monkeypatch: pytest.MonkeyPatch) -> _FakeLease:
    lease = _FakeLease()

    async def _fake_get_current_user_ws_query(token: str | None):
        if not token:
            raise RuntimeError("token required")
        return _fake_user()

    async def _fake_enforce_request_limit(**kwargs) -> None:
        return None

    async def _fake_acquire_connection_limit(**kwargs):
        return lease

    monkeypatch.setattr(
        session_routes,
        "get_current_user_ws_query",
        _fake_get_current_user_ws_query,
    )
    monkeypatch.setattr(
        session_routes,
        "enforce_request_limit",
        _fake_enforce_request_limit,
    )
    monkeypatch.setattr(
        session_routes,
        "acquire_connection_limit",
        _fake_acquire_connection_limit,
    )
    return lease


def test_takeover_shell_ws_forwards_input_and_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = _install_ws_overrides(monkeypatch)
    fake_sandbox = _FakeSandbox()
    fake_session_service = _FakeSessionService(fake_sandbox)
    fake_agent_service = _FakeAgentService()
    app.dependency_overrides[get_session_service] = lambda: fake_session_service
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service

    try:
        client = TestClient(app)
        try:
            with client.websocket_connect(
                "/api/sessions/s1/takeover/shell/ws?token=t1&takeover_id=tk_1"
            ) as ws:
                connected_status = ws.receive_json()
                assert connected_status == {"type": "status", "state": "connected"}

                ws.send_bytes(b"pwd\n")
                output = ws.receive_bytes()
                assert output == b"pwd\n"
        finally:
            client.close()
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_agent_service, None)

    assert lease.heartbeat_started is True
    assert lease.released is True
    assert fake_session_service.calls
    assert fake_session_service.calls[0]["takeover_id"] == "tk_1"
    assert fake_sandbox.write_calls
    assert fake_sandbox.write_calls[0]["input_text"] == "pwd\n"
    assert fake_sandbox.write_calls[0]["press_enter"] is False
    assert fake_agent_service.calls


def test_takeover_shell_ws_reports_lease_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ws_overrides(monkeypatch)
    fake_sandbox = _FakeSandbox()
    fake_session_service = _FakeSessionService(fake_sandbox)
    fake_agent_service = _FakeAgentService(conflict_on_call=2)
    app.dependency_overrides[get_session_service] = lambda: fake_session_service
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service

    try:
        client = TestClient(app)
        try:
            with client.websocket_connect(
                "/api/sessions/s1/takeover/shell/ws?token=t1&takeover_id=tk_2"
            ) as ws:
                connected_status = ws.receive_json()
                assert connected_status == {"type": "status", "state": "connected"}

                expired_status = ws.receive_json()
                assert expired_status == {
                    "type": "status",
                    "state": "lease_expired",
                }
        finally:
            client.close()
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_agent_service, None)


def test_takeover_shell_ws_prefers_sandbox_ws_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ws_overrides(monkeypatch)
    fake_sandbox = _FakeSandboxWithWsUrl("ws://sandbox.local/api/shell/ws")
    fake_session_service = _FakeSessionService(fake_sandbox)
    fake_agent_service = _FakeAgentService()
    app.dependency_overrides[get_session_service] = lambda: fake_session_service
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service

    connect_urls: list[str] = []

    class _FakeSandboxWsPeer:
        def __init__(self) -> None:
            self.sent_payloads: list[Any] = []
            self.recv_queue: asyncio.Queue[Any] = asyncio.Queue()

        async def send(self, payload: Any) -> None:
            self.sent_payloads.append(payload)

        async def recv(self) -> Any:
            return await self.recv_queue.get()

    class _FakeConnectCtx:
        def __init__(self, peer: _FakeSandboxWsPeer, url: str) -> None:
            self._peer = peer
            self._url = url

        async def __aenter__(self) -> _FakeSandboxWsPeer:
            connect_urls.append(self._url)
            return self._peer

        async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
            return False

    peer = _FakeSandboxWsPeer()
    peer.recv_queue.put_nowait(b"from-sandbox\n")

    def _fake_connect(url: str):
        return _FakeConnectCtx(peer, url)

    monkeypatch.setattr(session_routes.websockets, "connect", _fake_connect)

    try:
        client = TestClient(app)
        try:
            with client.websocket_connect(
                "/api/sessions/s1/takeover/shell/ws?token=t1&takeover_id=tk_3"
            ) as ws:
                connected_status = ws.receive_json()
                assert connected_status == {"type": "status", "state": "connected"}

                ws.send_bytes(b"pwd\n")
                output = ws.receive_bytes()
                assert output == b"from-sandbox\n"
        finally:
            client.close()
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_agent_service, None)

    assert connect_urls == ["ws://sandbox.local/api/shell/ws?session_id=takeover_s1_tk_3"]
    assert peer.sent_payloads
    assert peer.sent_payloads[0] == b"pwd\n"


def test_takeover_shell_ws_http_fallback_forwards_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_ws_overrides(monkeypatch)
    fake_sandbox = _FakeSandbox()
    fake_session_service = _FakeSessionService(fake_sandbox)
    fake_agent_service = _FakeAgentService()
    app.dependency_overrides[get_session_service] = lambda: fake_session_service
    app.dependency_overrides[get_agent_service] = lambda: fake_agent_service

    try:
        client = TestClient(app)
        try:
            with client.websocket_connect(
                "/api/sessions/s1/takeover/shell/ws?token=t1&takeover_id=tk_4"
            ) as ws:
                connected_status = ws.receive_json()
                assert connected_status == {"type": "status", "state": "connected"}

                ws.send_text('{"type":"resize","cols":123,"rows":40}')
                ws.send_bytes(b"echo ok\n")
                _ = ws.receive_bytes()
        finally:
            client.close()
    finally:
        app.dependency_overrides.pop(get_session_service, None)
        app.dependency_overrides.pop(get_agent_service, None)

    assert fake_sandbox.resize_calls == [
        {
            "session_id": "takeover_s1_tk_4",
            "cols": 123,
            "rows": 40,
        }
    ]
