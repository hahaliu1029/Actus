"""SPM PR-3 Task 28 — off-mode endpoint matrix (REST 409 / WS 4409).

Route-level wiring for the frozen spec §5.6 matrix (INV-SPM-7). The GATE LOGIC
is unit-tested at the service layer (``test_agent_service_provision_mode.py`` for
takeover-start/reopen + retry-from-suspend; ``_acquire_sandbox`` below for
file/shell/download); this file proves the ROUTE surfaces the correct HTTP/WS
signal and that read-only endpoints stay ungated.

Contract:
* REST needing a sandbox → HTTP 409 with ``msg == "SANDBOX_DISABLED"`` (the
  AppException handler maps ``SandboxDisabledError.code/status_code=409`` and the
  ``msg`` sentinel into the unified ``Response`` body — int ``code`` protocol).
* WS (vnc / takeover-shell) → after accept, an OUR-wire status payload
  ``{"type":"status","code":"SANDBOX_DISABLED"}`` then ``close(code=4409)``.
* ``GET /{id}/files`` (pure DB) + takeover READ stay ungated.
"""
from __future__ import annotations

from typing import Optional

import contextlib

import httpx
import pytest
from app.application.services.agent_service import AgentService
from app.application.services.session_service import SessionService
from app.domain.models.session import Session, SessionStatus
from app.domain.models.user import User, UserRole, UserStatus
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.rate_limit import rate_limit_read, rate_limit_write
from app.interfaces.endpoints import session_routes
from app.interfaces.service_dependencies import (
    get_agent_service,
    get_session_service,
    get_skill_creator_service,
)
from app.main import app
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from tests.app.application.services.conftest import default_snapshot as _default_snapshot


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _set_off(monkeypatch) -> None:
    """Force ``sandbox_provision_mode='off'`` on the process settings singleton
    (config still REJECTS off until PR-4, so also unlock ALLOWED)."""
    from core.config import Settings, get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "sandbox_provision_mode", "off", raising=False)
    monkeypatch.setattr(
        Settings,
        "SANDBOX_PROVISION_MODE_ALLOWED",
        {"always", "on_demand", "off"},
        raising=False,
    )


def _admin_user() -> User:
    # SUPER_ADMIN: satisfies AdminUser (skills/create) + session access (bypass
    # owner check) + takeover capability (allowed_roles default "super_admin,user").
    return User(
        id="admin-user",
        username="admin",
        role=UserRole.SUPER_ADMIN,
        status=UserStatus.ACTIVE,
    )


class _SpyLifecycle:
    """Records lifecycle calls so the retry-from-suspend test can assert ZERO
    resume (the off-check fires before any lifecycle touch)."""

    def __init__(self) -> None:
        self.resume_calls = 0
        self.acquire_calls = 0
        self.bind_calls = 0
        self.suspend_calls = 0
        self.destroy_calls = 0

    async def acquire(self, session_id: str):  # pragma: no cover - off short-circuits
        self.acquire_calls += 1
        raise AssertionError("lifecycle.acquire must not run under off")

    async def resume(self, session_id: str):  # pragma: no cover
        self.resume_calls += 1
        raise AssertionError("lifecycle.resume must not run under off")

    async def bind_new(self, session_id: str, *, user_id: Optional[str] = None):  # pragma: no cover
        self.bind_calls += 1
        raise AssertionError("lifecycle.bind_new must not run under off")


class _SessRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        if self._session and session_id == self._session.id:
            return self._session.model_copy(deep=True)
        return None


class _UoW:
    def __init__(self, session: Session) -> None:
        self.session = _SessRepo(session)

    async def __aenter__(self) -> "_UoW":
        return self

    async def __aexit__(self, *a) -> None:
        return None


class _OffRestEnv:
    """The ``off_client`` surface the brief consumes: an httpx client + the
    lifecycle spy (for the retry zero-touch assertion)."""

    def __init__(self, client: httpx.AsyncClient, lifecycle_spy: _SpyLifecycle,
                 session_id: str) -> None:
        self._client = client
        self.lifecycle_spy = lifecycle_spy
        self.session_id = session_id

    async def post(self, path: str, **kw) -> httpx.Response:
        return await self._client.post(f"/api{path}", **kw)

    async def get(self, path: str, **kw) -> httpx.Response:
        return await self._client.get(f"/api{path}", **kw)


async def _noop_rate_limit() -> None:
    return None


class _FakeCreatorService:  # only reached if the off-gate FAILS to fire
    async def create(self, *, description: str, sandbox=None, installed_by: str = ""):
        raise AssertionError("skills/create must 409 before the creator runs under off")
        yield  # pragma: no cover - makes this an async generator


@contextlib.asynccontextmanager
async def _off_rest_env(monkeypatch):
    _set_off(monkeypatch)
    session = Session(
        id="s1", user_id="owner-1", status=SessionStatus.RUNNING, files=[]
    )
    spy = _SpyLifecycle()
    session_service = SessionService(
        uow_factory=lambda: _UoW(session), sandbox_lifecycle_service=spy
    )
    agent_service = AgentService(
        uow_factory=lambda: _UoW(session),
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
        sandbox_lifecycle_service=spy,
    )
    from unittest.mock import MagicMock

    agent_service._supervisor = MagicMock()

    app.dependency_overrides[get_current_user] = _admin_user
    app.dependency_overrides[get_session_service] = lambda: session_service
    app.dependency_overrides[get_agent_service] = lambda: agent_service
    app.dependency_overrides[get_skill_creator_service] = lambda: _FakeCreatorService()
    app.dependency_overrides[rate_limit_read] = _noop_rate_limit
    app.dependency_overrides[rate_limit_write] = _noop_rate_limit
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield _OffRestEnv(client, spy, "s1")
    finally:
        for dep in (
            get_current_user, get_session_service, get_agent_service,
            get_skill_creator_service, rate_limit_read, rate_limit_write,
        ):
            app.dependency_overrides.pop(dep, None)


def _assert_sandbox_disabled(resp: httpx.Response) -> None:
    assert resp.status_code == 409, resp.text
    assert resp.json()["msg"] == "SANDBOX_DISABLED", resp.text


# ── REST matrix ─────────────────────────────────────────────────────────────


class TestOffEndpointMatrix:
    @pytest.mark.anyio
    async def test_file_upload_shell_download_409(self, monkeypatch) -> None:
        async with _off_rest_env(monkeypatch) as off_client:
            sid = off_client.session_id
            read = await off_client.post(f"/sessions/{sid}/file", json={"filepath": "/x"})
            _assert_sandbox_disabled(read)
            shell = await off_client.post(
                f"/sessions/{sid}/shell", json={"session_id": "sh-1"}
            )
            _assert_sandbox_disabled(shell)
            download = await off_client.get(
                f"/sessions/{sid}/file/download", params={"filepath": "/x"}
            )
            _assert_sandbox_disabled(download)
            # zero lifecycle touch across all three
            assert off_client.lifecycle_spy.acquire_calls == 0
            assert off_client.lifecycle_spy.resume_calls == 0

    @pytest.mark.anyio
    async def test_takeover_start_and_reopen_409_but_read_ok(self, monkeypatch) -> None:
        async with _off_rest_env(monkeypatch) as off_client:
            sid = off_client.session_id
            start = await off_client.post(
                f"/sessions/{sid}/takeover/start", json={"scope": "shell"}
            )
            _assert_sandbox_disabled(start)
            reopen = await off_client.post(f"/sessions/{sid}/takeover/reopen")
            _assert_sandbox_disabled(reopen)
            # READ (get_takeover) is NOT off-gated — it reaches domain logic (200).
            read = await off_client.get(f"/sessions/{sid}/takeover")
            assert read.status_code == 200, read.text
            assert read.json()["msg"] != "SANDBOX_DISABLED"
            # zero lifecycle touch on the gated paths
            assert off_client.lifecycle_spy.resume_calls == 0

    @pytest.mark.anyio
    async def test_retry_from_suspend_409_no_lifecycle_touch(self, monkeypatch) -> None:
        """r14/codex R13(L): off retry-from-suspend → 409 + lifecycle resume spy 0."""
        async with _off_rest_env(monkeypatch) as off_client:
            sid = off_client.session_id
            resp = await off_client.post(f"/sessions/{sid}/retry-from-suspend")
            _assert_sandbox_disabled(resp)
            assert off_client.lifecycle_spy.resume_calls == 0

    @pytest.mark.anyio
    async def test_skills_create_409(self, monkeypatch) -> None:
        async with _off_rest_env(monkeypatch) as off_client:
            resp = await off_client.post("/v2/skills/create", json={"description": "x"})
            _assert_sandbox_disabled(resp)

    @pytest.mark.anyio
    async def test_list_files_still_ok(self, monkeypatch) -> None:
        async with _off_rest_env(monkeypatch) as off_client:
            sid = off_client.session_id
            resp = await off_client.get(f"/sessions/{sid}/files")
            assert resp.status_code == 200, resp.text
            assert resp.json()["msg"] != "SANDBOX_DISABLED"


# ── WS matrix (accept → status → close 4409) ────────────────────────────────

from app.infrastructure.storage.redis import get_redis


def _install_ws_dep_stubs() -> None:
    """The WS routes declare ``Depends(get_session_service/get_agent_service/
    get_redis)`` — FastAPI resolves those BEFORE the handler body (where the
    off-check lives), and the real providers need an initialized Postgres. Stub
    them so the body is reached; the off-check fires before any is used."""
    from unittest.mock import MagicMock

    app.dependency_overrides[get_session_service] = lambda: MagicMock()
    app.dependency_overrides[get_agent_service] = lambda: MagicMock()
    app.dependency_overrides[get_redis] = lambda: MagicMock()


def _clear_ws_dep_stubs() -> None:
    for dep in (get_session_service, get_agent_service, get_redis):
        app.dependency_overrides.pop(dep, None)


def _connect_and_assert_4409(url: str) -> None:
    client = TestClient(app)
    received: list = []
    close_code: Optional[int] = None
    try:
        with client.websocket_connect(url) as ws:
            try:
                while True:
                    received.append(ws.receive_json())
            except WebSocketDisconnect as exc:
                close_code = exc.code
    finally:
        client.close()
    assert received and received[-1] == {"type": "status", "code": "SANDBOX_DISABLED"}
    assert close_code == 4409


def _install_vnc_ws_auth(monkeypatch) -> None:
    """vnc runs auth BEFORE accept — stub it so the handler reaches accept, where
    the off-check fires (takeover-shell checks off before auth, so it needs none)."""
    async def _user(token: str | None):
        return _admin_user()

    async def _noop(**kwargs) -> None:
        return None

    class _Lease:
        def start_heartbeat(self) -> None: ...
        async def release(self) -> None: ...

    async def _lease(**kwargs):
        return _Lease()

    monkeypatch.setattr(session_routes, "get_current_user_ws_query", _user)
    monkeypatch.setattr(session_routes, "enforce_request_limit", _noop)
    monkeypatch.setattr(session_routes, "acquire_connection_limit", _lease)


def test_vnc_ws_status_then_close_4409(monkeypatch) -> None:
    _set_off(monkeypatch)
    _install_vnc_ws_auth(monkeypatch)
    _install_ws_dep_stubs()
    try:
        _connect_and_assert_4409("/api/sessions/s1/vnc?token=t1")
    finally:
        _clear_ws_dep_stubs()


def test_takeover_shell_ws_status_then_close_4409(monkeypatch) -> None:
    """takeover shell WS is a separate route (session_routes.py) — the off-check
    sits right after accept, ahead of takeover_id/auth (dominant global cond)."""
    _set_off(monkeypatch)
    _install_ws_dep_stubs()
    try:
        _connect_and_assert_4409(
            "/api/sessions/s1/takeover/shell/ws?token=t1&takeover_id=tk_1"
        )
    finally:
        _clear_ws_dep_stubs()
