from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.models.session import (
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-1") -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def cdp_url(self) -> str:
        return f"http://{self._id}:9222"

    @property
    def shell_ws_url(self) -> str:
        return f"ws://{self._id}:8080/api/shell/ws"

    @property
    def vnc_url(self) -> str:
        return f"ws://{self._id}:5901"

    async def ensure_sandbox(self) -> None:
        pass

    async def destroy(self) -> bool:
        return True

    @classmethod
    async def create(cls, user_id: Optional[str] = None, **_kw) -> "FakeSandbox":
        # SPM Task 3 Step 4: accept + ignore session_id/attempt kwargs.
        return cls()


class FakeUoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.session = MagicMock()
        self.session.get_by_id = AsyncMock(side_effect=lambda sid: self._sessions.get(sid))
        self.session.save = AsyncMock(side_effect=self._save)
        self.session.get_all = AsyncMock(side_effect=lambda: list(self._sessions.values()))
        self.session.add_event = AsyncMock()
        self.sandbox_lifecycle_log = MagicMock()
        self.sandbox_lifecycle_log.create = AsyncMock()

    async def _save(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass


class _SpySink:
    def __init__(self) -> None:
        self.calls: list = []

    async def record(self, snapshot) -> None:
        self.calls.append(snapshot)


class _RaisingSink:
    async def record(self, snapshot) -> None:
        raise RuntimeError("sink boom")


def _unbound_session(sid: str = "sess-1", user_id=None) -> Session:
    return Session(
        id=sid,
        user_id=user_id,
        status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(state=SandboxBindingState.UNBOUND),
    )


def _make_service(sessions, sink, *, enabled: bool):
    uow = FakeUoW(sessions)
    svc = SandboxLifecycleService(
        sandbox_cls=FakeSandbox, uow_factory=lambda: uow, sink=sink,
        policy_snapshot_enabled=enabled,  # C5a flag via ctor — no monkeypatch needed
    )
    return svc, uow


async def test_observe_records_one_snapshot_when_on():
    sink = _SpySink()
    svc, _ = _make_service({"sess-1": _unbound_session()}, sink, enabled=True)
    handle = await svc.bind_new("sess-1")
    assert handle.generation == 1
    assert len(sink.calls) == 1
    snap = sink.calls[0]
    assert snap.surface == "container_create"
    assert snap.enforcement_mode == "observe_only"
    assert snap.subject.sandbox_id == "sbx-1"
    assert snap.subject.sandbox_generation == 1
    assert snap.subject.worker_type == "root"
    assert snap.command is None and snap.container is not None


async def test_observe_records_zero_when_off():
    sink = _SpySink()
    svc, _ = _make_service({"sess-1": _unbound_session()}, sink, enabled=False)
    handle = await svc.bind_new("sess-1")
    assert handle.generation == 1
    assert sink.calls == []


async def test_inv0_handle_and_binding_identical_on_vs_off():
    svc_on, uow_on = _make_service({"sess-1": _unbound_session()}, _SpySink(), enabled=True)
    h_on = await svc_on.bind_new("sess-1")
    b_on = uow_on._sessions["sess-1"].sandbox_binding

    svc_off, uow_off = _make_service({"sess-1": _unbound_session()}, _SpySink(), enabled=False)
    h_off = await svc_off.bind_new("sess-1")
    b_off = uow_off._sessions["sess-1"].sandbox_binding

    assert (h_on.id, h_on.generation) == (h_off.id, h_off.generation)
    assert (b_on.state, b_on.generation, b_on.id) == (b_off.state, b_off.generation, b_off.id)


async def test_sink_raise_does_not_break_bind():
    svc, _ = _make_service({"sess-1": _unbound_session()}, _RaisingSink(), enabled=True)
    handle = await svc.bind_new("sess-1")  # must NOT raise
    assert handle.generation == 1


async def test_user_id_never_reaches_snapshot():
    import json
    sink = _SpySink()
    svc, _ = _make_service({"sess-1": _unbound_session(user_id="SECRET_UID")}, sink, enabled=True)
    await svc.bind_new("sess-1")  # bind_new falls back to session.user_id
    dumped = json.dumps(sink.calls[0].model_dump(mode="json"))
    assert "SECRET_UID" not in dumped  # user_id stays in the input, never in PolicySubject/log
