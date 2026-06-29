from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import core.config as cfg
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.models.sandbox_policy import ContainerRuntimePolicy
from app.domain.models.session import (
    SandboxBinding, SandboxBindingState, Session, SessionStatus,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


_APPLIED = ContainerRuntimePolicy(
    capture_kind="applied", creation_mode="docker_run",
    image="actus/sandbox:latest", mem_limit="4g", run_as_user=None,
    read_only_rootfs=False, cap_drop=("NET_RAW",), cap_add=(), security_opt=(),
    pids_limit=512, mounts=(),
)


class HardenedFakeSandbox:
    """C5c-aware fake: create() accepts runtime_policy; carries applied policy."""
    last_runtime_policy = "UNSET"

    def __init__(self, applied=None) -> None:
        self._applied = applied

    @property
    def id(self) -> str:
        return "sbx-1"

    @property
    def cdp_url(self) -> str:
        return "http://sbx-1:9222"

    @property
    def shell_ws_url(self) -> str:
        return "ws://sbx-1:8080/api/shell/ws"

    @property
    def vnc_url(self) -> str:
        return "ws://sbx-1:5901"

    @property
    def applied_runtime_policy(self):
        return self._applied

    async def ensure_sandbox(self) -> None:
        ...

    async def destroy(self) -> bool:
        return True

    @classmethod
    async def create(cls, user_id: Optional[str] = None, *, runtime_policy=None):
        cls.last_runtime_policy = runtime_policy
        # Mirror _create_task: applied set iff a policy was passed (hardening on).
        return cls(applied=_APPLIED if runtime_policy is not None else None)


class LegacyFakeSandbox:
    """Pre-C5c fake: create() WITHOUT runtime_policy + no applied property."""
    create_called_with = "UNSET"

    @property
    def id(self) -> str:
        return "sbx-1"

    @property
    def cdp_url(self) -> str:
        return "http://sbx-1:9222"

    @property
    def shell_ws_url(self) -> str:
        return "ws://sbx-1:8080/api/shell/ws"

    @property
    def vnc_url(self) -> str:
        return "ws://sbx-1:5901"

    async def ensure_sandbox(self) -> None:
        ...

    async def destroy(self) -> bool:
        return True

    @classmethod
    async def create(cls, user_id: Optional[str] = None):
        cls.create_called_with = user_id
        return cls()


class _FakeUoW:
    def __init__(self, sessions):
        self._sessions = sessions
        self.session = MagicMock()
        self.session.get_by_id = AsyncMock(side_effect=lambda sid: self._sessions.get(sid))
        self.session.save = AsyncMock(side_effect=self._save)
        self.session.get_all = AsyncMock(side_effect=lambda: list(self._sessions.values()))
        self.session.add_event = AsyncMock()
        self.sandbox_lifecycle_log = MagicMock()
        self.sandbox_lifecycle_log.create = AsyncMock()

    async def _save(self, session):
        self._sessions[session.id] = session

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        ...


class _SpySink:
    def __init__(self):
        self.calls = []

    async def record(self, snapshot):
        self.calls.append(snapshot)


def _unbound_session(sid="sess-1", user_id=None):
    return Session(
        id=sid, user_id=user_id, status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(state=SandboxBindingState.UNBOUND),
    )


def _svc(sandbox_cls, sessions, sink, *, snapshot_enabled, hardening_enabled):
    uow = _FakeUoW(sessions)
    return SandboxLifecycleService(
        sandbox_cls=sandbox_cls, uow_factory=lambda: uow, sink=sink,
        policy_snapshot_enabled=snapshot_enabled,
        runtime_hardening_enabled=hardening_enabled,
    )


def _patch_get_settings(monkeypatch, *, hardening):
    class _S:
        sandbox_address = None
        sandbox_image = "actus/sandbox:latest"
        sandbox_network = None
        sandbox_mem_limit = "4g"
        sandbox_default_cwd = "/root"
        sandbox_https_proxy = None
        sandbox_http_proxy = None
        sandbox_no_proxy = None
        sandbox_memory_mount_target = "/workspace/.memory"
        sandbox_memory_mount_enabled = True
        sandbox_runtime_hardening_enabled = hardening
        sandbox_no_new_privileges_enabled = False
    monkeypatch.setattr(cfg, "get_settings", lambda: _S())


async def test_hardening_on_emits_applied_enforce_snapshot(monkeypatch):
    _patch_get_settings(monkeypatch, hardening=True)
    HardenedFakeSandbox.last_runtime_policy = "UNSET"
    sink = _SpySink()
    svc = _svc(HardenedFakeSandbox, {"sess-1": _unbound_session()}, sink,
               snapshot_enabled=True, hardening_enabled=True)
    handle = await svc.bind_new("sess-1")
    assert handle.generation == 1
    # create() received a (non-None) runtime_policy on the ON path.
    assert HardenedFakeSandbox.last_runtime_policy not in (None, "UNSET")
    assert len(sink.calls) == 1
    snap = sink.calls[0]
    assert snap.surface == "container_create"
    assert snap.enforcement_mode == "enforce"
    assert snap.container.capture_kind == "applied"
    assert snap.container.cap_drop == ("NET_RAW",)


async def test_hardening_on_without_applied_falls_back_to_observe(monkeypatch):
    # external_address / fake-without-applied → applied is None → C5a configured/observe.
    _patch_get_settings(monkeypatch, hardening=True)

    # NOTE: must be an ASYNC classmethod — bind_new does `await ...create(...)`. A sync
    # lambda would return a non-awaitable → TypeError, never reaching the fallback.
    async def _create_without_applied(cls, user_id=None, *, runtime_policy=None):
        return cls(applied=None)

    monkeypatch.setattr(
        HardenedFakeSandbox, "create", classmethod(_create_without_applied)
    )
    sink = _SpySink()
    svc = _svc(HardenedFakeSandbox, {"sess-1": _unbound_session()}, sink,
               snapshot_enabled=True, hardening_enabled=True)
    await svc.bind_new("sess-1")
    snap = sink.calls[0]
    assert snap.enforcement_mode == "observe_only"
    assert snap.container.capture_kind == "configured"


async def test_hardening_off_calls_create_without_kwarg(monkeypatch):
    # INV-0 back-compat: a LEGACY fake whose create() lacks runtime_policy must work.
    _patch_get_settings(monkeypatch, hardening=False)
    LegacyFakeSandbox.create_called_with = "UNSET"
    sink = _SpySink()
    svc = _svc(LegacyFakeSandbox, {"sess-1": _unbound_session()}, sink,
               snapshot_enabled=True, hardening_enabled=False)
    handle = await svc.bind_new("sess-1")  # must NOT raise (no runtime_policy passed)
    assert handle.generation == 1
    assert LegacyFakeSandbox.create_called_with is None  # called with user_id=None only
    assert sink.calls[0].enforcement_mode == "observe_only"  # C5a path
    assert sink.calls[0].container.capture_kind == "configured"  # OFF stays configured (§9.7)


async def test_hardening_on_snapshot_off_passes_policy_without_snapshot(monkeypatch):
    # Spec §8.2 truth-table row 3 (hardening ON + C5a audit OFF): kwargs hardened, NO
    # snapshot. Guards against an impl that wrongly nests the runtime-policy compile/pass
    # inside the snapshot gate — the two flags are INDEPENDENT (behavior vs audit).
    _patch_get_settings(monkeypatch, hardening=True)
    HardenedFakeSandbox.last_runtime_policy = "UNSET"
    sink = _SpySink()
    svc = _svc(HardenedFakeSandbox, {"sess-1": _unbound_session()}, sink,
               snapshot_enabled=False, hardening_enabled=True)
    await svc.bind_new("sess-1")
    assert HardenedFakeSandbox.last_runtime_policy not in (None, "UNSET")  # hardened
    assert sink.calls == []  # no snapshot (C5a audit gate off)
