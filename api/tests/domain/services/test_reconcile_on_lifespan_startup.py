"""Integration test: reconcile_orphans on startup.

§11.2: Seeds a DESTROYING session with a dead container in the database,
runs reconcile_orphans(), expects binding.state → DESTROYED.
Also tests CREATING recovery (stuck bind_new → UNBOUND).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class DeadContainerSandbox:
    """Sandbox stub where get() always returns None (container dead)."""

    @classmethod
    async def create(cls, user_id: str | None = None, **_kw):
        return None

    @classmethod
    async def get(cls, id: str):
        return None


class FakeUoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.session = MagicMock()
        self.session.get_by_id = AsyncMock(side_effect=lambda sid: self._sessions.get(sid))
        self.session.save = AsyncMock(side_effect=lambda s: self._sessions.__setitem__(s.id, s))
        self.session.get_all = AsyncMock(side_effect=lambda: list(self._sessions.values()))
        self.session.add_event = AsyncMock()
        self.sandbox_lifecycle_log = MagicMock()
        self.sandbox_lifecycle_log.create = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


async def test_reconcile_destroying_to_destroyed() -> None:
    """DESTROYING + container dead → DESTROYED on startup reconcile."""
    session = Session(
        id="sess-orphan",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-dead",
            state=SandboxBindingState.DESTROYING,
            generation=2,
            destroy_reason=DestroyReason.SESSION_DELETE,
        ),
    )
    uow = FakeUoW({"sess-orphan": session})
    service = SandboxLifecycleService(
        sandbox_cls=DeadContainerSandbox,
        uow_factory=lambda: uow,
    )

    await service.reconcile_orphans()

    saved = uow._sessions["sess-orphan"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroyed_at is not None


async def test_reconcile_creating_to_unbound() -> None:
    """CREATING (interrupted bind_new) → UNBOUND on startup reconcile."""
    session = Session(
        id="sess-stuck",
        status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(
            state=SandboxBindingState.CREATING,
            generation=0,
        ),
    )
    uow = FakeUoW({"sess-stuck": session})
    service = SandboxLifecycleService(
        sandbox_cls=DeadContainerSandbox,
        uow_factory=lambda: uow,
    )

    await service.reconcile_orphans()

    saved = uow._sessions["sess-stuck"]
    assert saved.sandbox_binding.state == SandboxBindingState.UNBOUND


async def test_reconcile_skips_active_sessions() -> None:
    """ACTIVE sessions are not touched by reconcile (lazy rehydrate in PR1)."""
    session = Session(
        id="sess-ok",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-alive",
            state=SandboxBindingState.ACTIVE,
            generation=1,
        ),
    )
    uow = FakeUoW({"sess-ok": session})
    service = SandboxLifecycleService(
        sandbox_cls=DeadContainerSandbox,
        uow_factory=lambda: uow,
    )

    await service.reconcile_orphans()

    saved = uow._sessions["sess-ok"]
    assert saved.sandbox_binding.state == SandboxBindingState.ACTIVE  # untouched


# ══════════════════════════════════════════════════════════════════════════════
# SPM PR-3 Task 28 — off-startup reconcile gating + lingering accounting
# ══════════════════════════════════════════════════════════════════════════════


class _CountingSandbox:
    """sandbox_cls double whose ``get()`` bumps a counter (no ``get_strict`` so
    the DESTROYING probe falls back to it). ``get_calls`` proves off startup
    skips the whole Docker-dependent DESTROYING loop (zero Docker touch)."""

    def __init__(self) -> None:
        self.get_calls = 0

    async def create(self, user_id: str | None = None, **_kw):  # pragma: no cover
        return None

    async def get(self, id: str):
        self.get_calls += 1
        return None


class _FakeSupervisorRegistry:
    """Records ``spawn`` targets. In production a spawned MailboxSupervisor runs
    the startup XAUTOCLAIM (PEL drain), so ``spawn`` being called for a root IS
    the "mailbox/PEL recovery ran" signal; ``_mailbox.xautoclaim_called`` models
    that downstream drain so the brief's assertion shape holds."""

    def __init__(self, mailbox: "_MailboxProbe") -> None:
        self.spawned: list[str] = []
        self._mailbox = mailbox

    async def health_check(self):
        return {}  # empty → every RUNNING root is treated as missing → spawn

    async def spawn(self, root_session_id: str) -> None:
        self.spawned.append(root_session_id)
        self._mailbox.xautoclaim_called = True

    async def stop(self, root_session_id: str) -> None:  # pragma: no cover
        return None

    def rebuilt_for(self, session_id: str) -> bool:
        return session_id in self.spawned


class _MailboxProbe:
    def __init__(self) -> None:
        self.xautoclaim_called = False


class _OffReconcileFakes:
    """Harness exposing the ``svc_and_fakes`` surface the Task 28 brief consumes.

    Built ONCE; ``make_binding`` mutates the shared ``_sessions`` dict + mailbox
    root list that the already-built uow reads LIVE, so the brief's call order
    (``svc, fakes = svc_and_fakes`` → ``fakes.make_binding(...)`` → reconcile)
    works without a rebuild.
    """

    def __init__(self) -> None:
        self.session_id = "sess-off"
        self._sessions: dict[str, Session] = {}
        self._mailbox_root_ids: list[str] = []
        self.sandbox_cls = _CountingSandbox()
        self.mailbox = _MailboxProbe()
        self.supervisor_registry = _FakeSupervisorRegistry(self.mailbox)
        self._uow = FakeUoW(self._sessions)
        # live mailbox-plane query (reads _mailbox_root_ids at call time)
        self._uow.session.find_running_mailbox_plane_root_ids = AsyncMock(
            side_effect=lambda: list(self._mailbox_root_ids)
        )

    def make_binding(
        self,
        *,
        state: SandboxBindingState,
        sandbox_id: str | None = None,
        is_mailbox_root: bool = False,
    ) -> None:
        self._sessions[self.session_id] = Session(
            id=self.session_id,
            status=SessionStatus.RUNNING,
            sandbox_binding=SandboxBinding(id=sandbox_id, state=state, generation=1),
        )
        if is_mailbox_root:
            self._mailbox_root_ids.append(self.session_id)

    def service(self) -> SandboxLifecycleService:
        return SandboxLifecycleService(
            sandbox_cls=self.sandbox_cls,
            uow_factory=lambda: self._uow,
            supervisor_registry=self.supervisor_registry,
        )

    def binding_state(self, session_id: str) -> SandboxBindingState:
        return self._sessions[session_id].sandbox_binding.state


@pytest.fixture()
def svc_and_fakes():
    fakes = _OffReconcileFakes()
    return fakes.service(), fakes


async def test_off_startup_preserves_destroying_and_zero_docker(svc_and_fakes) -> None:
    """off startup: DESTROYING binding left intact (not mis-finalized) + zero
    Docker touch (the probe/drain/destroy loop is skipped)."""
    svc, fakes = svc_and_fakes
    fakes.make_binding(state=SandboxBindingState.DESTROYING, sandbox_id="sb-1")

    await svc.reconcile_orphans(docker_dependent_enabled=False)

    assert fakes.binding_state(fakes.session_id) == SandboxBindingState.DESTROYING
    assert fakes.sandbox_cls.get_calls == 0


async def test_off_startup_keeps_mailbox_pel_recovery(svc_and_fakes) -> None:
    """off startup skips ONLY Docker steps: the non-Docker mailbox supervisor /
    PEL recovery still fires (R24-CLASS3 — PEL drain is Redis, not Docker)."""
    svc, fakes = svc_and_fakes
    fakes.make_binding(state=SandboxBindingState.ACTIVE, sandbox_id="sb-1", is_mailbox_root=True)

    await svc.reconcile_orphans(docker_dependent_enabled=False)

    assert fakes.supervisor_registry.rebuilt_for(fakes.session_id)  # supervisor rebuilt
    assert fakes.mailbox.xautoclaim_called                          # PEL drain fired
    assert fakes.sandbox_cls.get_calls == 0                         # still zero Docker


async def test_off_startup_still_repairs_creating(svc_and_fakes) -> None:
    """CREATING→UNBOUND DB repair is non-Docker → runs under off startup."""
    svc, fakes = svc_and_fakes
    fakes.make_binding(state=SandboxBindingState.CREATING, sandbox_id=None)

    await svc.reconcile_orphans(docker_dependent_enabled=False)

    assert fakes.binding_state(fakes.session_id) == SandboxBindingState.UNBOUND
    assert fakes.sandbox_cls.get_calls == 0


async def test_lingering_metric_counts_exact_mixed_states() -> None:
    """Predicate = sandbox_id NOT NULL × state ∈ {ACTIVE,SUSPENDED,DESTROYING};
    UNBOUND/id=None + DESTROYED must NOT count. Asserts the REAL provision-metrics
    singleton snapshot (r9/G3 anti-fake guard — proves production metric wired)."""
    from app.application.services.sandbox_provision_metrics import (
        SandboxProvisionMetrics,
    )

    def _mk(sid: str, state: SandboxBindingState, sandbox_id: str | None) -> Session:
        return Session(
            id=sid,
            status=SessionStatus.RUNNING,
            sandbox_binding=SandboxBinding(id=sandbox_id, state=state, generation=1),
        )

    sessions = {
        s.id: s
        for s in [
            _mk("s1", SandboxBindingState.ACTIVE, "sb-1"),
            _mk("s2", SandboxBindingState.SUSPENDED, "sb-2"),
            _mk("s3", SandboxBindingState.DESTROYING, "sb-3"),
            _mk("s4", SandboxBindingState.UNBOUND, None),      # default binding — excluded
            _mk("s5", SandboxBindingState.DESTROYED, "sb-5"),  # terminal — excluded
        ]
    }
    uow = FakeUoW(sessions)
    service = SandboxLifecycleService(
        sandbox_cls=DeadContainerSandbox,
        uow_factory=lambda: uow,
    )
    metrics = SandboxProvisionMetrics()

    n = await service.account_lingering_after_off(metrics)

    assert n == 3
    assert metrics.snapshot()["lingering_after_off"] == 3
