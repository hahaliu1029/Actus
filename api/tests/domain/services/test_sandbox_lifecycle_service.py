"""Unit tests for SandboxLifecycleService.

Covers §11.1: state transition matrix, acquire in each state,
bind_new concurrency, destroy quiesce barrier, reconcile_orphans,
resume idempotent on ACTIVE (eng review #5), destroy cleans lock (#7).
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

import pytest

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


from app.domain.errors.sandbox_lifecycle import (
    SandboxAlreadyDestroyed,
    SandboxBindingMissing,
    SandboxLifecycleError,
    SessionCreatingError,
    SessionDestroyingError,
    SessionFinalizedError,
    SessionSuspendedError,
    SessionUnboundError,
)
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    SandboxBindingState,
    Session,
    SessionStatus,
)
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService


# ── Helpers ──


def _make_session(
    session_id: str = "sess-1",
    state: SandboxBindingState = SandboxBindingState.UNBOUND,
    sandbox_id: Optional[str] = None,
    generation: int = 0,
    destroyed_at: Optional[datetime] = None,
    destroy_reason: Optional[DestroyReason] = None,
) -> Session:
    return Session(
        id=session_id,
        status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(
            id=sandbox_id,
            state=state,
            generation=generation,
            destroyed_at=destroyed_at,
            destroy_reason=destroy_reason,
        ),
    )


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-1") -> None:
        self._id = sandbox_id
        self._destroyed = False

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
        self._destroyed = True
        return True

    @classmethod
    async def create(cls, user_id: Optional[str] = None) -> "FakeSandbox":
        return cls()

    @classmethod
    async def get(cls, id: str) -> Optional["FakeSandbox"]:
        return cls(sandbox_id=id)


class FakeUoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.session = MagicMock()
        self.session.get_by_id = AsyncMock(side_effect=self._get_by_id)
        self.session.save = AsyncMock(side_effect=self._save)
        self.session.get_all = AsyncMock(side_effect=self._get_all)
        self.session.add_event = AsyncMock()
        self.sandbox_lifecycle_log = MagicMock()
        self.sandbox_lifecycle_log.create = AsyncMock()

    async def _get_by_id(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def _save(self, session: Session) -> None:
        self._sessions[session.id] = session

    async def _get_all(self) -> list[Session]:
        return list(self._sessions.values())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def _make_service(
    sessions: dict[str, Session] | None = None,
    sandbox_cls=FakeSandbox,
) -> tuple[SandboxLifecycleService, FakeUoW]:
    if sessions is None:
        sessions = {}
    uow = FakeUoW(sessions)
    service = SandboxLifecycleService(
        sandbox_cls=sandbox_cls,
        uow_factory=lambda: uow,
    )
    return service, uow


# ── Acquire by state ──


async def test_acquire_unbound_raises() -> None:
    session = _make_session(state=SandboxBindingState.UNBOUND)
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionUnboundError):
        await service.acquire("sess-1")


async def test_acquire_creating_raises() -> None:
    session = _make_session(state=SandboxBindingState.CREATING)
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionCreatingError):
        await service.acquire("sess-1")


async def test_acquire_suspended_raises() -> None:
    session = _make_session(
        state=SandboxBindingState.SUSPENDED, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionSuspendedError):
        await service.acquire("sess-1")


async def test_acquire_destroying_raises() -> None:
    session = _make_session(state=SandboxBindingState.DESTROYING)
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionDestroyingError):
        await service.acquire("sess-1")


async def test_acquire_destroyed_raises() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionFinalizedError):
        await service.acquire("sess-1")


async def test_acquire_active_returns_handle() -> None:
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    handle = await service.acquire("sess-1")
    assert handle.id == "sbx-1"
    assert handle.generation == 1


# ── bind_new ──


async def test_bind_new_success() -> None:
    session = _make_session(state=SandboxBindingState.UNBOUND)
    service, uow = _make_service({"sess-1": session})

    handle = await service.bind_new("sess-1")
    assert handle is not None
    assert handle.generation == 1

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.ACTIVE
    assert saved.sandbox_binding.generation == 1


async def test_bind_new_on_active_returns_handle() -> None:
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    handle = await service.bind_new("sess-1")
    assert handle.id == "sbx-1"


async def test_bind_new_on_destroyed_raises() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionFinalizedError):
        await service.bind_new("sess-1")


# ── suspend / resume ──


async def test_suspend_active() -> None:
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, uow = _make_service({"sess-1": session})
    await service.suspend("sess-1")
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.SUSPENDED
    assert saved.sandbox_binding.generation == 1


async def test_suspend_idempotent() -> None:
    session = _make_session(
        state=SandboxBindingState.SUSPENDED, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    await service.suspend("sess-1")


async def test_resume_from_suspended() -> None:
    session = _make_session(
        state=SandboxBindingState.SUSPENDED, sandbox_id="sbx-1", generation=1
    )
    service, uow = _make_service({"sess-1": session})
    handle = await service.resume("sess-1")
    assert handle.id == "sbx-1"
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.ACTIVE
    assert saved.sandbox_binding.generation == 1


async def test_resume_idempotent_on_active() -> None:
    """Eng review #5: resume() on ACTIVE returns handle without error."""
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)
    handle = await service.resume("sess-1")
    assert handle.id == "sbx-1"


async def test_resume_destroyed_raises() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SessionFinalizedError):
        await service.resume("sess-1")


# ── destroy ──


async def test_destroy_active_full_flow() -> None:
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    fake_sandbox = FakeSandbox("sbx-1")
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", fake_sandbox, generation=1)

    await service.destroy("sess-1", DestroyReason.SESSION_DELETE)

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroyed_at is not None
    assert saved.sandbox_binding.destroy_reason == DestroyReason.SESSION_DELETE
    assert saved.sandbox_binding.generation == 2
    assert fake_sandbox._destroyed


async def test_destroy_raises_already_destroyed_on_destroyed_state() -> None:
    """C3 PR-1 (spec §3.2 M2 + §7.3) — destroy on DESTROYED raises typed signal."""
    session = _make_session(
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SandboxAlreadyDestroyed) as exc:
        await service.destroy("sess-1", DestroyReason.SESSION_DELETE)
    assert exc.value.session_id == "sess-1"


async def test_destroy_raises_binding_missing_on_session_missing() -> None:
    """C3 PR-1 (spec §3.2 M2 + §7.3) — destroy on missing session raises typed signal.

    The UoW returns None for an unknown session_id; PR-1 treats this as
    SandboxBindingMissing so handlers can short-circuit as terminal-success.
    """
    service, _ = _make_service({})
    with pytest.raises(SandboxBindingMissing) as exc:
        await service.destroy("missing-sess", DestroyReason.SUBAGENT_TERMINAL_RESULT)
    assert exc.value.session_id == "missing-sess"


async def test_destroy_raises_binding_missing_on_unbound_state() -> None:
    """C3 PR-1 (plan Step 3.4) — destroy on UNBOUND raises SandboxBindingMissing.

    UNBOUND has no sandbox to tear down, so it's terminal-success identical to
    a missing row. Distinct from DESTROYED, which raises SandboxAlreadyDestroyed.
    """
    session = _make_session(state=SandboxBindingState.UNBOUND)
    service, _ = _make_service({"sess-1": session})
    with pytest.raises(SandboxBindingMissing) as exc:
        await service.destroy("sess-1", DestroyReason.SESSION_DELETE)
    assert exc.value.session_id == "sess-1"


@pytest.mark.parametrize(
    "state",
    [
        SandboxBindingState.ACTIVE,
        SandboxBindingState.SUSPENDED,
        SandboxBindingState.DESTROYING,
        SandboxBindingState.CREATING,
    ],
)
async def test_destroy_raises_binding_missing_when_state_non_terminal_but_id_none(
    state: SandboxBindingState,
) -> None:
    """C3 PR-1 (codex round 15 P2) — defensive guard for data inconsistency.

    A non-terminal state (ACTIVE / SUSPENDED / DESTROYING / CREATING) paired
    with ``binding.id is None`` means there's no real sandbox to destroy.
    Without the guard, destroy() would fall through to silent-no-op
    ``cancel_and_drain`` + ``destroy_infra`` (both no-ops because the registry
    has no entry), then mark the binding DESTROYED — polluting forensic audit
    with a phantom destroy row. The guard treats this as terminal-success
    identical to UNBOUND.

    The state must NOT transition (binding stays in its original state) and
    the lock entry must be popped to avoid a leak.
    """
    session = _make_session(state=state, sandbox_id=None)
    service, uow = _make_service({"sess-1": session})
    # Touch the lock so we can assert it gets popped.
    service._get_lock("sess-1")
    assert "sess-1" in service._per_session_locks

    with pytest.raises(SandboxBindingMissing) as exc:
        await service.destroy("sess-1", DestroyReason.SESSION_DELETE)
    assert exc.value.session_id == "sess-1"

    # State must NOT have transitioned — no phantom DESTROYED row.
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == state
    # destroy_reason must remain unset — no forensic pollution.
    assert saved.sandbox_binding.destroy_reason is None
    # Lock popped on terminal-success raise to avoid leak.
    assert "sess-1" not in service._per_session_locks


async def test_destroy_propagates_infra_failure_as_retryable() -> None:
    """C3 PR-1 (spec §7.3) — destroy_infra exceptions propagate as SandboxLifecycleError.

    Previous behavior: logger.exception + continue to mark DESTROYED. New behavior:
    propagate so callers / supervisor can distinguish forensics-success from infra
    failures requiring retry.
    """
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    async def _fail(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("docker daemon down")

    # destroy_infra is the registry-side teardown; patch the bound method.
    service._registry.destroy_infra = _fail  # type: ignore[assignment]

    with pytest.raises(SandboxLifecycleError) as exc:
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)
    assert not isinstance(exc.value, SandboxAlreadyDestroyed)
    assert not isinstance(exc.value, SandboxBindingMissing)
    assert "docker daemon down" in str(exc.value)


async def test_destroy_raises_lifecycle_error_when_destroy_infra_returns_false_via_registry() -> None:
    """C3 PR-1 (codex round 9 P1) — when ``DockerSandbox.destroy()`` returns
    ``False`` the registry translates it into ``SandboxLifecycleError``. This
    test simulates that translation at the registry boundary and asserts:

    1. The error propagates out of ``SandboxLifecycleService.destroy()``.
    2. The binding state stays ``DESTROYING`` (NOT advanced to ``DESTROYED``)
       so a subsequent reconcile / retry can finish the teardown.
    """
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    async def _registry_fail(*args, **kwargs):  # noqa: ANN001
        # Simulate what the real registry does when sandbox.destroy() == False.
        raise SandboxLifecycleError("docker remove failed")

    service._registry.destroy_infra = _registry_fail  # type: ignore[assignment]

    with pytest.raises(SandboxLifecycleError) as exc:
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)
    assert "docker remove failed" in str(exc.value)

    # Binding must remain DESTROYING (not advanced to DESTROYED).
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYING
    # Reason persisted from Phase 1 so reconcile preserves forensic class.
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE


async def test_destroy_rehydrates_registry_on_destroying_retry_with_missing_entry() -> None:
    """C3 PR-1 (codex round 13 P2) — if ``destroy()`` finds binding=DESTROYING
    but the in-memory registry is empty (e.g. process restarted between an
    earlier destroy that crashed leaving binding=DESTROYING and this retry),
    it must rehydrate via ``Sandbox.get()`` so that the subsequent
    ``destroy_infra`` actually targets the live container instead of silently
    no-oping (which would mark DESTROYED while the container leaks).
    """
    session = _make_session(
        state=SandboxBindingState.DESTROYING,
        sandbox_id="sbx-1",
        generation=2,
        destroy_reason=DestroyReason.FORCE_TERMINATE,
    )

    # Track what Sandbox.get is called with and what destroy() the rehydrated
    # sandbox sees, so we can confirm the registry was rehydrated AND used.
    get_calls: list[str] = []
    destroyed_sandbox_ids: list[str] = []

    class RehydratableSandbox:
        def __init__(self, sandbox_id: str = "sbx-1") -> None:
            self._id = sandbox_id

        @property
        def id(self) -> str:
            return self._id

        async def destroy(self) -> bool:
            destroyed_sandbox_ids.append(self._id)
            return True

        @classmethod
        async def create(cls, user_id: Optional[str] = None) -> "RehydratableSandbox":
            return cls()

        @classmethod
        async def get(cls, id: str) -> Optional["RehydratableSandbox"]:
            get_calls.append(id)
            return cls(sandbox_id=id)

    service, uow = _make_service({"sess-1": session}, sandbox_cls=RehydratableSandbox)
    # NOTE: registry is intentionally empty — simulates process restart while
    # binding was already DESTROYING.
    assert service._registry.get_sandbox("sess-1") is None

    await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)

    # Sandbox.get must have been called with the persisted binding.id so
    # destroy_infra has something to remove.
    assert get_calls == ["sbx-1"]
    # The rehydrated sandbox's destroy() must have been invoked via the
    # registry teardown path.
    assert destroyed_sandbox_ids == ["sbx-1"]
    # Binding must have advanced to DESTROYED on the happy retry path.
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED


async def test_destroy_raises_on_destroying_retry_when_rehydrate_returns_none() -> None:
    """C3 PR-1 (codex round 14 P2) — ``Sandbox.get()`` returning ``None`` is
    AMBIGUOUS in production: ``DockerSandbox.get()`` collapses both
    ``NotFound`` (terminal success) and ``APIError`` (transient daemon
    failure) into ``None``. The conservative posture is to preserve
    DESTROYING by raising ``SandboxLifecycleError`` so the next reconcile
    pass / operator retry resolves it once Docker is reachable. Premature
    DESTROYED would silently mark a LIVE container as gone during a Docker
    outage.

    A future ``DockerSandbox.get()`` refactor (planned for PR-3a supervisor
    lifecycle integration) will distinguish NotFound from APIError and
    enable a clean terminal-success short-circuit. Until then, defer to
    reconcile.
    """
    session = _make_session(
        state=SandboxBindingState.DESTROYING,
        sandbox_id="sbx-1",
        generation=2,
        destroy_reason=DestroyReason.FORCE_TERMINATE,
    )

    class NoContainerSandbox:
        @classmethod
        async def create(cls, user_id: Optional[str] = None):
            return None

        @classmethod
        async def get(cls, id: str):
            return None

    service, uow = _make_service({"sess-1": session}, sandbox_cls=NoContainerSandbox)
    assert service._registry.get_sandbox("sess-1") is None

    # Conservative contract: preserve DESTROYING via raise.
    with pytest.raises(SandboxLifecycleError) as exc:
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)
    assert "could not rehydrate" in str(exc.value)

    saved = uow._sessions["sess-1"]
    # Binding state must remain DESTROYING (NOT advanced to DESTROYED).
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYING
    # Original destroy_reason preserved for forensic classification.
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE
    # Registry must remain empty — no phantom entry was registered.
    assert service._registry.get_sandbox("sess-1") is None


async def test_destroy_syncs_registry_generation_after_destroying_transition() -> None:
    """C3 PR-1 (codex round 10 P2) — destroy() must sync the in-memory
    registry generation after the DB DESTROYING transition bumps it.

    Without this sync, any in-flight ``SandboxHandle`` that captured the
    OLD generation still passes ``_check_generation()`` against the stale
    registry entry, allowing command dispatch at a DESTROYING sandbox
    after ``destroy_infra`` raises and the registry entry stays alive for
    retry. The invariant: once destroy() advances DESTROYING + generation++,
    the registry's generation MUST match the new DB value even on the
    failure path.

    Setup: ACTIVE session at generation=5, registry entry at generation=5.
    Action: stub ``destroy_infra`` to raise, call destroy(), expect raise.
    Assert: ``service._registry.get_generation(session_id) == 6``.
    """
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=5
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=5)

    async def _fail(*args, **kwargs):  # noqa: ANN001
        raise SandboxLifecycleError("docker remove failed")

    service._registry.destroy_infra = _fail  # type: ignore[assignment]

    with pytest.raises(SandboxLifecycleError):
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)

    # Registry generation must be synced to the new DB value (5 + 1 = 6)
    # even though destroy_infra raised and the entry stays alive for retry.
    assert service._registry.get_generation("sess-1") == 6


async def test_destroy_normal_path_returns_none() -> None:
    """C3 PR-1 (spec §7.3) — ACTIVE → DESTROYED happy path still returns None."""
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    result = await service.destroy("sess-1", DestroyReason.SESSION_DELETE)
    assert result is None


async def test_destroy_cleans_lock() -> None:
    """Eng review #3: lock cleanup after destroy."""
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)
    service._get_lock("sess-1")
    assert "sess-1" in service._per_session_locks

    await service.destroy("sess-1", DestroyReason.SESSION_DELETE)
    assert "sess-1" not in service._per_session_locks


async def test_destroy_pops_lock_on_terminal_success_raise() -> None:
    """C3 PR-1 (codex round 6 P2) — terminal-success raises must pop the
    per-session lock entry to avoid lock-dict leaks.

    Covers three paths:
      1. UNBOUND state → SandboxBindingMissing
      2. DESTROYED state → SandboxAlreadyDestroyed
      3. missing session row → SandboxBindingMissing

    Each path goes through ``async with self._get_lock(session_id)`` which
    creates the lock if missing. Without explicit pop, every never-bound
    session deleted from the UI would leak a lock object over the process
    lifetime.
    """
    # Path 1: UNBOUND
    session_unbound = _make_session("sess-unbound", state=SandboxBindingState.UNBOUND)
    service, _ = _make_service({"sess-unbound": session_unbound})
    with pytest.raises(SandboxBindingMissing):
        await service.destroy("sess-unbound", DestroyReason.SESSION_DELETE)
    assert "sess-unbound" not in service._per_session_locks

    # Path 2: DESTROYED
    session_destroyed = _make_session(
        "sess-destroyed",
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service2, _ = _make_service({"sess-destroyed": session_destroyed})
    with pytest.raises(SandboxAlreadyDestroyed):
        await service2.destroy("sess-destroyed", DestroyReason.SESSION_DELETE)
    assert "sess-destroyed" not in service2._per_session_locks

    # Path 3: missing session row
    service3, _ = _make_service({})
    with pytest.raises(SandboxBindingMissing):
        await service3.destroy("sess-missing", DestroyReason.SESSION_DELETE)
    assert "sess-missing" not in service3._per_session_locks


async def test_destroy_reason_preserved_when_retried_with_different_reason() -> None:
    """C3 PR-1 (codex round 6 P2) — DESTROYING-resume path must preserve the
    originally persisted destroy_reason, even if a later destroy() is called
    with a different reason.

    Scenario:
      Phase 1: destroy(sess, FORCE_TERMINATE) raises (destroy_infra fails).
               Binding stays DESTROYING with reason=FORCE_TERMINATE.
      Phase 2: destroy(sess, SESSION_DELETE) succeeds (destroy_infra now works).
               Binding ends DESTROYED with destroy_reason=FORCE_TERMINATE
               (the original forensic), NOT SESSION_DELETE.
    """
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    # Phase 1: destroy_infra fails → binding goes DESTROYING + reason=FORCE_TERMINATE.
    async def _fail(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("docker daemon down")

    service._registry.destroy_infra = _fail  # type: ignore[assignment]

    with pytest.raises(SandboxLifecycleError):
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYING
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE

    # Phase 2: caller retries with a different reason; destroy_infra now succeeds.
    async def _succeed(*args, **kwargs):  # noqa: ANN001
        return None

    service._registry.destroy_infra = _succeed  # type: ignore[assignment]

    # Re-register so the registry has the entry the destroy() flow expects.
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=2)

    await service.destroy("sess-1", DestroyReason.SESSION_DELETE)

    # Final state: DESTROYED, with the ORIGINAL FORCE_TERMINATE reason preserved,
    # NOT the new SESSION_DELETE.
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE


# ── reconcile_orphans ──


async def test_reconcile_destroying_container_dead() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYING, sandbox_id="sbx-1", generation=2
    )

    class NoContainerSandbox:
        @classmethod
        async def create(cls, user_id: Optional[str] = None):
            return None

        @classmethod
        async def get(cls, id: str):
            return None

    service, uow = _make_service({"sess-1": session}, sandbox_cls=NoContainerSandbox)
    await service.reconcile_orphans()

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED


async def test_reconcile_orphans_keeps_destroying_on_infra_failure() -> None:
    """C3 PR-1 (codex round 3 P2) — reconcile_orphans must NOT transition
    DESTROYING → DESTROYED when destroy_infra raises.

    Previous behavior: swallow exception + still mark DESTROYED, leaving a
    possibly-live container leaking. New behavior: leave binding in DESTROYING
    so the next reconcile pass retries destroy_infra.
    """
    session = _make_session(
        state=SandboxBindingState.DESTROYING,
        sandbox_id="sbx-1",
        generation=2,
        destroy_reason=DestroyReason.FORCE_TERMINATE,
    )
    service, uow = _make_service({"sess-1": session})

    async def _fail(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("docker down")

    # Stub registry.destroy_infra to raise after the sandbox is registered.
    service._registry.destroy_infra = _fail  # type: ignore[assignment]

    await service.reconcile_orphans()

    saved = uow._sessions["sess-1"]
    # State must still be DESTROYING (the next reconcile pass will retry).
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYING
    # destroy_reason must not be overwritten with RECONCILE_ORPHAN.
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE
    # C3 PR-1 (codex round 4 P2): registry entry MUST be preserved so the next
    # reconcile pass / explicit destroy() can retry destroy_infra against the
    # live container. Clearing it would make subsequent destroy_infra a silent
    # no-op and leak the container.
    assert service._registry.get_sandbox("sess-1") is not None


async def test_destroy_reason_preserved_across_destroy_infra_retry() -> None:
    """C3 PR-1 (codex round 4 P2) — destroy(FORCE_TERMINATE) that fails infra
    must persist destroy_reason in DESTROYING, so a later reconcile_orphans pass
    does not overwrite it with RECONCILE_ORPHAN.

    Scenario:
      1. destroy(session, FORCE_TERMINATE) is called.
      2. destroy_infra raises → binding stays DESTROYING.
      3. reconcile_orphans runs (with destroy_infra now working).
      4. Final binding.destroy_reason must equal FORCE_TERMINATE, NOT
         RECONCILE_ORPHAN — preserving forensic classification.
    """
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    # Phase 1: destroy_infra fails.
    async def _fail(*args, **kwargs):  # noqa: ANN001
        raise RuntimeError("docker daemon down")

    service._registry.destroy_infra = _fail  # type: ignore[assignment]

    with pytest.raises(SandboxLifecycleError):
        await service.destroy("sess-1", DestroyReason.FORCE_TERMINATE)

    # Binding must be DESTROYING with destroy_reason ALREADY persisted
    # (not waiting for the DESTROYED step).
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYING
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE

    # Phase 2: reconcile picks up — destroy_infra now succeeds.
    async def _succeed(*args, **kwargs):  # noqa: ANN001
        return None

    service._registry.destroy_infra = _succeed  # type: ignore[assignment]

    await service.reconcile_orphans()

    # Final state: DESTROYED, with the ORIGINAL FORCE_TERMINATE reason preserved.
    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroy_reason == DestroyReason.FORCE_TERMINATE


# ── rehydrate / orphan ──


async def test_rehydrate_on_registry_miss() -> None:
    """ACTIVE binding + empty registry + container alive → rehydrate success."""
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )
    service, _ = _make_service({"sess-1": session})
    handle = await service.acquire("sess-1")
    assert handle.id == "sbx-1"


async def test_orphan_transitions_to_destroyed() -> None:
    """ACTIVE binding + empty registry + container dead → DESTROYED."""
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=1
    )

    class NoContainerSandbox:
        @classmethod
        async def create(cls, user_id: Optional[str] = None):
            return None

        @classmethod
        async def get(cls, id: str):
            return None

    service, uow = _make_service({"sess-1": session}, sandbox_cls=NoContainerSandbox)
    with pytest.raises(SessionFinalizedError):
        await service.acquire("sess-1")

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroy_reason == DestroyReason.RECONCILE_ORPHAN


# ── SandboxBinding serialization (eng review #8) ──


def test_sandbox_binding_serialization_roundtrip() -> None:
    binding = SandboxBinding(
        id="sbx-1",
        state=SandboxBindingState.ACTIVE,
        generation=3,
        created_at=datetime(2026, 4, 16, tzinfo=UTC),
    )
    data = binding.model_dump(mode="json")
    restored = SandboxBinding.model_validate(data)
    assert restored == binding
    assert restored.state == SandboxBindingState.ACTIVE
    assert restored.generation == 3


def test_sandbox_binding_frozen() -> None:
    binding = SandboxBinding()
    with pytest.raises(Exception):
        binding.state = SandboxBindingState.ACTIVE  # type: ignore[misc]


# ── C2 coordinator-cancel Part B: try_register_from_binding (tests 14–15) ──


class _GoneSandbox(FakeSandbox):
    """Sandbox class whose ``get`` always returns None (container removed /
    daemon unreachable — both collapse to None in DockerSandbox.get)."""

    @classmethod
    async def get(cls, id: str):
        return None


async def test_try_register_active_binding_registers_and_destroy_kills_real_container() -> None:
    session = _make_session(
        state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1", generation=2
    )
    service, _ = _make_service({"sess-1": session})
    assert service._registry.get_sandbox("sess-1") is None  # fresh process

    ok = await service.try_register_from_binding("sess-1")
    assert ok is True
    registered = service._registry.get_sandbox("sess-1")
    assert registered is not None

    # A subsequent destroy() now docker-rm's the REAL registered container
    # (not just marking the row DESTROYED while the container leaks — R4 P1).
    await service.destroy("sess-1", DestroyReason.TERMINAL_CHILD_REAPER)
    assert registered._destroyed is True
    assert session.sandbox_binding.state == SandboxBindingState.DESTROYED


async def test_try_register_already_populated_returns_true_without_reget() -> None:
    session = _make_session(state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1")
    service, _ = _make_service({"sess-1": session})
    sentinel = FakeSandbox(sandbox_id="already")
    service._registry.register("sess-1", sentinel, generation=0)

    ok = await service.try_register_from_binding("sess-1")
    assert ok is True
    # in-process entry preserved; NOT re-fetched / replaced
    assert service._registry.get_sandbox("sess-1") is sentinel


async def test_try_register_container_gone_returns_false_no_state_change() -> None:
    session = _make_session(state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1")
    service, _ = _make_service({"sess-1": session}, sandbox_cls=_GoneSandbox)

    ok = await service.try_register_from_binding("sess-1")
    assert ok is False
    assert service._registry.get_sandbox("sess-1") is None
    # row left ACTIVE — never silently DESTROYED (R5 P2 / INV-C7)
    assert session.sandbox_binding.state == SandboxBindingState.ACTIVE


async def test_try_register_unbound_returns_false() -> None:
    session = _make_session(state=SandboxBindingState.UNBOUND)
    service, _ = _make_service({"sess-1": session})
    assert await service.try_register_from_binding("sess-1") is False


async def test_try_register_missing_row_returns_false() -> None:
    service, _ = _make_service({})
    assert await service.try_register_from_binding("nope") is False


async def test_try_register_false_paths_do_not_leak_locks() -> None:
    # codex final-audit P3: a False return has no destroy() handoff, so the
    # per-session lock created by _get_lock must be popped (mirrors destroy()'s
    # pop-to-avoid-leak). Without the pop a startup sweep of many gone children
    # would accumulate dead asyncio.Lock objects in _per_session_locks.
    gone = _make_session(state=SandboxBindingState.ACTIVE, sandbox_id="sbx-1")
    service, _ = _make_service({"sess-1": gone}, sandbox_cls=_GoneSandbox)
    assert await service.try_register_from_binding("sess-1") is False
    assert "sess-1" not in service._per_session_locks  # gone container path

    assert await service.try_register_from_binding("nope") is False
    assert "nope" not in service._per_session_locks  # missing-row path

    unbound = _make_session(state=SandboxBindingState.UNBOUND)
    service2, _ = _make_service({"sess-2": unbound})
    assert await service2.try_register_from_binding("sess-2") is False
    assert "sess-2" not in service2._per_session_locks  # wrong-state path

    # Sandbox.get RAISES path (codex R2 P3): pins the except-branch _pop_lock_for
    # specifically — without it a revert of only that branch would slip through.
    class _RaisingSandbox(FakeSandbox):
        @classmethod
        async def get(cls, id: str):
            raise RuntimeError("docker daemon unreachable")

    raising = _make_session(state=SandboxBindingState.ACTIVE, sandbox_id="sbx-3")
    service3, _ = _make_service({"sess-3": raising}, sandbox_cls=_RaisingSandbox)
    assert await service3.try_register_from_binding("sess-3") is False
    assert "sess-3" not in service3._per_session_locks  # Sandbox.get-raises path
