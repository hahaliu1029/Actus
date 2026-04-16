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
    async def create(cls) -> "FakeSandbox":
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


async def test_destroy_idempotent_on_destroyed() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYED,
        destroyed_at=datetime.now(UTC),
    )
    service, _ = _make_service({"sess-1": session})
    await service.destroy("sess-1", DestroyReason.SESSION_DELETE)


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


# ── reconcile_orphans ──


async def test_reconcile_destroying_container_dead() -> None:
    session = _make_session(
        state=SandboxBindingState.DESTROYING, sandbox_id="sbx-1", generation=2
    )

    class NoContainerSandbox:
        @classmethod
        async def create(cls):
            return None

        @classmethod
        async def get(cls, id: str):
            return None

    service, uow = _make_service({"sess-1": session}, sandbox_cls=NoContainerSandbox)
    await service.reconcile_orphans()

    saved = uow._sessions["sess-1"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED


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
        async def create(cls):
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
