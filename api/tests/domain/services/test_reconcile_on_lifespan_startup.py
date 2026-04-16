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
    async def create(cls):
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
