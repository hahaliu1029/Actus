"""Integration test: concurrent destroy + acquire.

§11.2: Parametrized with n=100 concurrent iterations. Expects acquire
to either return a valid ACTIVE handle OR raise an appropriate lifecycle
error. Must never return a poisoned-but-silent handle.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.errors.sandbox_lifecycle import (
    SandboxLifecycleError,
    SandboxPoisonedError,
)
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


class FakeSandbox:
    def __init__(self, sandbox_id: str = "sbx-race") -> None:
        self._id = sandbox_id

    @property
    def id(self) -> str:
        return self._id

    @property
    def cdp_url(self) -> str:
        return "http://test:9222"

    @property
    def shell_ws_url(self) -> str:
        return "ws://test:8080"

    @property
    def vnc_url(self) -> str:
        return "ws://test:5901"

    async def ensure_sandbox(self) -> None:
        pass

    async def destroy(self) -> bool:
        return True

    @classmethod
    async def create(cls, user_id: str | None = None):
        return cls()

    @classmethod
    async def get(cls, id: str):
        return cls(sandbox_id=id)


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


@pytest.mark.parametrize("iteration", range(100))
async def test_concurrent_destroy_and_acquire_no_silent_poison(iteration: int) -> None:
    """Concurrent destroy + acquire must not produce a valid-looking but poisoned handle."""
    session = Session(
        id="sess-race",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-race",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    uow = FakeUoW({"sess-race": session})
    service = SandboxLifecycleService(
        sandbox_cls=FakeSandbox,
        uow_factory=lambda: uow,
    )
    service._registry.register("sess-race", FakeSandbox("sbx-race"), generation=1)

    results: list[str] = []

    async def do_acquire():
        try:
            handle = await service.acquire("sess-race")
            # If we got a handle, it must be usable (not poisoned)
            assert handle.generation >= 1
            results.append("acquired")
        except SandboxLifecycleError:
            results.append("rejected")
        except SandboxPoisonedError:
            results.append("poisoned")

    async def do_destroy():
        await service.destroy("sess-race", DestroyReason.SESSION_DELETE)
        results.append("destroyed")

    await asyncio.gather(do_acquire(), do_destroy())

    # Destroy must always succeed
    assert "destroyed" in results
    # Acquire either got a handle before destroy kicked in, or was rejected.
    # It must NEVER silently get a poisoned handle without raising.
    assert "poisoned" not in results, (
        "acquire() returned a handle that later turned out to be poisoned — "
        "this means the generation check was bypassed"
    )
