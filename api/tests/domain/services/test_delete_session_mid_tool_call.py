"""Integration test: delete_session while a tool call is in flight.

§11.2: Constructs a session with an active sandbox handle executing a slow
operation, then triggers destroy(). Expects:
- The in-flight task is cancelled
- Sandbox destroy succeeds
- binding.state == DESTROYED, destroyed_at set
- Subsequent acquire raises SessionFinalizedError
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.errors.sandbox_lifecycle import SessionFinalizedError
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


class SlowSandbox:
    """Sandbox stub whose exec_command takes 10s (simulates in-flight tool call)."""

    def __init__(self) -> None:
        self._id = "sbx-slow"
        self._destroyed = False

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

    async def exec_command(self, *args, **kwargs):
        await asyncio.sleep(10)
        return MagicMock(success=True)

    async def ensure_sandbox(self) -> None:
        pass

    async def destroy(self) -> bool:
        self._destroyed = True
        return True

    @classmethod
    async def create(cls):
        return cls()

    @classmethod
    async def get(cls, id: str):
        return cls()


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


async def test_destroy_cancels_inflight_and_finalizes() -> None:
    """destroy() during in-flight exec_command → cancel + DESTROYED."""
    session = Session(
        id="sess-mid",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-slow",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    uow = FakeUoW({"sess-mid": session})
    service = SandboxLifecycleService(
        sandbox_cls=SlowSandbox,
        uow_factory=lambda: uow,
    )

    # Acquire a handle and start a slow operation on it
    handle = await service.acquire("sess-mid")

    slow_done = asyncio.Event()

    async def slow_tool_call():
        try:
            await handle.exec_command("sess-mid", "/root", "sleep 10")
        except (asyncio.CancelledError, Exception):
            pass
        finally:
            slow_done.set()

    tool_task = asyncio.create_task(slow_tool_call())
    await asyncio.sleep(0.05)  # Let the tool call start

    # Destroy while the tool call is in flight
    await service.destroy("sess-mid", DestroyReason.SESSION_DELETE)

    # Tool call should have been cancelled
    await asyncio.wait_for(slow_done.wait(), timeout=5.0)

    # Verify final state
    saved = uow._sessions["sess-mid"]
    assert saved.sandbox_binding.state == SandboxBindingState.DESTROYED
    assert saved.sandbox_binding.destroyed_at is not None
    assert saved.sandbox_binding.destroy_reason == DestroyReason.SESSION_DELETE

    # Subsequent acquire should raise
    with pytest.raises(SessionFinalizedError):
        await service.acquire("sess-mid")
