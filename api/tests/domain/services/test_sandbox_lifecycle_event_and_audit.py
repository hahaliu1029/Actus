"""Tests for PR2 §10: SandboxStateChangedEvent emission + audit log writes.

Verifies that every _transition() call:
1. Persists a SandboxStateChangedEvent to session events
2. Writes an audit log entry to sandbox_lifecycle_log
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.models.event import SandboxStateChangedEvent
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
    def __init__(self, sandbox_id: str = "sbx-1") -> None:
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
    sessions: dict[str, Session],
) -> tuple[SandboxLifecycleService, FakeUoW]:
    uow = FakeUoW(sessions)
    service = SandboxLifecycleService(
        sandbox_cls=FakeSandbox,
        uow_factory=lambda: uow,
    )
    return service, uow


# ── Event emission tests ──


async def test_bind_new_emits_event_and_audit_log() -> None:
    """bind_new (UNBOUND→CREATING→ACTIVE) should emit events + audit."""
    session = Session(
        id="sess-1",
        status=SessionStatus.PENDING,
        sandbox_binding=SandboxBinding(state=SandboxBindingState.UNBOUND, generation=0),
    )
    service, uow = _make_service({"sess-1": session})

    await service.bind_new("sess-1")

    # _transition is called twice: UNBOUND→CREATING, CREATING→ACTIVE
    assert uow.session.add_event.await_count == 2
    assert uow.sandbox_lifecycle_log.create.await_count == 2

    # Check the second event (CREATING→ACTIVE)
    second_event_call = uow.session.add_event.await_args_list[1]
    event = second_event_call.args[1]
    assert isinstance(event, SandboxStateChangedEvent)
    assert event.old_state == "creating"
    assert event.new_state == "active"
    assert event.generation == 1

    # Check audit log call
    second_audit_call = uow.sandbox_lifecycle_log.create.await_args_list[1]
    assert second_audit_call.kwargs["session_id"] == "sess-1"
    assert second_audit_call.kwargs["old_state"] == "creating"
    assert second_audit_call.kwargs["new_state"] == "active"


async def test_suspend_emits_event() -> None:
    """ACTIVE→SUSPENDED emits SandboxStateChangedEvent."""
    session = Session(
        id="sess-1",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-1",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})

    await service.suspend("sess-1")

    uow.session.add_event.assert_awaited_once()
    event = uow.session.add_event.await_args.args[1]
    assert isinstance(event, SandboxStateChangedEvent)
    assert event.old_state == "active"
    assert event.new_state == "suspended"
    assert event.generation == 1  # no increment on suspend


async def test_destroy_emits_events() -> None:
    """ACTIVE→DESTROYING→DESTROYED emits two events + audit log entries."""
    session = Session(
        id="sess-1",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-1",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    await service.destroy("sess-1", DestroyReason.SESSION_DELETE)

    # Two transitions: ACTIVE→DESTROYING, DESTROYING→DESTROYED
    assert uow.session.add_event.await_count == 2
    assert uow.sandbox_lifecycle_log.create.await_count == 2

    # First event: ACTIVE→DESTROYING
    first_event = uow.session.add_event.await_args_list[0].args[1]
    assert first_event.old_state == "active"
    assert first_event.new_state == "destroying"

    # Second event: DESTROYING→DESTROYED
    second_event = uow.session.add_event.await_args_list[1].args[1]
    assert second_event.old_state == "destroying"
    assert second_event.new_state == "destroyed"
    assert second_event.reason == "session_delete"


async def test_event_includes_sandbox_id() -> None:
    """Event carries sandbox_id when binding has one."""
    session = Session(
        id="sess-1",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-42",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})

    await service.suspend("sess-1")

    event = uow.session.add_event.await_args.args[1]
    assert event.sandbox_id == "sbx-42"


async def test_resume_emits_event() -> None:
    """SUSPENDED→ACTIVE emits event."""
    session = Session(
        id="sess-1",
        status=SessionStatus.COMPLETED,
        sandbox_binding=SandboxBinding(
            id="sbx-1",
            state=SandboxBindingState.SUSPENDED,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})
    service._registry.register("sess-1", FakeSandbox("sbx-1"), generation=1)

    await service.resume("sess-1")

    uow.session.add_event.assert_awaited_once()
    event = uow.session.add_event.await_args.args[1]
    assert event.old_state == "suspended"
    assert event.new_state == "active"


async def test_live_sink_stream_id_stamps_event_for_dedup() -> None:
    """When a live event sink is present, the stream ID from output_stream.put()
    replaces the event's default UUID, so PG and Redis carry the same ID and
    get_events_since() dedup treats them as one event."""
    session = Session(
        id="sess-1",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-1",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})

    # Register a fake live sink that returns a Redis-style stream ID
    fake_stream_id = "1713264000000-0"

    async def fake_sink(evt: object) -> str:
        return fake_stream_id

    service._registry.register_live_event_sink("sess-1", fake_sink)

    await service.suspend("sess-1")

    event = uow.session.add_event.await_args.args[1]
    assert event.id == fake_stream_id, (
        "Event persisted to PG must carry the Redis stream ID "
        "so get_events_since() dedup treats live + recovered as the same event"
    )


async def test_no_sink_event_keeps_default_uuid() -> None:
    """Without a live sink, event retains its BaseEvent default UUID."""
    session = Session(
        id="sess-1",
        status=SessionStatus.RUNNING,
        sandbox_binding=SandboxBinding(
            id="sbx-1",
            state=SandboxBindingState.ACTIVE,
            generation=1,
            created_at=datetime.now(UTC),
        ),
    )
    service, uow = _make_service({"sess-1": session})

    await service.suspend("sess-1")

    event = uow.session.add_event.await_args.args[1]
    # Default UUID from BaseEvent — not a Redis stream ID
    assert "-" in event.id  # UUID format: xxxxxxxx-xxxx-...
    assert "1713" not in event.id  # not a timestamp-based stream ID
