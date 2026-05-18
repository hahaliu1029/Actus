import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest
from app.application.errors.exceptions import ConflictError
from app.application.services.agent_service import AgentService
from app.domain.models.session import Session, SessionStatus

from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, session: Session | None = None) -> None:
        self.update_status_calls: list[tuple[str, SessionStatus]] = []
        self.update_to_terminal_calls: list[tuple[str, SessionStatus, str]] = []
        self.update_latest_message_calls: list[tuple[str, str]] = []
        self.add_event_calls: list[tuple[str, object]] = []
        self._session = session

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.update_status_calls.append((session_id, status))

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> None:
        self.update_to_terminal_calls.append((session_id, status, terminal_reason))
        if self._session and self._session.id == session_id:
            self._session.status = status
            self._session.terminal_reason = terminal_reason
            self._session.completed_at = datetime.now()

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        self.update_latest_message_calls.append((session_id, message))

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def get_by_id(self, session_id: str) -> Session | None:
        if self._session is None or self._session.id != session_id:
            return None
        return self._session


class _Uow:
    def __init__(self, session: Session | None = None) -> None:
        self.session = _SessionRepo(session=session)

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _DummyInputStream:
    async def put(self, event_json: str) -> str:
        return "evt-user-1"


class _DummyOutputStream:
    def __init__(self, owner: "_DummyTask") -> None:
        self._owner = owner

    async def get(self, start_id: str = None, block_ms: int = None):
        self._owner.done_flag = True
        return None, None


class _DummyTask:
    def __init__(self) -> None:
        self.done_flag = False
        self.input_stream = _DummyInputStream()
        self.output_stream = _DummyOutputStream(self)

    @property
    def done(self) -> bool:
        return self.done_flag

    async def invoke(self) -> None:
        return None


def _make_service(uow: _Uow) -> AgentService:
    return AgentService(
        uow_factory=lambda: uow,
        config_snapshot=_default_snapshot(),
        sandbox_cls=object,
        task_cls=object,
        search_engine=object(),
        file_storage=object(),
    )


async def test_chat_without_message_reconciles_running_status_when_task_missing(
    monkeypatch,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        was_background=True,
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    emitter = AsyncMock()
    service._memory_notification_emitter = emitter

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return None

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)

    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    )

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)

    assert uow.session.update_to_terminal_calls == [
        ("session-1", SessionStatus.COMPLETED, "resume_state_lost"),
    ]
    emitter.emit.assert_awaited_once_with(
        user_id="user-1",
        event_type="bg_failed_resume",
        payload={"session_id": "session-1"},
    )


async def test_chat_without_message_keeps_suspended_background_when_task_missing(
    monkeypatch,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="suspended",
        retry_budget_remaining=1,
        was_background=True,
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    emitter = AsyncMock()
    service._memory_notification_emitter = emitter

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return None

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)

    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    )

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)

    assert uow.session.update_to_terminal_calls == []
    emitter.emit.assert_not_awaited()
    assert session.execution_phase == "suspended"
    assert session.retry_budget_remaining == 1


async def test_chat_with_message_on_suspended_background_raises_conflict(
    monkeypatch,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="suspended",
        retry_budget_remaining=1,
        was_background=True,
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return None

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)

    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message="continue",
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    )

    with pytest.raises(ConflictError):
        await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)

    assert uow.session.add_event_calls == []
    assert uow.session.update_to_terminal_calls == []


async def test_chat_with_attachment_on_suspended_background_raises_conflict(
    monkeypatch,
) -> None:
    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
        execution_mode="background",
        execution_phase="suspended",
        retry_budget_remaining=1,
        was_background=True,
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return None

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)

    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message=None,
        attachments=["file-1"],
        latest_event_id=None,
        timestamp=None,
    )

    with pytest.raises(ConflictError):
        await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)

    assert uow.session.add_event_calls == []
    assert uow.session.update_to_terminal_calls == []


async def test_chat_with_message_does_not_trigger_running_status_reconcile(
    monkeypatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    created_task = _DummyTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return None

    async def fake_create_task(_session: Session, *, tool_filter=None):
        # Explicit keyword-only signature mirrors production
        # ``_create_task(self, session, *, tool_filter=None)`` so any
        # future kwarg drift (e.g. a new sigchain parameter) breaks this
        # fake noisily rather than being silently absorbed by ``**kwargs``.
        return created_task

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_safe_update_unread_count", fake_safe_update_unread_count)

    chat_gen = service.chat(
        session_id="session-1",
        user_id="user-1",
        message="hello",
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    )

    first_event = await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)
    assert first_event.type == "message"
    assert first_event.role == "user"
    assert first_event.message == "hello"
    assert uow.session.update_status_calls == []

    await chat_gen.aclose()
