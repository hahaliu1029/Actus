import asyncio

import pytest
from app.application.services.agent_service import AgentService
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self) -> None:
        self.update_status_calls: list[tuple[str, SessionStatus]] = []
        self.update_latest_message_calls: list[tuple[str, str]] = []
        self.add_event_calls: list[tuple[str, object]] = []

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.update_status_calls.append((session_id, status))

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        self.update_latest_message_calls.append((session_id, message))

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None


class _Uow:
    def __init__(self) -> None:
        self.session = _SessionRepo()

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
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=object,
        task_cls=object,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )


async def test_chat_without_message_reconciles_running_status_when_task_missing(
    monkeypatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)

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

    assert uow.session.update_status_calls == [
        ("session-1", SessionStatus.COMPLETED),
    ]


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

    async def fake_create_task(_session: Session):
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
