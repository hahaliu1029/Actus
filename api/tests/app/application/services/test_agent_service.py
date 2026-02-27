import asyncio
from typing import Optional

import pytest
from app.application.errors.exceptions import BadRequestError
from app.application.services.agent_service import AgentService
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.latest_message_calls: list[dict] = []
        self.add_event_calls: list[tuple[str, object]] = []

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        return None

    async def update_latest_message(
        self, session_id: str, message: str, timestamp
    ) -> None:
        self.latest_message_calls.append(
            {
                "session_id": session_id,
                "message": message,
                "timestamp": timestamp,
            }
        )

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _NoopUoW:
    def __init__(self) -> None:
        self.session = _NoopSessionRepository()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class _DummyOutputStream:
    def __init__(self, owner: "_DummyTask") -> None:
        self._owner = owner
        self.block_ms_calls: list[Optional[int]] = []

    async def get(self, start_id: str = None, block_ms: int = None):
        self.block_ms_calls.append(block_ms)
        if block_ms == 0:
            await asyncio.sleep(1)
            return None, None

        self._owner.done_flag = True
        return None, None


class _DummyTask:
    def __init__(self) -> None:
        self.done_flag = False
        self.output_stream = _DummyOutputStream(self)
        self.input_stream = _DummyInputStream()

    @property
    def done(self) -> bool:
        return self.done_flag

    async def invoke(self) -> None:
        return None


class _DummyTaskClass:
    @classmethod
    def get(cls, task_id: str):
        return None

    @classmethod
    def create(cls, task_runner):
        return _DummyTask()

    @classmethod
    async def destroy(cls) -> None:
        return None


def _uow_factory() -> _NoopUoW:
    return _NoopUoW()


class _DummyInputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, event_json: str) -> str:
        self.events.append(event_json)
        return "evt-user-1"


async def test_chat_polling_does_not_block_when_no_output_event(monkeypatch) -> None:
    service = AgentService(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )
    task = _DummyTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return task

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(
        service,
        "_check_attachments_access",
        fake_check_attachments_access,
    )
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(
        service,
        "_safe_update_unread_count",
        fake_safe_update_unread_count,
    )

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

    assert task.output_stream.block_ms_calls
    assert task.output_stream.block_ms_calls[0] is not None
    assert task.output_stream.block_ms_calls[0] > 0


async def test_chat_with_message_yields_user_message_event_immediately(monkeypatch) -> None:
    service = AgentService(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )
    task = _DummyTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=SessionStatus.RUNNING)

    async def fake_check_attachments_access(*args, **kwargs) -> None:
        return None

    async def fake_get_task(_session: Session):
        return task

    async def fake_safe_update_unread_count(_session_id: str) -> None:
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_check_attachments_access", fake_check_attachments_access)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
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


@pytest.mark.parametrize(
    "session_status",
    [SessionStatus.TAKEOVER_PENDING, SessionStatus.TAKEOVER],
)
async def test_chat_with_message_forbidden_in_takeover_states(
    monkeypatch,
    session_status: SessionStatus,
) -> None:
    service = AgentService(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=object,
        task_cls=_DummyTaskClass,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="session-1", user_id="user-1", status=session_status)

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
        message="hello",
        attachments=None,
        latest_event_id=None,
        timestamp=None,
    )

    with pytest.raises(BadRequestError):
        await asyncio.wait_for(chat_gen.__anext__(), timeout=0.2)
