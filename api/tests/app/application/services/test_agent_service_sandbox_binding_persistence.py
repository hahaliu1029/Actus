from __future__ import annotations

from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.agent_service import AgentService
from app.application.services.sandbox_lifecycle_service import SandboxLifecycleService
from app.domain.models.session import SandboxBindingState, Session, SessionStatus
from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeSandbox:
    _counter = 0

    def __init__(self, sandbox_id: str) -> None:
        self._id = sandbox_id

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
        return None

    async def destroy(self) -> bool:
        return True

    async def get_browser(self):
        return MagicMock()

    @classmethod
    async def create(cls) -> "_FakeSandbox":
        cls._counter += 1
        return cls(f"sbx-{cls._counter}")

    @classmethod
    async def get(cls, sandbox_id: str) -> Optional["_FakeSandbox"]:
        return cls(sandbox_id)


class _SessionRepo:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions

    async def get_by_id(self, session_id: str) -> Optional[Session]:
        session = self._sessions.get(session_id)
        return session.model_copy(deep=True) if session is not None else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def add_event(self, session_id: str, event) -> None:
        del session_id, event
        return None


class _SandboxLifecycleLogRepo:
    def __init__(self) -> None:
        self.create = AsyncMock()


class _UoW:
    def __init__(self, sessions: dict[str, Session]) -> None:
        self._sessions = sessions
        self.session = _SessionRepo(sessions)
        self.sandbox_lifecycle_log = _SandboxLifecycleLogRepo()

    async def __aenter__(self) -> "_UoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None


class _DummyTask:
    def __init__(self) -> None:
        self.id = "task-1"
        self.output_stream = MagicMock()
        self.output_stream.put = AsyncMock(return_value="evt-1")


class _DummyTaskClass:
    @classmethod
    def create(cls, task_runner):
        del task_runner
        return _DummyTask()


async def test_create_task_with_lifecycle_does_not_overwrite_new_binding(monkeypatch) -> None:
    _FakeSandbox._counter = 0
    session = Session(
        id="sess-1",
        status=SessionStatus.PENDING,
    )
    sessions = {"sess-1": session}
    uow = _UoW(sessions)
    lifecycle = SandboxLifecycleService(
        sandbox_cls=_FakeSandbox,
        uow_factory=lambda: uow,
    )
    service = AgentService(
        uow_factory=lambda: uow,
        config_snapshot=_default_snapshot(),
        sandbox_cls=_FakeSandbox,
        task_cls=_DummyTaskClass,
        search_engine=MagicMock(),
        file_storage=MagicMock(),
        sandbox_lifecycle_service=lifecycle,
    )

    monkeypatch.setattr(
        "app.application.services.agent_service.AgentTaskRunner",
        lambda *args, **kwargs: MagicMock(),
    )

    await service._create_task(session)

    saved = sessions["sess-1"]
    assert saved.task_id == "task-1"
    assert saved.sandbox_binding.id == "sbx-1"
    assert saved.sandbox_binding.state == SandboxBindingState.ACTIVE
    assert saved.sandbox_binding.generation == 1
