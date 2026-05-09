from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.application.services.agent_service import AgentService
from app.application.services import agent_service as agent_service_module
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _RedisClient:
    def __init__(self, reason: str) -> None:
        self.client = self
        self.reason = reason
        self.keys: list[str] = []

    async def hgetall(self, key: str) -> dict[str, str]:
        self.keys.append(key)
        return {"pending_terminal_reason": self.reason}


class _Supervisor:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    async def _on_runner_session_complete(
        self,
        *,
        session_id: str,
        user_id: str,
        cancel_reason: str | None,
    ) -> None:
        self.calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "cancel_reason": cancel_reason,
            }
        )


async def test_compose_callback_runs_supervisor_cleanup_when_original_raises() -> None:
    service = AgentService.__new__(AgentService)
    service._redis_client = _RedisClient("user_cancel")
    service._supervisor = _Supervisor()
    original_calls: list[str] = []

    async def original(session_id: str) -> None:
        original_calls.append(session_id)
        raise RuntimeError("original failed")

    composed = service._compose_completion_callbacks(
        original=original,
        session_id="session-1",
        user_id="user-1",
    )

    with pytest.raises(RuntimeError, match="original failed"):
        await composed("session-1")

    assert original_calls == ["session-1"]
    assert service._redis_client.keys == ["supervisor:hot:session-1"]
    assert service._supervisor.calls == [
        {
            "session_id": "session-1",
            "user_id": "user-1",
            "cancel_reason": "user_cancel",
        }
    ]


async def test_create_task_registers_cancelable_task_not_task_runner(
    monkeypatch,
) -> None:
    created_runners: list[object] = []

    class _FakeAgentTaskRunner:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            created_runners.append(self)

    class _FakeTask:
        id = "task-1"

        def cancel(self, reason: str = "stop") -> bool:
            return True

    class _TaskClass:
        created_with: object | None = None
        task = _FakeTask()

        @classmethod
        def create(cls, *, task_runner):
            cls.created_with = task_runner
            return cls.task

    class _SessionRepo:
        def __init__(self, session: Session) -> None:
            self.session_obj = session

        async def get_by_id(self, session_id: str) -> Session:
            assert session_id == self.session_obj.id
            return self.session_obj

        async def save(self, session: Session) -> None:
            self.session_obj = session

    class _Uow:
        def __init__(self, session: Session) -> None:
            self.session = _SessionRepo(session)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    class _Sandbox:
        id = "sandbox-1"

        async def get_browser(self):
            return object()

    class _Lifecycle:
        def __init__(self) -> None:
            self.registry = SimpleNamespace(
                register_live_event_sink=lambda *_args, **_kwargs: None
            )

        async def acquire(self, _session_id: str):
            return _Sandbox()

    class _Supervisor:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def _register_runner(self, *, session_id: str, runner: object) -> None:
            self.calls.append({"session_id": session_id, "runner": runner})

    monkeypatch.setattr(
        agent_service_module,
        "AgentTaskRunner",
        _FakeAgentTaskRunner,
    )
    import app.application.services.cost_callback_factory as cost_callback_factory

    monkeypatch.setattr(
        cost_callback_factory,
        "build_cost_callback_handler",
        lambda **_kwargs: None,
    )

    session = Session(
        id="session-1",
        user_id="user-1",
        status=SessionStatus.RUNNING,
    )
    service = AgentService.__new__(AgentService)
    service._config_snapshot = SimpleNamespace(
        llm=object(),
        agent_config=SimpleNamespace(
            tool_confirmation=SimpleNamespace(legacy_rule_fallback=False)
        ),
        mcp_config=object(),
        a2a_config=object(),
        skill_risk_policy=None,
        overflow_config=None,
        skill_creator_service=None,
        summary_llm=None,
        supports_vision=False,
        supports_pdf_input=False,
        file_understanding_config=None,
        vision_fallback_model=None,
        memory_gate_llm=None,
        memory_gate_threshold=0.8,
        memory_gate_batch_cap=10,
        tool_runtime=None,
    )
    service._sandbox_lifecycle_service = _Lifecycle()
    service._uow_factory = lambda: _Uow(session)
    service._file_storage = object()
    service._search_engine = object()
    service._checkpointer_pool = None
    service._memory_flusher = None
    service._memory_embedding_provider = None
    service._memory_session_factory = None
    service._memory_repo_factory = None
    service._memory_write_service = None
    service._redis_client = None
    service._memory_session_save_cap = 10
    service._memory_gate_breaker = None
    service._memory_gate_daily_cap = 10
    service._memory_notification_emitter = None
    service._confirmation_manager = None
    service._task_cls = _TaskClass
    service._supervisor = _Supervisor()

    task = await service._create_task(session)

    assert task is _TaskClass.task
    assert created_runners == [_TaskClass.created_with]
    assert callable(task.cancel)
    assert service._supervisor.calls == [
        {"session_id": "session-1", "runner": task}
    ]
    assert service._supervisor.calls[0]["runner"] is not _TaskClass.created_with
