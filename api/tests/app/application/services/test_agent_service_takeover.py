import asyncio
import json
from datetime import datetime

import pytest
from app.application.services.agent_service import AgentService
from app.application.errors.exceptions import BadRequestError, ConflictError, ForbiddenError
from app.domain.models.app_config import A2AConfig, AgentConfig, MCPConfig
from app.domain.models.event import ControlAction, ControlEvent, ControlScope, ControlSource
from app.domain.models.session import Session, SessionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, session: Session | None = None) -> None:
        self.update_status_calls: list[tuple[str, SessionStatus]] = []
        self.add_event_calls: list[tuple[str, object]] = []
        self.get_by_id_for_update_calls: list[str] = []
        self._session = session

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.update_status_calls.append((session_id, status))
        # 模拟 completed_at 行为：COMPLETED 时设置，TAKEOVER_PENDING 时清空
        if self._session and self._session.id == session_id:
            if status == SessionStatus.COMPLETED:
                self._session.completed_at = datetime.now()
            elif status == SessionStatus.TAKEOVER_PENDING:
                self._session.completed_at = None

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))

    async def get_by_id(self, session_id: str):
        if not self._session:
            return None
        return self._session if self._session.id == session_id else None

    async def get_by_id_for_update(self, session_id: str):
        self.get_by_id_for_update_calls.append(session_id)
        return await self.get_by_id(session_id)


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


class _DummyOutputStream:
    def __init__(self) -> None:
        self.put_payloads: list[str] = []

    async def put(self, payload: str) -> str:
        self.put_payloads.append(payload)
        return "evt-control-1"


class _DummyInputStream:
    def __init__(self) -> None:
        self.put_payloads: list[str] = []

    async def put(self, payload: str) -> str:
        self.put_payloads.append(payload)
        return "evt-input-1"


class _ResumableTask:
    def __init__(self) -> None:
        self.input_stream = _DummyInputStream()
        self.output_stream = _DummyOutputStream()
        self.invoke_called = False
        self.cancel_reason: str | None = None
        self.done = False

    async def invoke(self) -> None:
        self.invoke_called = True

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_reason = reason
        return True


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


async def test_start_takeover_from_pending_updates_status_and_appends_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER_PENDING)
    append_calls: list[dict] = []
    timeout_calls: list[dict] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    def fake_schedule_takeover_timeout(
        *,
        session_id: str,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int,
    ) -> None:
        timeout_calls.append(
            {
                "session_id": session_id,
                "takeover_id": takeover_id,
                "operator_user_id": operator_user_id,
                "ttl_seconds": ttl_seconds,
            }
        )

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_schedule_takeover_timeout", fake_schedule_takeover_timeout)

    result = await service.start_takeover("s1", "u1", scope="shell")

    assert result["status"] == SessionStatus.TAKEOVER
    assert result["request_status"] == "started"
    assert result["scope"] == "shell"
    assert isinstance(result["expires_at"], int)
    assert uow.session.update_status_calls == [("s1", SessionStatus.TAKEOVER)]
    assert append_calls[0]["action"] == ControlAction.STARTED
    assert append_calls[0]["source"] == ControlSource.USER
    assert append_calls[0]["scope"] == ControlScope.SHELL
    assert isinstance(append_calls[0]["expires_at"], datetime)
    assert timeout_calls
    assert timeout_calls[0]["session_id"] == "s1"
    assert timeout_calls[0]["operator_user_id"] == "u1"


async def test_start_takeover_running_returns_starting_and_schedules_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.RUNNING)
    schedule_calls: list[dict] = []

    class _StuckTask:
        def __init__(self) -> None:
            self.done = False
            self.cancel_reason: str | None = None

        cancel_reason: str | None = None

        def cancel(self, reason: str = "stop") -> bool:
            self.cancel_reason = reason
            return True

    task = _StuckTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_get_task(_session: Session):
        return task

    def fake_schedule_takeover_completion(**kwargs):
        schedule_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_get_task", fake_get_task)
    monkeypatch.setattr(service, "_schedule_takeover_completion", fake_schedule_takeover_completion)

    result = await service.start_takeover(
        "s1",
        "u1",
        scope="browser",
        cancel_timeout_seconds=1,
    )

    assert task.cancel_reason == "takeover_start"
    assert result["request_status"] == "starting"
    assert result["status"] == SessionStatus.RUNNING
    assert result["scope"] == "browser"
    assert result["takeover_id"] is not None
    assert isinstance(result["expires_at"], int)
    assert uow.session.update_status_calls == []
    assert schedule_calls
    assert schedule_calls[0]["session_id"] == "s1"
    assert schedule_calls[0]["task"] is task
    assert schedule_calls[0]["scope"] == ControlScope.BROWSER
    assert schedule_calls[0]["operator_user_id"] == "u1"
    assert schedule_calls[0]["cancel_timeout_seconds"] == 1


async def test_reject_takeover_continue_switches_back_to_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER_PENDING,
        events=[
            ControlEvent(
                action=ControlAction.REQUESTED,
                source=ControlSource.AGENT,
                scope=ControlScope.SHELL,
                takeover_id="tk_pending_1",
            )
        ],
    )
    append_calls: list[dict] = []
    task = _ResumableTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_create_task(_session: Session):
        return task

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)

    result = await service.reject_takeover("s1", "u1", decision="continue")

    assert result == {"status": SessionStatus.RUNNING, "reason": "continue"}
    assert task.invoke_called is True
    assert len(task.input_stream.put_payloads) == 1
    handoff_payload = json.loads(task.input_stream.put_payloads[0])
    assert handoff_payload["role"] == "system"
    assert "用户拒绝接管请求" in handoff_payload["message"]
    assert uow.session.update_status_calls == [("s1", SessionStatus.RUNNING)]
    assert append_calls[0]["action"] == ControlAction.REJECTED
    assert append_calls[0]["reason"] == "continue"
    assert append_calls[0]["takeover_id"] == "tk_pending_1"
    assert append_calls[0]["source"] == ControlSource.USER
    assert append_calls[0]["task"] is task


async def test_reject_takeover_terminate_marks_completed_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER_PENDING,
        events=[
            ControlEvent(
                action=ControlAction.REQUESTED,
                source=ControlSource.AGENT,
                scope=ControlScope.SHELL,
                takeover_id="tk_pending_terminate",
            )
        ],
    )
    append_calls: list[dict] = []
    release_calls: list[str] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    async def fake_force_release_takeover_lease(session_id: str):
        release_calls.append(session_id)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release_takeover_lease)

    result = await service.reject_takeover("s1", "u1", decision="terminate")

    assert result == {"status": SessionStatus.COMPLETED, "reason": "terminate"}
    assert uow.session.update_status_calls == [("s1", SessionStatus.COMPLETED)]
    assert append_calls[0]["action"] == ControlAction.REJECTED
    assert append_calls[0]["reason"] == "terminate"
    assert append_calls[0]["takeover_id"] == "tk_pending_terminate"
    assert release_calls == ["s1"]


async def test_end_takeover_complete_marks_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.SHELL,
                takeover_id="tk_ended_1",
            )
        ],
    )
    append_calls: list[dict] = []
    release_calls: list[dict] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    async def fake_release_takeover_lease(_session_id: str, **kwargs):
        release_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_release_takeover_lease", fake_release_takeover_lease)

    result = await service.end_takeover("s1", "u1", handoff_mode="complete")

    assert result == {"status": SessionStatus.COMPLETED, "handoff_mode": "complete"}
    assert uow.session.update_status_calls == [("s1", SessionStatus.COMPLETED)]
    assert append_calls[0]["action"] == ControlAction.ENDED
    assert append_calls[0]["handoff_mode"] == "complete"
    assert append_calls[0]["takeover_id"] == "tk_ended_1"
    assert append_calls[0]["source"] == ControlSource.USER
    assert release_calls == [
        {"takeover_id": "tk_ended_1", "operator_user_id": "u1"}
    ]


async def test_end_takeover_continue_passes_takeover_id_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.SHELL,
                takeover_id="tk_continue_1",
            )
        ],
    )
    append_calls: list[dict] = []
    release_calls: list[dict] = []
    task = _ResumableTask()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_create_task(_session: Session):
        return task

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    async def fake_release_takeover_lease(_session_id: str, **kwargs):
        release_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_create_task", fake_create_task)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_release_takeover_lease", fake_release_takeover_lease)

    result = await service.end_takeover("s1", "u1", handoff_mode="continue")

    assert result == {"status": SessionStatus.RUNNING, "handoff_mode": "continue"}
    assert task.invoke_called is True
    assert uow.session.update_status_calls == [("s1", SessionStatus.RUNNING)]
    assert append_calls[0]["action"] == ControlAction.ENDED
    assert append_calls[0]["handoff_mode"] == "continue"
    assert append_calls[0]["takeover_id"] == "tk_continue_1"
    assert release_calls == [
        {"takeover_id": "tk_continue_1", "operator_user_id": "u1"}
    ]


async def test_start_takeover_when_already_takeover_returns_latest_takeover_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.BROWSER,
                request_status="started",
                takeover_id="tk_existing",
            )
        ],
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)

    result = await service.start_takeover("s1", "u1", scope="shell")

    assert result["status"] == SessionStatus.TAKEOVER
    assert result["request_status"] == "started"
    assert result["scope"] == "browser"
    assert result["takeover_id"] == "tk_existing"
    assert uow.session.update_status_calls == []


async def test_end_takeover_continue_resume_failed_rolls_back_to_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)
    append_control_calls: list[dict] = []
    append_error_calls: list[dict] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_resume_task_with_handoff(*args, **kwargs):
        raise RuntimeError("sandbox unavailable")

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_control_calls.append(kwargs)
        return None

    async def fake_append_error_event(_session_id: str, **kwargs):
        append_error_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", fake_resume_task_with_handoff)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_append_error_event", fake_append_error_event)

    result = await service.end_takeover("s1", "u1", handoff_mode="continue")

    assert result == {"status": SessionStatus.COMPLETED, "handoff_mode": "complete"}
    assert uow.session.update_status_calls == [("s1", SessionStatus.COMPLETED)]
    assert append_error_calls
    assert "恢复执行失败" in append_error_calls[0]["error"]
    assert append_control_calls[0]["action"] == ControlAction.ENDED
    assert append_control_calls[0]["source"] == ControlSource.SYSTEM
    assert append_control_calls[0]["handoff_mode"] == "complete"
    assert append_control_calls[0]["reason"] == "resume_failed"


async def test_complete_takeover_after_cancel_timeout_emits_rejected_and_releases_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _ResumableTask()
    append_calls: list[dict] = []
    release_calls: list[dict] = []

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    async def fake_release_takeover_lease(_session_id: str, **kwargs):
        release_calls.append(kwargs)
        return None

    monotonic_values = [100.0, 200.0, 200.0]

    def _fake_monotonic() -> float:
        if monotonic_values:
            return monotonic_values.pop(0)
        return 200.0

    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_release_takeover_lease", fake_release_takeover_lease)
    monkeypatch.setattr(
        "app.application.services.agent_service.time.monotonic",
        _fake_monotonic,
    )

    await service._complete_takeover_after_cancel(
        session_id="s1",
        task=task,
        scope=ControlScope.SHELL,
        takeover_id="tk_001",
        operator_user_id="u1",
        cancel_timeout_seconds=1,
    )

    assert append_calls
    assert append_calls[0]["action"] == ControlAction.REJECTED
    assert append_calls[0]["request_status"] == "rejected"
    assert append_calls[0]["reason"] == "cancel_timeout"
    assert append_calls[0]["scope"] == ControlScope.SHELL
    assert release_calls == [
        {"takeover_id": "tk_001", "operator_user_id": "u1"}
    ]


async def test_renew_takeover_success_emits_control_renewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)
    append_calls: list[dict] = []
    timeout_calls: list[dict] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_renew_takeover_lease(*args, **kwargs) -> bool:
        return True

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    def fake_schedule_takeover_timeout(
        *,
        session_id: str,
        takeover_id: str,
        operator_user_id: str,
        ttl_seconds: int,
    ) -> None:
        timeout_calls.append(
            {
                "session_id": session_id,
                "takeover_id": takeover_id,
                "operator_user_id": operator_user_id,
                "ttl_seconds": ttl_seconds,
            }
        )

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_renew_takeover_lease", fake_renew_takeover_lease)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_schedule_takeover_timeout", fake_schedule_takeover_timeout)

    result = await service.renew_takeover("s1", "u1", takeover_id="tk_renew_1")

    assert result == {
        "status": SessionStatus.TAKEOVER,
        "request_status": "renewed",
        "takeover_id": "tk_renew_1",
        "expires_at": result["expires_at"],
    }
    assert isinstance(result["expires_at"], int)
    assert append_calls
    assert append_calls[0]["action"] == ControlAction.RENEWED
    assert append_calls[0]["request_status"] == "renewed"
    assert append_calls[0]["takeover_id"] == "tk_renew_1"
    assert isinstance(append_calls[0]["expires_at"], datetime)
    assert timeout_calls
    assert timeout_calls[0]["session_id"] == "s1"
    assert timeout_calls[0]["takeover_id"] == "tk_renew_1"


async def test_renew_takeover_uses_settings_ttl_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)
    renew_calls: list[int] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_renew_takeover_lease(*args, **kwargs) -> bool:
        renew_calls.append(kwargs["ttl_seconds"])
        return True

    async def fake_append_control_event(_session_id: str, **kwargs):
        return None

    monkeypatch.setattr(service._settings, "feature_takeover_lease_ttl_seconds", 120, raising=False)
    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_renew_takeover_lease", fake_renew_takeover_lease)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)

    await service.renew_takeover("s1", "u1", takeover_id="tk_renew_2")

    assert renew_calls == [120]


async def test_renew_takeover_concurrent_competition_returns_one_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)
    append_calls: list[dict] = []
    renew_call_count = 0
    lock = asyncio.Lock()

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_renew_takeover_lease(*args, **kwargs) -> bool:
        nonlocal renew_call_count
        await asyncio.sleep(0)
        async with lock:
            renew_call_count += 1
            return renew_call_count == 1

    async def fake_append_control_event(_session_id: str, **kwargs):
        append_calls.append(kwargs)
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_renew_takeover_lease", fake_renew_takeover_lease)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)

    async def _renew_once():
        try:
            return await service.renew_takeover("s1", "u1", takeover_id="tk_race_1")
        except Exception as exc:  # noqa: BLE001 - 测试需要捕获并发分支异常
            return exc

    results = await asyncio.gather(_renew_once(), _renew_once())
    conflict_results = [item for item in results if isinstance(item, ConflictError)]
    success_results = [
        item
        for item in results
        if isinstance(item, dict) and item.get("request_status") == "renewed"
    ]

    assert len(success_results) == 1
    assert len(conflict_results) == 1
    assert len(append_calls) == 1


async def test_append_control_event_writes_output_stream_and_persists() -> None:
    uow = _Uow()
    service = _make_service(uow)
    task = _ResumableTask()

    control_event = await service._append_control_event(
        "s1",
        action=ControlAction.STARTED,
        source=ControlSource.SYSTEM,
        scope=ControlScope.SHELL,
        request_status="started",
        takeover_id="tk_001",
        task=task,
    )

    assert control_event.id == "evt-control-1"
    assert len(task.output_stream.put_payloads) == 1
    assert uow.session.add_event_calls
    assert uow.session.add_event_calls[0][0] == "s1"
    assert uow.session.add_event_calls[0][1].id == "evt-control-1"


async def test_append_control_event_uses_isolated_uow_instance() -> None:
    created_uows: list[_Uow] = []

    def _uow_factory() -> _Uow:
        uow = _Uow()
        created_uows.append(uow)
        return uow

    service = AgentService(
        uow_factory=_uow_factory,
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

    control_event = await service._append_control_event(
        "s1",
        action=ControlAction.RENEWED,
        source=ControlSource.USER,
        request_status="renewed",
        takeover_id="tk_isolated_1",
    )

    assert isinstance(control_event.id, str)
    assert len(created_uows) >= 2
    assert created_uows[0].session.add_event_calls == []
    assert created_uows[1].session.add_event_calls
    assert created_uows[1].session.add_event_calls[0][0] == "s1"


async def test_start_takeover_forbidden_when_feature_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    monkeypatch.setattr(service._settings, "feature_takeover_enabled", False, raising=False)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id="u1", status=SessionStatus.WAITING)

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)

    with pytest.raises(ForbiddenError):
        await service.start_takeover("s1", "u1", scope="shell")


async def test_pending_timeout_expires_takeover_pending_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER_PENDING,
        events=[
            ControlEvent(
                action=ControlAction.REQUESTED,
                source=ControlSource.AGENT,
                scope=ControlScope.SHELL,
                takeover_id="tk_pending_1",
            )
        ],
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    release_calls: list[str] = []

    async def fake_sleep(_: float) -> None:
        return None

    async def fake_force_release_takeover_lease(session_id: str):
        release_calls.append(session_id)
        return None

    monkeypatch.setattr("app.application.services.agent_service.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release_takeover_lease)

    await service._handle_takeover_pending_timeout(session_id="s1", ttl_seconds=1)

    assert uow.session.add_event_calls
    timeout_event = uow.session.add_event_calls[0][1]
    assert timeout_event.action == ControlAction.EXPIRED
    assert timeout_event.reason == "pending_timeout"
    assert timeout_event.request_status == "expired"
    assert timeout_event.takeover_id == "tk_pending_1"
    assert uow.session.update_status_calls == [("s1", SessionStatus.COMPLETED)]
    assert uow.session.get_by_id_for_update_calls == ["s1"]
    assert release_calls == ["s1"]


async def test_takeover_timeout_moves_takeover_to_pending_and_schedules_pending_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.SHELL,
                takeover_id="tk_takeover_1",
            )
        ],
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    force_release_calls: list[str] = []
    pending_timeout_calls: list[str] = []

    async def fake_sleep(_: float) -> None:
        return None

    async def fake_force_release_takeover_lease(session_id: str):
        force_release_calls.append(session_id)
        return None

    async def fake_verify_takeover_lease_owner(**_kwargs) -> bool:
        return False

    def fake_schedule_pending_timeout(session_id: str) -> None:
        pending_timeout_calls.append(session_id)

    monkeypatch.setattr("app.application.services.agent_service.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release_takeover_lease)
    monkeypatch.setattr(service, "_verify_takeover_lease_owner", fake_verify_takeover_lease_owner)
    monkeypatch.setattr(service, "_schedule_pending_timeout", fake_schedule_pending_timeout)

    await service._handle_takeover_lease_timeout(
        session_id="s1",
        takeover_id="tk_takeover_1",
        operator_user_id="u1",
        ttl_seconds=1,
    )

    assert uow.session.add_event_calls
    timeout_event = uow.session.add_event_calls[0][1]
    assert timeout_event.action == ControlAction.EXPIRED
    assert timeout_event.reason == "takeover_timeout"
    assert timeout_event.request_status == "expired"
    assert timeout_event.takeover_id == "tk_takeover_1"
    assert uow.session.update_status_calls == [("s1", SessionStatus.TAKEOVER_PENDING)]
    assert pending_timeout_calls == ["s1"]
    assert force_release_calls == ["s1"]
    assert uow.session.get_by_id_for_update_calls == ["s1"]


async def test_start_takeover_lease_conflict_raises_conflict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id="u1", status=SessionStatus.WAITING)

    async def fake_acquire_takeover_lease(*args, **kwargs) -> bool:
        return False

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_acquire_takeover_lease", fake_acquire_takeover_lease)

    with pytest.raises(ConflictError):
        await service.start_takeover("s1", "u1", scope="shell")


async def test_renew_takeover_lease_conflict_raises_conflict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)

    async def fake_renew_takeover_lease(*args, **kwargs) -> bool:
        return False

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_renew_takeover_lease", fake_renew_takeover_lease)

    with pytest.raises(ConflictError):
        await service.renew_takeover("s1", "u1", takeover_id="tk_conflict")


async def test_assert_takeover_shell_access_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.SHELL,
                takeover_id="tk_shell_1",
            )
        ],
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_verify_takeover_lease_owner(**kwargs) -> bool:
        return kwargs["takeover_id"] == "tk_shell_1"

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_verify_takeover_lease_owner", fake_verify_takeover_lease_owner)

    await service.assert_takeover_shell_access(
        session_id="s1",
        user_id="u1",
        takeover_id="tk_shell_1",
        is_admin=False,
        user_role="user",
    )


async def test_assert_takeover_shell_access_raises_conflict_when_takeover_id_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.SHELL,
                takeover_id="tk_shell_1",
            )
        ],
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_verify_takeover_lease_owner(**kwargs) -> bool:
        return True

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_verify_takeover_lease_owner", fake_verify_takeover_lease_owner)

    with pytest.raises(ConflictError):
        await service.assert_takeover_shell_access(
            session_id="s1",
            user_id="u1",
            takeover_id="tk_shell_2",
            is_admin=False,
            user_role="user",
        )


async def test_assert_takeover_shell_access_raises_bad_request_when_scope_not_shell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER,
        events=[
            ControlEvent(
                action=ControlAction.STARTED,
                source=ControlSource.USER,
                scope=ControlScope.BROWSER,
                takeover_id="tk_browser_1",
            )
        ],
    )

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)

    with pytest.raises(BadRequestError):
        await service.assert_takeover_shell_access(
            session_id="s1",
            user_id="u1",
            takeover_id="tk_browser_1",
            is_admin=False,
            user_role="user",
        )


async def test_start_takeover_forbidden_when_single_worker_only_and_multi_worker_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    monkeypatch.setattr(
        service._settings,
        "feature_takeover_single_worker_only",
        True,
        raising=False,
    )
    monkeypatch.setenv("WEB_CONCURRENCY", "2")

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id="u1", status=SessionStatus.WAITING)

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)

    with pytest.raises(ForbiddenError):
        await service.start_takeover("s1", "u1", scope="shell")


async def test_shutdown_cancels_background_tasks() -> None:
    class _DummyTaskCls:
        destroyed = False

        @classmethod
        async def destroy(cls) -> None:
            cls.destroyed = True

    service = AgentService(
        uow_factory=lambda: _Uow(),
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(),
        a2a_config=A2AConfig(),
        sandbox_cls=object,
        task_cls=_DummyTaskCls,
        json_parser=object(),
        search_engine=object(),
        file_storage=object(),
    )

    pending_task = asyncio.create_task(asyncio.sleep(60))
    background_task = asyncio.create_task(asyncio.sleep(60))
    service._pending_timeout_tasks["s1"] = pending_task
    service._background_tasks.add(background_task)

    await service.shutdown()
    await asyncio.sleep(0)

    assert pending_task.cancelled() is True
    assert background_task.cancelled() is True
    assert _DummyTaskCls.destroyed is True


async def test_update_status_completed_sets_completed_at(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """当 reject_takeover(terminate) 触发 update_status(COMPLETED) 时，
    mock 的 _SessionRepo 应模拟设置 completed_at。"""
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.TAKEOVER_PENDING,
        events=[
            ControlEvent(
                action=ControlAction.REQUESTED,
                source=ControlSource.AGENT,
                scope=ControlScope.SHELL,
                takeover_id="tk_completed_at_test",
            )
        ],
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    async def fake_append_control_event(_session_id: str, **kwargs):
        return None

    async def fake_force_release_takeover_lease(session_id: str):
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_append_control_event", fake_append_control_event)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release_takeover_lease)

    assert session.completed_at is None
    await service.reject_takeover("s1", "u1", decision="terminate")
    assert session.completed_at is not None


async def test_reopen_takeover_success_schedules_pending_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reopen_takeover 成功时应更新状态到 takeover_pending 并调度 pending timeout。"""
    from datetime import timedelta

    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=60),
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    schedule_calls: list[str] = []

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    def fake_schedule_pending_timeout(session_id: str) -> None:
        schedule_calls.append(session_id)

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_schedule_pending_timeout", fake_schedule_pending_timeout)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 300, raising=False
    )

    result = await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")

    assert result["status"] == SessionStatus.TAKEOVER_PENDING
    assert result["request_status"] == "reopened"
    assert result["remaining_seconds"] > 0
    assert schedule_calls == ["s1"]
    assert uow.session.update_status_calls == [("s1", SessionStatus.TAKEOVER_PENDING)]
    # reopen 后 completed_at 应被清空，避免统计误判
    assert session.completed_at is None
    assert uow.session.add_event_calls
    event = uow.session.add_event_calls[0][1]
    assert event.action == ControlAction.REOPENED
    assert event.source == ControlSource.USER


async def test_reopen_takeover_expired_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """超过 reopen 窗口应抛出 REOPEN_WINDOW_EXPIRED。"""
    from datetime import timedelta

    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=600),
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 300, raising=False
    )

    with pytest.raises(BadRequestError, match="REOPEN_WINDOW_EXPIRED"):
        await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")


async def test_reopen_takeover_disabled_when_window_non_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """window_seconds <= 0 时应抛出 REOPEN_DISABLED。"""
    uow = _Uow()
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return Session(id="s1", user_id="u1", status=SessionStatus.COMPLETED)

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 0, raising=False
    )

    with pytest.raises(BadRequestError, match="REOPEN_DISABLED"):
        await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")


async def test_reopen_takeover_boundary_elapsed_equals_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """elapsed == window_seconds 时恰好在边界内，应成功。"""
    from datetime import timedelta

    window = 300
    # 使用略小于 window 的值避免测试中微小的时间漂移
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=window - 1),
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    def fake_schedule_pending_timeout(session_id: str) -> None:
        pass

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_schedule_pending_timeout", fake_schedule_pending_timeout)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", window, raising=False
    )

    result = await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")
    assert result["status"] == SessionStatus.TAKEOVER_PENDING


async def test_reopen_takeover_admin_can_reopen_others_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """admin 用户可以 reopen 他人会话。"""
    from datetime import timedelta

    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=60),
    )
    uow = _Uow(session=session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    def fake_schedule_pending_timeout(session_id: str) -> None:
        pass

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_schedule_pending_timeout", fake_schedule_pending_timeout)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 300, raising=False
    )

    result = await service.reopen_takeover(
        "s1", "admin_user", is_admin=True, user_role="super_admin"
    )
    assert result["status"] == SessionStatus.TAKEOVER_PENDING


async def test_reopen_takeover_concurrent_requests_only_one_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """并发两次 reopen，仅一次成功，另一次因状态非 completed 而失败，且不重复写入事件。"""
    from datetime import timedelta

    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=60),
    )
    uow = _Uow(session=session)
    service = _make_service(uow)
    call_count = {"reopen": 0}

    original_get_for_update = uow.session.get_by_id_for_update

    async def mock_get_for_update(session_id):
        call_count["reopen"] += 1
        s = await original_get_for_update(session_id)
        if call_count["reopen"] > 1:
            # 模拟第二次读到的状态已被第一次修改为 takeover_pending
            s.status = SessionStatus.TAKEOVER_PENDING
        return s

    async def fake_get_accessible_session(*args, **kwargs) -> Session:
        return session

    def fake_schedule_pending_timeout(session_id: str) -> None:
        pass

    monkeypatch.setattr(uow.session, "get_by_id_for_update", mock_get_for_update)
    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_schedule_pending_timeout", fake_schedule_pending_timeout)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 300, raising=False
    )

    # 第一次调用成功
    result = await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")
    assert result["status"] == SessionStatus.TAKEOVER_PENDING

    # 第二次调用失败（状态已非 completed）
    with pytest.raises(BadRequestError, match="当前状态不支持恢复接管"):
        await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")

    # 验证事件只写入了一次
    reopened_events = [
        e for _, e in uow.session.add_event_calls
        if getattr(e, "action", None) == ControlAction.REOPENED
    ]
    assert len(reopened_events) == 1
