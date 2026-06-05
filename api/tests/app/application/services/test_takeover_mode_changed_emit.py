"""A4-0 T-EMIT-HTTP (Pattern A): the 4 txn-after takeover sites emit a
SessionModeChangedEvent with the right to/reason/from_mode and the in-txn
mode_revision; live sites emit BEFORE the ControlEvent."""
from datetime import datetime, timedelta

import pytest

from app.application.services.agent_service import AgentService
from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    SessionModeChangedEvent,
)
from app.domain.models.session import Session, SessionStatus
from tests.app.application.services.conftest import default_snapshot as _default_snapshot

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, session: Session | None = None, *, rev: int = 10) -> None:
        self.update_status_calls: list = []
        self.add_event_calls: list = []
        self._session = session
        self._rev = rev

    async def update_status(self, session_id: str, status) -> None:
        self.update_status_calls.append((session_id, status))
        self._rev += 1

    async def read_status_with_revision(self, session_id: str):
        return SessionStatus.RUNNING, self._rev

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))

    async def get_by_id(self, session_id: str):
        return self._session

    async def get_by_id_for_update(self, session_id: str):
        return self._session


class _Uow:
    def __init__(self, session: Session | None = None, *, rev: int = 10) -> None:
        self.session = _SessionRepo(session, rev=rev)

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def commit(self) -> None:  # parity with the proven takeover-test _Uow
        return None

    async def rollback(self) -> None:
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


def _mode_events(uow: _Uow) -> list[SessionModeChangedEvent]:
    return [
        e for (_sid, e) in uow.session.add_event_calls
        if isinstance(e, SessionModeChangedEvent)
    ]


async def test_start_takeover_db_only_emits_takeover_mode_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER_PENDING)
    uow = _Uow(session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*a, **k) -> Session:
        return session

    async def fake_append(_sid, **k):
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_append_control_event", fake_append)
    monkeypatch.setattr(service, "_schedule_takeover_timeout", lambda **k: None)

    await service.start_takeover("s1", "u1", scope="shell")

    events = _mode_events(uow)
    assert len(events) == 1
    assert events[0].to == "takeover"
    assert events[0].reason == "takeover_started"
    assert events[0].from_mode == "takeover_pending"  # source status
    assert events[0].mode_revision == 11  # the bumped revision


async def test_end_takeover_emits_running_mode_changed_before_control_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER)
    uow = _Uow(session)
    service = _make_service(uow)
    order: list[str] = []

    async def fake_get_accessible_session(*a, **k) -> Session:
        return session

    class _Task:
        class _OS:
            async def put(self, payload):
                order.append("mode_changed")
                return "e1"
        def __init__(self):
            self.output_stream = _Task._OS()

    async def fake_resume(*a, **k):
        return _Task()

    async def fake_append(_sid, **k):
        order.append("control")
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", fake_resume)
    monkeypatch.setattr(service, "_append_control_event", fake_append)

    await service.end_takeover("s1", "u1", handoff_mode="continue")

    events = _mode_events(uow)
    assert any(e.to == "running" and e.reason == "takeover_ended" for e in events)
    # INV-1: the mode-changed put precedes the ControlEvent on the live path.
    assert order.index("mode_changed") < order.index("control")


async def test_reject_takeover_continue_emits_running_mode_changed_before_control_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # reject_takeover requires the TAKEOVER_PENDING precondition (guard ~4447).
    session = Session(id="s1", user_id="u1", status=SessionStatus.TAKEOVER_PENDING)
    uow = _Uow(session)
    service = _make_service(uow)
    order: list[str] = []

    async def fake_get_accessible_session(*a, **k) -> Session:
        return session

    class _Task:
        class _OS:
            async def put(self, payload):
                order.append("mode_changed")
                return "e1"
        def __init__(self):
            self.output_stream = _Task._OS()

    async def fake_resume(*a, **k):
        return _Task()

    async def fake_append(_sid, **k):
        order.append("control")
        return None

    async def fake_force_release(_sid, **k):
        return None

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_resume_task_with_handoff", fake_resume)
    monkeypatch.setattr(service, "_append_control_event", fake_append)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release)

    await service.reject_takeover("s1", "u1", decision="continue")

    events = _mode_events(uow)
    assert any(
        e.to == "running" and e.reason == "takeover_rejected" for e in events
    )
    # from_mode is fixed to the precondition mode (TAKEOVER_PENDING).
    assert events[0].from_mode == "takeover_pending"
    assert events[0].mode_revision == 11  # the bumped revision
    # INV-1: the mode-changed put precedes the ControlEvent on the live path.
    assert order.index("mode_changed") < order.index("control")


async def test_complete_takeover_after_cancel_done_emits_takeover_mode_changed_before_control_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uow = _Uow()
    service = _make_service(uow)
    order: list[str] = []

    class _Task:
        class _OS:
            async def put(self, payload):
                order.append("mode_changed")
                return "e1"

        def __init__(self):
            self.done = True
            self.output_stream = _Task._OS()

    task = _Task()

    async def fake_append(_sid, **k):
        order.append("control")
        return None

    monkeypatch.setattr(service, "_append_control_event", fake_append)
    monkeypatch.setattr(service, "_schedule_takeover_timeout", lambda **k: None)

    await service._complete_takeover_after_cancel(
        session_id="s1",
        task=task,
        scope=ControlScope.SHELL,
        takeover_id="tk_001",
        operator_user_id="u1",
        cancel_timeout_seconds=1,
    )

    events = _mode_events(uow)
    assert any(
        e.to == "takeover" and e.reason == "takeover_started" for e in events
    )
    # from_mode is fixed to "running" (this path only fires after a RUNNING cancel).
    assert events[0].from_mode == "running"
    assert events[0].mode_revision == 11  # the bumped revision
    # INV-1: the mode-changed put precedes the ControlEvent on the live path.
    assert order.index("mode_changed") < order.index("control")


async def test_reopen_takeover_inline_emits_takeover_pending_mode_changed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = Session(
        id="s1",
        user_id="u1",
        status=SessionStatus.COMPLETED,
        completed_at=datetime.now() - timedelta(seconds=60),  # inside reopen window
    )
    uow = _Uow(session)
    service = _make_service(uow)

    async def fake_get_accessible_session(*a, **k) -> Session:
        return session

    monkeypatch.setattr(service, "_get_accessible_session", fake_get_accessible_session)
    monkeypatch.setattr(service, "_schedule_pending_timeout", lambda session_id: None)
    monkeypatch.setattr(
        service._settings, "feature_takeover_reopen_window_seconds", 300, raising=False
    )

    await service.reopen_takeover("s1", "u1", is_admin=False, user_role="user")

    events = _mode_events(uow)
    assert len(events) == 1
    assert events[0].to == "takeover_pending"
    assert events[0].reason == "takeover_reopened"
    assert events[0].from_mode is None  # source terminal (COMPLETED) → not a ModeLiteral


async def test_lease_timeout_inline_emits_takeover_pending_mode_changed(
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
    uow = _Uow(session)
    service = _make_service(uow)

    async def fake_sleep(_: float) -> None:
        return None

    async def fake_force_release(session_id: str):
        return None

    async def fake_verify_owner(**_kwargs) -> bool:
        return False

    monkeypatch.setattr("app.application.services.agent_service.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(service, "_force_release_takeover_lease", fake_force_release)
    monkeypatch.setattr(service, "_verify_takeover_lease_owner", fake_verify_owner)
    monkeypatch.setattr(service, "_schedule_pending_timeout", lambda session_id: None)

    await service._handle_takeover_lease_timeout(
        session_id="s1",
        takeover_id="tk_takeover_1",
        operator_user_id="u1",
        ttl_seconds=1,
    )

    events = _mode_events(uow)
    assert any(
        e.to == "takeover_pending"
        and e.reason == "takeover_lease_timeout"
        and e.from_mode == "takeover"
        for e in events
    )
