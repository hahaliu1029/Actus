"""A4-0 T-EMIT-LOOP + T-REVISION (agent-loop): _emit_flow_event emits a
SessionModeChangedEvent BEFORE the trigger Wait/Control event, carrying the
in-txn-captured mode_revision."""
import json

import pytest

from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ToolConfirmationEvent,
    WaitEvent,
)
from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self, *, rev: int = 7, raise_read: bool = False) -> None:
        self.status_updates: list = []
        self.add_event_calls: list = []
        self._rev = rev
        self._raise_read = raise_read

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def read_status_with_revision(self, session_id: str):
        if self._raise_read:
            raise KeyError("boom")
        return SessionStatus(self.status_updates[-1][1]), self._rev

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _Uow:
    def __init__(self, repo: _SessionRepo) -> None:
        self.session = repo

    async def __aenter__(self) -> "_Uow":
        return self

    async def __aexit__(self, *exc) -> None:
        return None


class _OutputStream:
    def __init__(self) -> None:
        self.events: list[str] = []

    async def put(self, payload: str) -> str:
        self.events.append(payload)
        return f"evt-{len(self.events)}"


class _Task:
    def __init__(self) -> None:
        self.output_stream = _OutputStream()


def _make_runner(repo: _SessionRepo) -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "s1"
    runner._uow = _Uow(repo)
    runner._session_state_machine = DefaultSessionStateMachine(uow_factory=lambda: None)
    return runner


async def test_wait_event_emits_mode_changed_before_wait_and_sets_waiting() -> None:
    repo = _SessionRepo(rev=7)
    runner = _make_runner(repo)
    task = _Task()

    await runner._emit_flow_event(task, WaitEvent())

    # Two events on the live stream, mode-changed FIRST (INV-1).
    assert len(task.output_stream.events) == 2
    first = json.loads(task.output_stream.events[0])
    second = json.loads(task.output_stream.events[1])
    assert first["type"] == "session_mode_changed"
    assert first["to"] == "waiting"
    assert first["from_mode"] == "running"
    assert first["reason"] == "wait"
    assert first["mode_revision"] == 7  # T-REVISION: this write's revision
    assert second["type"] == "wait"
    # DB status write happened.
    assert repo.status_updates == [("s1", SessionStatus.WAITING)]


async def test_tool_confirmation_emits_mode_changed_before_trigger() -> None:
    repo = _SessionRepo(rev=3)
    runner = _make_runner(repo)
    task = _Task()
    event = ToolConfirmationEvent(
        tool_call_id="c1",
        tool_name="shell_execute",
        tool_args={},
        risk_level="high",
        risk_reason="r",
        matched_patterns=[],
        timeout_seconds=60,
    )

    await runner._emit_flow_event(task, event)

    first = json.loads(task.output_stream.events[0])
    second = json.loads(task.output_stream.events[1])
    assert first["type"] == "session_mode_changed"
    assert first["to"] == "waiting"
    assert second["type"] == "tool_confirmation"


async def test_control_requested_emits_takeover_pending_mode_changed_first() -> None:
    repo = _SessionRepo(rev=4)
    runner = _make_runner(repo)
    task = _Task()
    event = ControlEvent(action=ControlAction.REQUESTED, scope=ControlScope.SHELL)

    await runner._emit_flow_event(task, event)

    first = json.loads(task.output_stream.events[0])
    second = json.loads(task.output_stream.events[1])
    assert first["type"] == "session_mode_changed"
    assert first["to"] == "takeover_pending"
    assert first["reason"] == "takeover_requested"
    assert first["mode_revision"] == 4
    assert second["type"] == "control"
    assert repo.status_updates == [("s1", SessionStatus.TAKEOVER_PENDING)]


async def test_revision_read_failure_omits_mode_revision_and_does_not_raise() -> None:
    repo = _SessionRepo(raise_read=True)
    runner = _make_runner(repo)
    task = _Task()

    await runner._emit_flow_event(task, WaitEvent())  # must not raise

    first = json.loads(task.output_stream.events[0])
    assert first["type"] == "session_mode_changed"
    assert first["mode_revision"] is None
    # status + trigger still delivered.
    assert repo.status_updates == [("s1", SessionStatus.WAITING)]
    assert json.loads(task.output_stream.events[1])["type"] == "wait"
