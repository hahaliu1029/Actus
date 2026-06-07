"""A4-0 T-ENUM: _emit_flow_event returns a typed FlowYieldSignal (not strings),
and the magic-string returns are gone from the source."""
import inspect

import pytest

from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ToolConfirmationEvent,
    WaitEvent,
)
from app.domain.models.session import SessionStatus
from app.domain.services.agent_task_runner import AgentTaskRunner, FlowYieldSignal
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SessionRepo:
    def __init__(self) -> None:
        self.status_updates: list = []
        self.add_event_calls: list = []
        self._rev = 0

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))
        self._rev += 1

    async def read_status_with_revision(self, session_id: str):
        return SessionStatus.WAITING, self._rev

    async def add_event(self, session_id: str, event) -> None:
        self.add_event_calls.append((session_id, event))


class _Uow:
    def __init__(self) -> None:
        self.session = _SessionRepo()

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


def _make_runner() -> AgentTaskRunner:
    runner = object.__new__(AgentTaskRunner)
    runner._session_id = "s1"
    runner._uow = _Uow()
    runner._session_state_machine = DefaultSessionStateMachine(uow_factory=lambda: None)
    return runner


async def test_emit_flow_event_returns_wait_signal_for_wait_event() -> None:
    runner = _make_runner()
    result = await runner._emit_flow_event(_Task(), WaitEvent())
    assert result is FlowYieldSignal.WAIT


async def test_emit_flow_event_returns_wait_signal_for_tool_confirmation() -> None:
    runner = _make_runner()
    event = ToolConfirmationEvent(
        tool_call_id="c1",
        tool_name="shell_execute",
        tool_args={},
        risk_level="high",
        risk_reason="r",
        matched_patterns=[],
        timeout_seconds=60,
    )
    result = await runner._emit_flow_event(_Task(), event)
    assert result is FlowYieldSignal.WAIT


async def test_emit_flow_event_returns_takeover_signal_for_control_requested() -> None:
    runner = _make_runner()
    event = ControlEvent(action=ControlAction.REQUESTED, scope=ControlScope.SHELL)
    result = await runner._emit_flow_event(_Task(), event)
    assert result is FlowYieldSignal.TAKEOVER_REQUESTED


async def test_emit_flow_event_returns_none_for_other_events() -> None:
    runner = _make_runner()
    # A non-trigger event: ControlEvent(STARTED) is NOT a trigger (only REQUESTED
    # is), needs no scope (the validator requires scope only for REQUESTED), and
    # hits no Title/Message/Health side-effect branch — it is just put + returns None.
    result = await runner._emit_flow_event(
        _Task(), ControlEvent(action=ControlAction.STARTED)
    )
    assert result is None


def test_flow_yield_signal_is_not_a_str_enum() -> None:
    # D4: a plain Enum so a stray `== "wait"` no longer matches (full retirement).
    assert FlowYieldSignal.WAIT != "wait"
    assert FlowYieldSignal.TAKEOVER_REQUESTED != "takeover"


def test_emit_flow_event_source_has_no_magic_string_returns() -> None:
    src = inspect.getsource(AgentTaskRunner._emit_flow_event)
    assert 'return "wait"' not in src
    assert 'return "takeover"' not in src
