from __future__ import annotations

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)


def test_require_state_machine_raises_when_none() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._session_state_machine = None
    with pytest.raises(RuntimeError, match="SessionStateMachine"):
        runner._require_state_machine()


def test_require_state_machine_returns_injected() -> None:
    runner = object.__new__(AgentTaskRunner)
    ssm = DefaultSessionStateMachine(uow_factory=lambda: None)
    runner._session_state_machine = ssm
    assert runner._require_state_machine() is ssm
