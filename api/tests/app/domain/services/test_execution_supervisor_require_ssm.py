from __future__ import annotations

import pytest

from app.domain.services.execution_supervisor import ExecutionSupervisor
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)


def test_supervisor_accepts_and_requires_ssm() -> None:
    ssm = DefaultSessionStateMachine(uow_factory=lambda: None)
    supervisor = ExecutionSupervisor(
        redis_client=object(),
        session_repository=object(),
        session_state_machine=ssm,
    )
    assert supervisor._require_state_machine() is ssm


def test_supervisor_require_ssm_raises_when_none() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=object(),
        session_repository=object(),
    )
    with pytest.raises(RuntimeError, match="SessionStateMachine"):
        supervisor._require_state_machine()
