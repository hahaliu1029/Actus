from __future__ import annotations

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeRepo:
    """Minimal SessionRepository surface used by set_mode / terminate."""

    def __init__(self, terminal_result: bool = True) -> None:
        self._terminal_result = terminal_result
        self.update_status_calls: list[tuple[str, SessionStatus]] = []
        self.update_to_terminal_calls: list[tuple[str, SessionStatus, str]] = []

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        self.update_status_calls.append((session_id, status))

    async def update_to_terminal(
        self, session_id: str, status: SessionStatus, terminal_reason: str
    ) -> bool:
        self.update_to_terminal_calls.append((session_id, status, terminal_reason))
        return self._terminal_result


def _boom_factory():
    raise AssertionError("caller-owned SSM must not open its own UoW")


async def test_set_mode_delegates_to_update_status_only() -> None:
    repo = _FakeRepo()
    ssm = DefaultSessionStateMachine(uow_factory=_boom_factory)
    result = await ssm.set_mode(
        "s1", SessionStatus.RUNNING, "invoke_start", session_repo=repo
    )
    assert result is None
    assert repo.update_status_calls == [("s1", SessionStatus.RUNNING)]
    assert repo.update_to_terminal_calls == []


async def test_terminate_returns_repo_bool_true() -> None:
    repo = _FakeRepo(terminal_result=True)
    ssm = DefaultSessionStateMachine(uow_factory=_boom_factory)
    result = await ssm.terminate(
        "s1", SessionStatus.COMPLETED, "user_cancel", session_repo=repo
    )
    assert result is True
    assert repo.update_to_terminal_calls == [
        ("s1", SessionStatus.COMPLETED, "user_cancel")
    ]
    assert repo.update_status_calls == []


async def test_terminate_returns_repo_bool_false() -> None:
    repo = _FakeRepo(terminal_result=False)
    ssm = DefaultSessionStateMachine(uow_factory=_boom_factory)
    result = await ssm.terminate(
        "s1", SessionStatus.TIMED_OUT, "server_restart", session_repo=repo
    )
    assert result is False
