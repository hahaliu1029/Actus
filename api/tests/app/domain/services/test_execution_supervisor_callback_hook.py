from __future__ import annotations

import pytest

from app.domain.services.execution_supervisor import ExecutionSupervisor

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _build_supervisor() -> ExecutionSupervisor:
    return ExecutionSupervisor(redis_client=object(), session_repository=object())


class _Repo:
    def __init__(self) -> None:
        self.updates: list[tuple[str, dict[str, str]]] = []

    async def update_supervisor_fields(self, session_id: str, **fields) -> None:
        self.updates.append((session_id, fields))


class _CancelableTask:
    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_calls.append(reason)
        return True


async def test_suspend_idle_cancels_live_task_with_supervisor_suspend(
    monkeypatch,
) -> None:
    repo = _Repo()
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    task = _CancelableTask()
    supervisor._register_runner("session-3", task)
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.suspend_idle(session_id="session-3", user_id="user-1")

    assert repo.updates == [
        (
            "session-3",
            {
                "execution_phase": "suspended",
                "suspended_reason": "bg_idle_timeout",
            },
        )
    ]
    assert task.cancel_calls == ["supervisor_suspend"]
    assert revoke_calls == []
    assert "session-3" not in supervisor._runners


async def test_runner_session_complete_skips_revoke_for_supervisor_suspend(
    monkeypatch,
) -> None:
    supervisor = _build_supervisor()
    runner = object()
    supervisor._register_runner("session-1", runner)
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor._on_runner_session_complete(
        session_id="session-1",
        user_id="user-1",
        cancel_reason="supervisor_suspend",
    )

    assert revoke_calls == []
    assert "session-1" not in supervisor._runners


async def test_runner_session_complete_revokes_for_terminal_reason(
    monkeypatch,
) -> None:
    supervisor = _build_supervisor()
    supervisor._register_runner("session-2", object())
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor._on_runner_session_complete(
        session_id="session-2",
        user_id="user-1",
        cancel_reason="user_cancel",
    )

    assert revoke_calls == [
        {"session_id": "session-2", "user_id": "user-1", "reason": "user_cancel"}
    ]
    assert "session-2" not in supervisor._runners
