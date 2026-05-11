from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.domain.models.session import Session, SessionStatus
from app.domain.services.execution_supervisor import ExecutionSupervisor

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _build_supervisor() -> ExecutionSupervisor:
    return ExecutionSupervisor(redis_client=object(), session_repository=object())


class _RecordingRedis:
    def __init__(self, *, hincrby_value: int = 1) -> None:
        self.hincrby_value = hincrby_value
        self.hincrby_calls: list[tuple[str, str, int]] = []
        self.expire_calls: list[tuple[str, int]] = []
        self.hset_calls: list[tuple[str, str, int]] = []

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        self.hincrby_calls.append((key, field, amount))
        return self.hincrby_value

    async def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))

    async def hset(self, key: str, field: str, value: int) -> None:
        self.hset_calls.append((key, field, value))


class _BrokenRedis:
    async def hincrby(self, key: str, field: str, amount: int) -> int:
        raise RuntimeError("redis bounced")


class _Repo:
    def __init__(self, session: Session | None = None) -> None:
        self.updates: list[tuple[str, dict[str, str]]] = []
        self.terminal_updates: list[tuple[str, SessionStatus, str]] = []
        self._session = session

    async def update_supervisor_fields(self, session_id: str, **fields) -> None:
        self.updates.append((session_id, fields))
        if self._session and self._session.id == session_id:
            for key, value in fields.items():
                setattr(self._session, key, value)

    async def get_by_id(self, session_id: str) -> Session | None:
        if self._session is None or self._session.id != session_id:
            return None
        return self._session

    async def update_to_terminal(
        self,
        session_id: str,
        status: SessionStatus,
        terminal_reason: str,
    ) -> None:
        self.terminal_updates.append((session_id, status, terminal_reason))
        if self._session and self._session.id == session_id:
            self._session.status = status
            self._session.terminal_reason = terminal_reason


class _CancelableTask:
    def __init__(self) -> None:
        self.cancel_calls: list[str] = []

    def cancel(self, reason: str = "stop") -> bool:
        self.cancel_calls.append(reason)
        return True


async def test_inflight_inc_swallow_redis_error() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_BrokenRedis(), session_repository=object()
    )

    value = await supervisor.inflight_inc(session_id="session-1", kind="llm")

    assert value == 0


async def test_inflight_dec_swallow_redis_error() -> None:
    supervisor = ExecutionSupervisor(
        redis_client=_BrokenRedis(), session_repository=object()
    )

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == 0


async def test_inflight_dec_does_not_refresh_hot_hash_ttl() -> None:
    redis = _RecordingRedis(hincrby_value=2)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == 2
    assert redis.hincrby_calls == [
        ("supervisor:hot:session-1", "inflight_llm_count", -1)
    ]
    assert redis.expire_calls == []


async def test_inflight_dec_preserves_negative_value_for_drift_observation() -> None:
    redis = _RecordingRedis(hincrby_value=-1)
    supervisor = ExecutionSupervisor(redis_client=redis, session_repository=object())

    value = await supervisor.inflight_dec(session_id="session-1", kind="llm")

    assert value == -1
    assert redis.hset_calls == []
    assert redis.expire_calls == []


async def test_suspend_idle_cancels_live_task_with_supervisor_suspend(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-3",
            user_id="user-1",
            was_background=True,
        )
    )
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


async def test_terminate_watchdog_timeout_emits_bg_failed_watchdog(
    monkeypatch,
) -> None:
    repo = _Repo(
        Session(
            id="session-expired",
            user_id="user-1",
            status=SessionStatus.RUNNING,
            execution_mode="background",
            execution_phase="running",
            was_background=True,
        )
    )
    supervisor = ExecutionSupervisor(redis_client=object(), session_repository=repo)
    emitter = AsyncMock()
    revoke_calls: list[dict[str, str]] = []

    async def fake_revoke(**kwargs) -> int:
        revoke_calls.append(kwargs)
        return 1

    monkeypatch.setattr(supervisor, "_lua_revoke", fake_revoke)

    await supervisor.terminate(
        session_id="session-expired",
        user_id="user-1",
        terminal_reason="watchdog_timeout",
        status=SessionStatus.TIMED_OUT,
        notification_emitter=emitter,
    )

    assert repo.terminal_updates == [
        ("session-expired", SessionStatus.TIMED_OUT, "watchdog_timeout")
    ]
    emitter.emit.assert_awaited_once_with(
        user_id="user-1",
        event_type="bg_failed_watchdog",
        payload={"session_id": "session-expired"},
    )
    assert revoke_calls == [
        {
            "session_id": "session-expired",
            "user_id": "user-1",
            "reason": "watchdog_timeout",
        }
    ]


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
