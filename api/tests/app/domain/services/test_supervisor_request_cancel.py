from __future__ import annotations

import pytest

from app.domain.services.execution_supervisor import ExecutionSupervisor

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def hset(
        self,
        key: str,
        field: str | None = None,
        value: str | None = None,
        *,
        mapping: dict[str, str] | None = None,
    ) -> int:
        bucket = self.hashes.setdefault(key, {})
        if mapping is not None:
            bucket.update({k: str(v) for k, v in mapping.items()})
            return len(mapping)
        if field is None:
            raise AssertionError("field is required without mapping")
        bucket[field] = "" if value is None else str(value)
        return 1

    async def expire(self, key: str, seconds: int) -> bool:
        self.expire_calls.append((key, seconds))
        return True


def _build_supervisor(redis: _FakeRedis) -> ExecutionSupervisor:
    return ExecutionSupervisor(redis_client=redis, session_repository=object())


async def test_request_cancel_stamps_default_user_cancel_reason() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    await supervisor.request_cancel(session_id="session-1", user_id="user-1")

    assert redis.hashes["supervisor:hot:session-1"] == {
        "cancellation_pending": "1",
        "pending_terminal_reason": "user_cancel",
    }
    assert redis.expire_calls == [("supervisor:hot:session-1", 300)]


async def test_request_cancel_is_idempotent_and_refreshes_hot_hash_ttl() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    await supervisor.request_cancel(session_id="session-1", user_id="user-1")
    await supervisor.request_cancel(session_id="session-1", user_id="user-1")

    assert redis.hashes["supervisor:hot:session-1"] == {
        "cancellation_pending": "1",
        "pending_terminal_reason": "user_cancel",
    }
    assert redis.expire_calls == [
        ("supervisor:hot:session-1", 300),
        ("supervisor:hot:session-1", 300),
    ]


async def test_request_cancel_invokes_optional_stop_session_after_stamp() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)
    calls: list[dict[str, str]] = []

    async def stop_session(*, session_id: str, user_id: str) -> None:
        calls.append({"session_id": session_id, "user_id": user_id})

    await supervisor.request_cancel(
        session_id="session-1",
        user_id="user-1",
        stop_session=stop_session,
    )

    assert redis.hashes["supervisor:hot:session-1"] == {
        "cancellation_pending": "1",
        "pending_terminal_reason": "user_cancel",
    }
    assert calls == [{"session_id": "session-1", "user_id": "user-1"}]


async def test_request_cancel_propagates_callback_error_after_stamp() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    async def stop_session(*, session_id: str, user_id: str) -> None:
        raise RuntimeError("stop failed")

    with pytest.raises(RuntimeError, match="stop failed"):
        await supervisor.request_cancel(
            session_id="session-1",
            user_id="user-1",
            stop_session=stop_session,
        )

    assert redis.hashes["supervisor:hot:session-1"] == {
        "cancellation_pending": "1",
        "pending_terminal_reason": "user_cancel",
    }
    assert redis.expire_calls == [("supervisor:hot:session-1", 300)]
