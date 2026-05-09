from __future__ import annotations

import asyncio

import pytest

from app.domain.services import execution_supervisor as supervisor_module
from app.domain.services.execution_supervisor import ExecutionSupervisor

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.strings: dict[str, str] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.eval_calls: list[tuple[str, int, tuple[object, ...]]] = []

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        bucket = self.hashes.setdefault(key, {})
        value = int(bucket.get(field, "0")) + amount
        bucket[field] = str(value)
        return value

    async def expire(self, key: str, seconds: int) -> bool:
        if key not in self.hashes and key not in self.strings:
            self.expire_calls.append((key, seconds))
            return False
        self.ttls[key] = seconds
        self.expire_calls.append((key, seconds))
        return True

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        if nx and key in self.strings:
            return False
        self.strings[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        self.eval_calls.append((script, numkeys, args))
        if numkeys != 1:
            raise AssertionError(f"unexpected numkeys: {numkeys}")
        key = str(args[0])
        if "HINCRBY" in script:
            seconds = int(args[1])
            if key not in self.hashes:
                return 0
            bucket = self.hashes[key]
            value = int(bucket.get("subscriber_count", "0")) - 1
            if value < 0:
                value = 0
            bucket["subscriber_count"] = str(value)
            self.ttls[key] = seconds
            self.expire_calls.append((key, seconds))
            return value

        expected = str(args[1])
        if "EXPIRE" in script:
            seconds = int(args[2])
            if self.strings.get(key) == expected:
                self.ttls[key] = seconds
                self.expire_calls.append((key, seconds))
                return 1
            return 0
        if self.strings.get(key) == expected:
            del self.strings[key]
            self.ttls.pop(key, None)
            return 1
        return 0


def _build_supervisor(redis: _FakeRedis) -> ExecutionSupervisor:
    return ExecutionSupervisor(redis_client=redis, session_repository=object())


async def test_owner_lease_acquired_sets_ttl_and_releases_on_exit() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    async with supervisor.subscriber_scope(
        session_id="session-1",
        connection_id="tab-1",
    ) as scope:
        assert scope.is_conflict is False
        assert scope.current_owner == "tab-1"
        assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "1"
        assert redis.strings["supervisor:owner:session-1"] == "tab-1"
        assert redis.ttls["supervisor:owner:session-1"] == 10
        assert redis.ttls["supervisor:hot:session-1"] == 300

    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "0"
    assert "supervisor:owner:session-1" not in redis.strings


async def test_second_tab_conflicts_with_current_owner() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    async with supervisor.subscriber_scope(
        session_id="session-1",
        connection_id="tab-1",
    ):
        async with supervisor.subscriber_scope(
            session_id="session-1",
            connection_id="tab-2",
        ) as scope:
            assert scope.is_conflict is True
            assert scope.current_owner == "tab-1"
            assert redis.strings["supervisor:owner:session-1"] == "tab-1"

    owner_expire_calls = [
        call for call in redis.expire_calls if call[0] == "supervisor:owner:session-1"
    ]
    assert owner_expire_calls == []


async def test_nested_subscriber_scopes_increment_and_decrement_count() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    async with supervisor.subscriber_scope(
        session_id="session-1",
        connection_id="tab-1",
    ):
        assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "1"

        async with supervisor.subscriber_scope(
            session_id="session-1",
            connection_id="tab-2",
        ):
            assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "2"

        assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "1"

    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "0"


async def test_owner_release_does_not_delete_new_owner() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)

    async with supervisor.subscriber_scope(
        session_id="session-1",
        connection_id="tab-1",
    ):
        redis.strings["supervisor:owner:session-1"] = "tab-2"

    assert redis.strings["supervisor:owner:session-1"] == "tab-2"


class _CancelOnFirstHotExpireRedis(_FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.hot_expire_cancelled = False

    async def expire(self, key: str, seconds: int) -> bool:
        if key == "supervisor:hot:session-1" and not self.hot_expire_cancelled:
            self.hot_expire_cancelled = True
            raise asyncio.CancelledError
        return await super().expire(key, seconds)


async def test_enter_cancel_after_increment_decrements_and_sets_hot_ttl() -> None:
    redis = _CancelOnFirstHotExpireRedis()
    supervisor = _build_supervisor(redis)

    with pytest.raises(asyncio.CancelledError):
        async with supervisor.subscriber_scope(
            session_id="session-1",
            connection_id="tab-1",
        ):
            pass

    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "0"
    assert redis.ttls["supervisor:hot:session-1"] == 300
    assert "supervisor:owner:session-1" not in redis.strings


class _SlowCleanupRedis(_FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.decrement_started = asyncio.Event()
        self.allow_decrement = asyncio.Event()

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        if key == "supervisor:hot:session-1" and amount == -1:
            self.decrement_started.set()
            await self.allow_decrement.wait()
        return await super().hincrby(key, field, amount)

    async def eval(self, script: str, numkeys: int, *args: object) -> int:
        if "HINCRBY" in script and str(args[0]) == "supervisor:hot:session-1":
            self.decrement_started.set()
            await self.allow_decrement.wait()
        return await super().eval(script, numkeys, *args)


async def test_exit_cancel_detaches_cleanup_so_count_is_not_leaked() -> None:
    redis = _SlowCleanupRedis()
    supervisor = _build_supervisor(redis)
    scope_entered = asyncio.Event()

    async def run_scope() -> None:
        async with supervisor.subscriber_scope(
            session_id="session-1",
            connection_id="tab-1",
        ):
            scope_entered.set()
            await asyncio.Future()

    task = asyncio.create_task(run_scope())
    await scope_entered.wait()
    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "1"

    task.cancel()
    await redis.decrement_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    redis.allow_decrement.set()
    await asyncio.sleep(0)

    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "0"
    assert "supervisor:owner:session-1" not in redis.strings


class _OuterCancelDuringEnterRedis(_FakeRedis):
    def __init__(self) -> None:
        super().__init__()
        self.enter_started = asyncio.Event()
        self.allow_enter = asyncio.Event()

    async def hincrby(self, key: str, field: str, amount: int) -> int:
        if key == "supervisor:hot:session-1" and amount == 1:
            self.enter_started.set()
            try:
                await self.allow_enter.wait()
            except asyncio.CancelledError:
                await self.allow_enter.wait()
                await super().hincrby(key, field, amount)
                raise
        return await super().hincrby(key, field, amount)


async def test_outer_cancel_during_enter_cleans_up_when_inner_increment_finishes() -> None:
    redis = _OuterCancelDuringEnterRedis()
    supervisor = _build_supervisor(redis)

    async def run_scope() -> None:
        async with supervisor.subscriber_scope(
            session_id="session-1",
            connection_id="tab-1",
        ):
            pass

    task = asyncio.create_task(run_scope())
    await redis.enter_started.wait()

    task.cancel()
    redis.allow_enter.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert redis.hashes["supervisor:hot:session-1"]["subscriber_count"] == "0"
    assert redis.ttls["supervisor:hot:session-1"] == 300


async def test_cleanup_continues_when_renew_task_already_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)
    hot_key = "supervisor:hot:session-1"
    owner_key = "supervisor:owner:session-1"
    redis.hashes[hot_key] = {"subscriber_count": "1"}
    redis.strings[owner_key] = "tab-1"
    logged_warnings: list[str] = []

    def record_warning(message: str, *args: object, **kwargs: object) -> None:
        logged_warnings.append(message)

    monkeypatch.setattr(supervisor_module.logger, "warning", record_warning)

    async def failed_renew() -> None:
        raise RuntimeError("renew failed")

    renew_task = asyncio.create_task(failed_renew())
    await asyncio.sleep(0)

    await supervisor._cleanup_subscriber_scope(
        hot_key=hot_key,
        owner_key=owner_key,
        connection_id="tab-1",
        lease_acquired=True,
        renew_task=renew_task,
    )

    assert redis.hashes[hot_key]["subscriber_count"] == "0"
    assert owner_key not in redis.strings
    assert logged_warnings == [
        "subscriber scope owner renew task failed before cleanup"
    ]


async def test_cleanup_does_not_recreate_expired_hot_hash() -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)
    hot_key = "supervisor:hot:session-1"
    owner_key = "supervisor:owner:session-1"
    redis.strings[owner_key] = "tab-1"

    await supervisor._cleanup_subscriber_scope(
        hot_key=hot_key,
        owner_key=owner_key,
        connection_id="tab-1",
        lease_acquired=True,
        renew_task=None,
    )

    assert hot_key not in redis.hashes
    assert hot_key not in redis.ttls
    assert owner_key not in redis.strings


async def test_owner_renew_uses_compare_and_expire_lua(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = _FakeRedis()
    supervisor = _build_supervisor(redis)
    owner_key = "supervisor:owner:session-1"
    redis.strings[owner_key] = "tab-1"
    sleep_calls = 0

    async def fake_sleep(seconds: int) -> None:
        nonlocal sleep_calls
        assert seconds == 5
        sleep_calls += 1
        if sleep_calls == 1:
            redis.strings[owner_key] = "tab-2"

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    await supervisor._renew_owner_lease(
        owner_key=owner_key,
        connection_id="tab-1",
    )

    assert len(redis.eval_calls) == 1
    script, numkeys, args = redis.eval_calls[0]
    assert "EXPIRE" in script
    assert numkeys == 1
    assert args == (owner_key, "tab-1", 10)
    assert redis.expire_calls == []
    assert redis.strings[owner_key] == "tab-2"
