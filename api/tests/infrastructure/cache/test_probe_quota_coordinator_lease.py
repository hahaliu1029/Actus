"""Coordinator concurrency is a renewable run-id lease, not a counter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from app.infrastructure.cache.probe_quota import ProbeQuotaService


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class _Clock:
    value: float = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@dataclass
class _LeaseRedis:
    """Small semantic fake for the three coordinator Lua contracts."""

    leases: dict[str, dict[str, float]] = field(default_factory=dict)
    daily: dict[str, float] = field(default_factory=dict)

    @property
    def client(self) -> "_LeaseRedis":
        return self

    async def eval(
        self, script: str, numkeys: int, key: str, *args: Any,
    ) -> int:
        assert numkeys == 1
        members = self.leases.setdefault(key, {})
        run_id = str(args[0])
        if "ZREMRANGEBYSCORE" in script:
            now = float(args[1])
            cap = int(args[2])
            ttl = float(args[3])
            members = {
                member: expiry
                for member, expiry in members.items()
                if expiry > now
            }
            self.leases[key] = members
            if run_id in members:
                members[run_id] = now + ttl
                return 1
            if len(members) >= cap:
                return 0
            members[run_id] = now + ttl
            return 1
        if "ZSCORE" in script:
            now = float(args[1])
            ttl = float(args[2])
            if run_id not in members:
                return 0
            if members[run_id] <= now:
                del members[run_id]
                return 0
            members[run_id] = now + ttl
            return 1
        if "ZREM" in script:
            return int(members.pop(run_id, None) is not None)
        raise AssertionError("unexpected coordinator quota script")

    async def incrbyfloat(self, key: str, amount: float) -> float:
        value = self.daily.get(key, 0.0) + amount
        self.daily[key] = value
        return value

    async def expire(self, _key: str, _seconds: int) -> bool:
        return True


def _service(clock: _Clock) -> tuple[ProbeQuotaService, _LeaseRedis]:
    redis = _LeaseRedis()
    return ProbeQuotaService(redis, clock=clock), redis


async def test_two_runs_are_independent_same_run_is_idempotent_and_cap_rejects() -> None:
    clock = _Clock()
    service, redis = _service(clock)

    assert await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="run-a", cap=2,
    )
    first_expiry = redis.leases[service._concurrency_key("user")]["run-a"]
    clock.advance(10)
    assert await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="run-a", cap=2,
    )
    assert redis.leases[service._concurrency_key("user")]["run-a"] > first_expiry
    assert await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="run-b", cap=2,
    )
    assert not await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="run-c", cap=2,
    )
    assert set(redis.leases[service._concurrency_key("user")]) == {
        "run-a", "run-b",
    }


async def test_acquire_prunes_expired_members_before_enforcing_cap() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    key = service._concurrency_key("user")
    redis.leases[key] = {"stale": clock.value}

    assert await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="new", cap=1,
    )
    assert redis.leases[key].keys() == {"new"}


async def test_renew_is_existence_only_and_never_reacquires_lost_member() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    key = service._concurrency_key("user")

    assert not await service.renew_coordinator_concurrency(
        user_id="user", coordinator_run_id="lost",
    )
    assert redis.leases[key] == {}


async def test_renew_treats_logically_expired_member_as_lost() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    key = service._concurrency_key("user")
    redis.leases[key] = {"expired": clock.value}

    assert not await service.renew_coordinator_concurrency(
        user_id="user", coordinator_run_id="expired",
    )
    assert redis.leases[key] == {}


async def test_continuous_renew_past_six_hours_keeps_the_same_slot() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    key = service._concurrency_key("user")
    assert await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="long", cap=1,
    )

    for _ in range(25):
        clock.advance(60 * 60)
        assert await service.renew_coordinator_concurrency(
            user_id="user", coordinator_run_id="long",
        )

    assert clock.value > 24 * 60 * 60
    assert set(redis.leases[key]) == {"long"}
    assert not await service.acquire_coordinator_concurrency(
        user_id="user", coordinator_run_id="other", cap=1,
    )


async def test_release_is_exact_late_idempotent_and_preserves_other_run() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    key = service._concurrency_key("user")
    for run_id in ("run-a", "run-b"):
        assert await service.acquire_coordinator_concurrency(
            user_id="user", coordinator_run_id=run_id, cap=2,
        )

    await service.release_coordinator_quotas(
        user_id="user", coordinator_run_id="run-a",
    )
    await service.release_coordinator_quotas(
        user_id="user", coordinator_run_id="run-a",
    )

    assert set(redis.leases[key]) == {"run-b"}


async def test_release_preserves_existing_daily_cost_rollback_semantics() -> None:
    clock = _Clock()
    service, redis = _service(clock)
    daily_key = service._daily_key("user")
    redis.daily[daily_key] = 12.5

    await service.release_coordinator_quotas(
        user_id="user", coordinator_run_id="run", cost_usd=2.5,
    )

    assert redis.daily[daily_key] == 10.0
