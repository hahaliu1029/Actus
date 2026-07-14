"""Tests for ProbeQuotaService coordinator daily-cost + concurrency methods.

Coverage:
- per-user daily cost cap and rollback;
- run-id ZSET acquire/renew/release service contracts;
- fail-closed acquire/renew and best-effort exact release.

These are mock-level adapter tests. Stateful lease semantics live in
``test_probe_quota_coordinator_lease.py``.
"""
from __future__ import annotations

import datetime as _dt

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.infrastructure.cache.probe_quota import ProbeQuotaService


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock RedisClient mirroring the production access pattern.

    Production code calls ``self._redis.client.incrbyfloat(...)`` /
    ``self._redis.client.incr(...)`` / ``self._redis.client.decr(...)`` /
    ``self._redis.client.expire(...)``. Tests configure those attributes
    as AsyncMocks.
    """
    redis = MagicMock()
    redis.client = MagicMock()
    redis.client.incrbyfloat = AsyncMock()
    redis.client.incr = AsyncMock()
    redis.client.decr = AsyncMock()
    redis.client.expire = AsyncMock()
    redis.client.eval = AsyncMock()
    return redis


@pytest.fixture
def quota_service(mock_redis: MagicMock) -> ProbeQuotaService:
    return ProbeQuotaService(redis_client=mock_redis)


# ── Coordinator daily cost ───────────────────────────────────────────────────


class TestCoordinatorDailyCost:
    async def test_acquire_daily_cost_under_cap(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """new_val < cap → True; expire is called with 25h TTL; no rollback."""
        mock_redis.client.incrbyfloat.return_value = 10.0

        ok = await quota_service.acquire_coordinator_daily_cost(
            user_id="u-1", cost_usd=10.0, cap_usd=50.0
        )

        assert ok is True
        # incrbyfloat called exactly once (no rollback)
        assert mock_redis.client.incrbyfloat.call_count == 1
        # key derivation uses today's UTC date
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        expected_key = f"actus:coord:daily_cost:u-1:{today}"
        mock_redis.client.incrbyfloat.assert_called_once_with(expected_key, 10.0)
        # expire called with 25h
        mock_redis.client.expire.assert_called_once_with(expected_key, 25 * 3600)

    async def test_acquire_daily_cost_exceeds_cap(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """new_val > cap → False; rollback INCRBYFLOAT(-cost); no expire."""
        # First call returns 60 (over cap 50); second (rollback) returns 50.
        mock_redis.client.incrbyfloat.side_effect = [60.0, 50.0]

        ok = await quota_service.acquire_coordinator_daily_cost(
            user_id="u-1", cost_usd=10.0, cap_usd=50.0
        )

        assert ok is False
        assert mock_redis.client.incrbyfloat.call_count == 2
        # second call is the rollback with negative cost
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        expected_key = f"actus:coord:daily_cost:u-1:{today}"
        rollback_call = mock_redis.client.incrbyfloat.call_args_list[1]
        assert rollback_call.args == (expected_key, -10.0)
        # expire MUST NOT be called when rejected
        mock_redis.client.expire.assert_not_called()

    async def test_acquire_daily_cost_exactly_at_cap_is_allowed(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """new_val == cap → True (predicate is strict `>`)."""
        mock_redis.client.incrbyfloat.return_value = 50.0

        ok = await quota_service.acquire_coordinator_daily_cost(
            user_id="u-1", cost_usd=50.0, cap_usd=50.0
        )

        assert ok is True
        # No rollback
        assert mock_redis.client.incrbyfloat.call_count == 1
        # TTL refresh
        mock_redis.client.expire.assert_called_once()

    async def test_acquire_daily_cost_key_includes_user_and_date(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Daily key uses ISO date + user_id for per-user-per-day isolation."""
        mock_redis.client.incrbyfloat.return_value = 1.0

        await quota_service.acquire_coordinator_daily_cost(
            user_id="user-42", cost_usd=1.0, cap_usd=100.0
        )

        call_args = mock_redis.client.incrbyfloat.call_args
        key = call_args.args[0]
        assert key.startswith("actus:coord:daily_cost:user-42:")
        # ISO YYYY-MM-DD shape after the user id
        date_suffix = key.rsplit(":", 1)[-1]
        # Should parse as a valid ISO date.
        _dt.date.fromisoformat(date_suffix)

    async def test_daily_cost_exact_cap_then_tiny_increment_rejected(
        self,
    ) -> None:
        """[codex R2 P2-1 boundary] Cumulative reaches cap exactly (allowed),
        then a tiny additional increment trips + rolls back.

        Pins the strict-``>`` predicate at the cap edge: the first
        acquire takes the counter to 50.0 (==cap, allowed); a follow-up
        0.001 acquire would take it to 50.001 (>cap, rejected) and the
        rollback INCRBYFLOAT(-0.001) must fire so the counter returns
        to 50.0. Without the rollback the rejected attempt would leak
        cost into the bucket and starve future legitimate acquires.
        """
        redis_client = MagicMock()
        redis_client.client = MagicMock()
        # First acquire (50.0 → at cap) returns 50.0 (one call).
        # Second acquire over-cap path: incrbyfloat(0.001) → 50.001,
        #     then rollback incrbyfloat(-0.001) → 50.0.
        # Total 3 incrbyfloat calls observed across the two acquires.
        redis_client.client.incrbyfloat = AsyncMock(
            side_effect=[50.0, 50.001, 50.0]
        )
        redis_client.client.expire = AsyncMock()
        svc = ProbeQuotaService(redis_client)

        first = await svc.acquire_coordinator_daily_cost(
            user_id="u1", cost_usd=50.0, cap_usd=50.0,
        )
        assert first is True

        second = await svc.acquire_coordinator_daily_cost(
            user_id="u1", cost_usd=0.001, cap_usd=50.0,
        )
        assert second is False

        # 1 acquire (first) + 1 acquire-then-rollback (second) = 3 calls.
        assert redis_client.client.incrbyfloat.await_count == 3
        # The third call is the rollback with the NEGATIVE second cost.
        rollback_call = redis_client.client.incrbyfloat.await_args_list[2]
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        expected_key = f"actus:coord:daily_cost:u1:{today}"
        assert rollback_call.args == (expected_key, -0.001)
        # TTL refresh fired ONCE — for the first (successful) acquire only.
        # The rejected second acquire MUST NOT refresh TTL.
        assert redis_client.client.expire.await_count == 1

    async def test_acquire_daily_cost_redis_error_fail_closed(self) -> None:
        """[fail-closed] If INCRBYFLOAT raises, return False — do NOT raise."""
        redis_client = MagicMock()
        redis_client.client = MagicMock()
        redis_client.client.incrbyfloat = AsyncMock(
            side_effect=RuntimeError("redis down")
        )
        svc = ProbeQuotaService(redis_client)
        result = await svc.acquire_coordinator_daily_cost(
            user_id="u1", cost_usd=5.0, cap_usd=50.0,
        )
        assert result is False


# ── Coordinator concurrency ──────────────────────────────────────────────────


class TestCoordinatorConcurrency:
    async def test_acquire_concurrency_under_cap(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Lua accepts a new run-id member."""
        mock_redis.client.eval.return_value = 1

        ok = await quota_service.acquire_coordinator_concurrency(
            user_id="u-1", coordinator_run_id="run-1", cap=2
        )

        assert ok is True
        assert mock_redis.client.eval.await_args.args[2:4] == (
            "actus:coord:concurrent:u-1", "run-1",
        )

    async def test_acquire_concurrency_at_cap_allowed(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Lua result is authoritative at the cap boundary."""
        mock_redis.client.eval.return_value = 1

        ok = await quota_service.acquire_coordinator_concurrency(
            user_id="u-1", coordinator_run_id="run-2", cap=2
        )

        assert ok is True

    async def test_acquire_concurrency_exceeds_cap_rejected(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Lua rejects a new member when the pruned ZSET is at cap."""
        mock_redis.client.eval.return_value = 0

        ok = await quota_service.acquire_coordinator_concurrency(
            user_id="u-1", coordinator_run_id="run-3", cap=2
        )

        assert ok is False

    async def test_acquire_concurrency_redis_error_fail_closed(self) -> None:
        """[fail-closed] If EVAL raises, return False — do NOT raise."""
        redis_client = MagicMock()
        redis_client.client = MagicMock()
        redis_client.client.eval = AsyncMock(side_effect=RuntimeError("redis down"))
        svc = ProbeQuotaService(redis_client)
        result = await svc.acquire_coordinator_concurrency(
            user_id="u1", coordinator_run_id="run", cap=2,
        )
        assert result is False

    # ── Crash-recovery TTL (codex round 3 P1-5) ─────────────────────────

    async def test_acquire_concurrency_sets_ttl_for_crash_recovery(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Acquire passes the six-hour one-shot cleanup window to Lua."""
        from app.infrastructure.cache.probe_quota import (
            CONCURRENCY_TTL_SECONDS,
        )

        mock_redis.client.eval.return_value = 1

        ok = await quota_service.acquire_coordinator_concurrency(
            user_id="u-1", coordinator_run_id="run-1", cap=2
        )

        assert ok is True
        assert mock_redis.client.eval.await_args.args[-1] == str(
            CONCURRENCY_TTL_SECONDS
        )
        assert CONCURRENCY_TTL_SECONDS == 21600  # 6h = 21600s
        assert CONCURRENCY_TTL_SECONDS > 3600

    async def test_acquire_concurrency_rejection_uses_single_atomic_eval(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """A rejected acquire has no client-side rollback command."""
        mock_redis.client.eval.return_value = 0

        ok = await quota_service.acquire_coordinator_concurrency(
            user_id="u-1", coordinator_run_id="run-3", cap=2
        )

        assert ok is False
        mock_redis.client.expire.assert_not_called()

    async def test_acquire_concurrency_eval_failure_is_observable_and_closed(
        self,
        quota_service: ProbeQuotaService,
        mock_redis: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Atomic EVAL failure logs and rejects instead of bypassing cap."""
        import logging

        mock_redis.client.eval.side_effect = RuntimeError("eval failed")

        with caplog.at_level(
            logging.WARNING, logger="app.infrastructure.cache.probe_quota"
        ):
            ok = await quota_service.acquire_coordinator_concurrency(
                user_id="u-1", coordinator_run_id="run-1", cap=2
            )

        assert ok is False
        matching = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "acquire_coordinator_concurrency" in r.getMessage()
        ]
        assert matching, (
            "expected a WARNING log mentioning acquire failure; got: "
            f"{[r.getMessage() for r in caplog.records]}"
        )


# ── Release ───────────────────────────────────────────────────────────────────


class TestRelease:
    async def test_release_with_cost_decrements_both(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """cost rollback and exact run-id ZREM are independent operations."""
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1", cost_usd=12.5
        )

        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        daily_key = f"actus:coord:daily_cost:u-1:{today}"
        conc_key = "actus:coord:concurrent:u-1"

        mock_redis.client.incrbyfloat.assert_called_once_with(daily_key, -12.5)
        assert mock_redis.client.eval.await_args.args[2:] == (conc_key, "run-1")

    async def test_release_zero_cost_only_removes_run_member(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """cost == 0 skips daily rollback and still removes the run member."""
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1", cost_usd=0.0
        )

        mock_redis.client.incrbyfloat.assert_not_called()
        assert mock_redis.client.eval.await_args.args[2:] == (
            "actus:coord:concurrent:u-1", "run-1",
        )

    async def test_release_default_cost_only_removes_run_member(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Default cost_usd=0.0 → no daily rollback."""
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1",
        )

        mock_redis.client.incrbyfloat.assert_not_called()
        mock_redis.client.eval.assert_awaited_once()

    async def test_release_swallows_redis_error_on_incrbyfloat(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Daily rollback failure must NOT raise (best-effort)."""
        mock_redis.client.incrbyfloat.side_effect = Exception("Redis down")

        # Must NOT raise.
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1", cost_usd=10.0
        )

    async def test_release_swallows_redis_error_on_zrem(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Concurrency ZREM failure must NOT raise."""
        mock_redis.client.eval.side_effect = Exception("Redis down")

        # Must NOT raise.
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1", cost_usd=0.0
        )

    async def test_release_negative_cost_is_treated_as_no_daily_rollback(
        self, quota_service: ProbeQuotaService, mock_redis: MagicMock
    ) -> None:
        """Defensive: negative cost_usd ≤ 0 → skip daily rollback.

        Pins the contract that the caller must pass positive cost_usd
        when a daily rollback is desired.
        """
        await quota_service.release_coordinator_quotas(
            user_id="u-1", coordinator_run_id="run-1", cost_usd=-5.0
        )

        mock_redis.client.incrbyfloat.assert_not_called()
        mock_redis.client.eval.assert_awaited_once()


# ── Key derivation ────────────────────────────────────────────────────────────


class TestKeyHelpers:
    def test_daily_key_format(self) -> None:
        key = ProbeQuotaService._daily_key("user-7")
        today = _dt.datetime.now(_dt.UTC).date().isoformat()
        assert key == f"actus:coord:daily_cost:user-7:{today}"

    def test_concurrency_key_format(self) -> None:
        key = ProbeQuotaService._concurrency_key("user-7")
        assert key == "actus:coord:concurrent:user-7"
