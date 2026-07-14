"""Atomic Redis Lua quota for active subagent research probes per user.

This module also exposes coordinator-scoped daily-cost and concurrency
quotas. Daily cost retains its INCRBYFLOAT + conditional rollback contract.
Concurrency is a per-user sorted-set lease: member is coordinator_run_id and
score is the crash-cleanup expiry epoch.

Why atomic (probe quota only): a naive HLEN + HSET sequence (even
pipelined) is not atomic — two concurrent acquires can both observe
count=1 and both insert, yielding count=3 when max=2. Lua script runs
server-side as a single atomic unit on Redis main thread.

Coordinator concurrency Lua contracts:
- acquire prunes expired members, refreshes the same run idempotently, then
  enforces the cap before adding a new member;
- renew refreshes only an existing, unexpired member and never reacquires;
- release removes the exact run id and is naturally idempotent.

The six-hour value is one crash-cleanup window, not a task wallclock.
Long-running active runs periodically renew the same member and can therefore
run beyond six hours. The outer run-scoped backend owner stops and drains the
renew loop before exact release, so a late renew cannot recreate ownership.

Lua behavior:
1. Read all field/timestamp pairs from the hash (HGETALL).
2. Remove entries older than `ttl_seconds` (stale cleanup at acquire time).
3. If `probe_run_id` already in hash (post-cleanup), refresh ts + TTL and
   return 1 — same probe retrying must not be quota-rejected by its own
   prior live lease. A previously-cleaned stale entry falls through to (4).
4. Otherwise re-check count. If >= max_active, return 0.
5. Otherwise HSET the new probe_run_id with current timestamp + EXPIRE key
   to refresh hash TTL. Return 1.

TTL note: `PROBE_QUOTA_TTL_SECONDS = 900` (15 min) is the per-slot lease
window — an independent crash-recovery ceiling so transient client crashes
don't permanently leak slots. Root execution wallclock defaults to unlimited,
so callers MUST treat this 15 min quota lease as its own hard upper bound;
long-running probes should periodically re-acquire (refresh) or get a higher
ttl by callers' choice.

Failure modes:
- Redis exception during eval → acquire returns False (fail closed).
- Stale cleanup races with another acquire → Lua atomicity prevents.
- Crash after acquire but before release → entry self-expires after ttl.
- Same probe_run_id retried while full → allowed (refresh of own slot).
"""
from __future__ import annotations

import datetime as _dt
import logging
import time
from typing import Callable, Final

from app.infrastructure.storage.redis import RedisClient

logger = logging.getLogger(__name__)

MAX_ACTIVE_PROBES_PER_USER_DEFAULT: Final[int] = 2
PROBE_QUOTA_TTL_SECONDS: Final[int] = 900  # 15 min crash-recovery lease
# One crash-cleanup window for a coordinator run-id lease. Active runs renew.
CONCURRENCY_TTL_SECONDS: Final[int] = 21600  # 6h

ACQUIRE_COORDINATOR_CONCURRENCY_LUA: Final[str] = """
local key = KEYS[1]
local run_id = ARGV[1]
local now = tonumber(ARGV[2])
local cap = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

redis.call('ZREMRANGEBYSCORE', key, '-inf', now)
if redis.call('ZSCORE', key, run_id) then
    redis.call('ZADD', key, now + ttl, run_id)
    redis.call('EXPIRE', key, ttl)
    return 1
end
if redis.call('ZCARD', key) >= cap then
    return 0
end
redis.call('ZADD', key, now + ttl, run_id)
redis.call('EXPIRE', key, ttl)
return 1
"""

RENEW_COORDINATOR_CONCURRENCY_LUA: Final[str] = """
local key = KEYS[1]
local run_id = ARGV[1]
local now = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

local expiry = redis.call('ZSCORE', key, run_id)
if not expiry then
    return 0
end
if tonumber(expiry) <= now then
    redis.call('ZREM', key, run_id)
    return 0
end
redis.call('ZADD', key, now + ttl, run_id)
redis.call('EXPIRE', key, ttl)
return 1
"""

RELEASE_COORDINATOR_CONCURRENCY_LUA: Final[str] = """
return redis.call('ZREM', KEYS[1], ARGV[1])
"""

# Lua script: stale-cleanup + idempotent-refresh + count-check + insert, all atomic.
ACQUIRE_PROBE_LUA: Final[str] = """
local key = KEYS[1]
local probe_id = ARGV[1]
local now = tonumber(ARGV[2])
local max_active = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])

-- Clean entries older than ttl (stale cleanup on every acquire).
-- Safe to HDEL during iteration: we iterate the local HGETALL snapshot, not live hash.
local cutoff = now - ttl
local fields = redis.call('HGETALL', key)
for i = 1, #fields, 2 do
    local ts = tonumber(fields[i+1])
    if ts == nil or ts < cutoff then
        redis.call('HDEL', key, fields[i])
    end
end

-- Idempotent refresh: same probe_run_id retrying must NOT be rejected by its own
-- live lease. Only consult the post-cleanup state (stale own entries fall through
-- to the count check and are treated as new).
if redis.call('HEXISTS', key, probe_id) == 1 then
    redis.call('HSET', key, probe_id, now)
    redis.call('EXPIRE', key, ttl)
    return 1
end

-- New probe — enforce quota.
local count = redis.call('HLEN', key)
if count >= max_active then
    return 0
end

-- Insert new entry and refresh hash TTL.
redis.call('HSET', key, probe_id, now)
redis.call('EXPIRE', key, ttl)
return 1
"""


class ProbeQuotaService:
    """Per-user active probe quota with atomic Lua acquire/release."""

    def __init__(
        self,
        redis_client: RedisClient,
        max_active: int = MAX_ACTIVE_PROBES_PER_USER_DEFAULT,
        ttl_seconds: int = PROBE_QUOTA_TTL_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._redis = redis_client
        self._max_active = max_active
        self._ttl_seconds = ttl_seconds
        self._clock = clock

    @staticmethod
    def _key(user_id: str) -> str:
        return f"actus:active_probes:{user_id}"

    async def acquire(self, user_id: str, probe_run_id: str) -> bool:
        """Atomically acquire a probe slot. Returns True on success.

        Fail-closed semantics: any Redis error → False. Better to deny a
        legit probe than let quota be bypassed during Redis outage.
        """
        try:
            client = self._redis.client
            result = await client.eval(
                ACQUIRE_PROBE_LUA,
                1,  # number of keys
                self._key(user_id),
                probe_run_id,
                str(time.time()),
                str(self._max_active),
                str(self._ttl_seconds),
            )
            return bool(int(result))
        except Exception as e:
            logger.warning(
                "probe_quota acquire failed (fail-closed): user_id=%s probe_run_id=%s err=%s",
                user_id, probe_run_id, e,
            )
            return False

    async def release(self, user_id: str, probe_run_id: str) -> None:
        """Release a probe slot. Best-effort: errors logged, not raised."""
        try:
            client = self._redis.client
            await client.hdel(self._key(user_id), probe_run_id)
        except Exception as e:
            logger.warning(
                "probe_quota release failed: user_id=%s probe_run_id=%s err=%s",
                user_id, probe_run_id, e,
            )
            # Don't raise — release is in finally block; stale entries
            # self-expire via TTL.

    # ── Coordinator quotas (§14.3 #2 / #3) ──────────────────────────────────
    #
    # Concurrency acquire/release are idempotent for the same run id. The
    # run-scoped backend owner pairs successful acquire with exact release.

    @staticmethod
    def _daily_key(user_id: str) -> str:
        """Per-user-per-day cost counter key.

        Date is computed in UTC (explicit ``datetime.now(UTC).date()``) so
        rollover happens at UTC midnight regardless of host timezone. Combined
        with the 25h EXPIRE this yields a one-hour grace window where two
        days' keys coexist (acceptable: each key has independent cap
        accounting).
        """
        d = _dt.datetime.now(_dt.UTC).date().isoformat()
        return f"actus:coord:daily_cost:{user_id}:{d}"

    @staticmethod
    def _concurrency_key(user_id: str) -> str:
        """Per-user sorted set of run-id leases scored by expiry epoch."""
        return f"actus:coord:concurrent:{user_id}"

    async def acquire_coordinator_daily_cost(
        self, *, user_id: str, cost_usd: float, cap_usd: float
    ) -> bool:
        """Reserve ``cost_usd`` against the user's daily cost cap.

        Semantics: INCRBYFLOAT then compare. If the new value strictly
        exceeds ``cap_usd``, roll back with INCRBYFLOAT(-cost_usd) and
        return False. Reaching the cap exactly is allowed (predicate is
        strict ``>``). On success, refresh a 25h TTL so the daily key
        survives midnight rollover with overlap.

        Fail-closed: any Redis error during the initial INCRBYFLOAT →
        return False (matches probe ``acquire`` semantics).
        """
        key = self._daily_key(user_id)
        client = self._redis.client
        try:
            new_val = await client.incrbyfloat(key, cost_usd)
        except Exception as exc:
            logger.warning(
                "acquire_coordinator_daily_cost failed (fail-closed): "
                "user_id=%s cost_usd=%s err=%s",
                user_id, cost_usd, exc,
            )
            return False

        if float(new_val) > cap_usd:
            # Over cap → rollback. Rollback errors are logged but do not
            # change the rejection outcome.
            try:
                await client.incrbyfloat(key, -cost_usd)
            except Exception as exc:
                logger.warning(
                    "acquire_coordinator_daily_cost rollback failed "
                    "(counter leak until TTL): user_id=%s cost_usd=%s err=%s",
                    user_id, cost_usd, exc,
                )
            return False

        # Under or at cap → refresh TTL. expire errors are non-fatal: the
        # counter still reflects the reservation; worst case the key
        # persists slightly longer than 25h.
        try:
            await client.expire(key, 25 * 3600)
        except Exception as exc:
            logger.warning(
                "acquire_coordinator_daily_cost expire failed: "
                "user_id=%s err=%s",
                user_id, exc,
            )
        return True

    async def acquire_coordinator_concurrency(
        self, *, user_id: str, coordinator_run_id: str, cap: int
    ) -> bool:
        """Atomically acquire or idempotently refresh one run-id lease."""
        try:
            result = await self._redis.client.eval(
                ACQUIRE_COORDINATOR_CONCURRENCY_LUA,
                1,
                self._concurrency_key(user_id),
                coordinator_run_id,
                str(self._clock()),
                str(cap),
                str(CONCURRENCY_TTL_SECONDS),
            )
            return bool(int(result))
        except Exception as exc:
            logger.warning(
                "acquire_coordinator_concurrency failed (fail-closed): "
                "user_id=%s coordinator_run_id=%s err=%s",
                user_id, coordinator_run_id, exc,
            )
            return False

    async def renew_coordinator_concurrency(
        self, *, user_id: str, coordinator_run_id: str,
    ) -> bool:
        """Refresh an existing live lease; never recreate a lost lease."""
        try:
            result = await self._redis.client.eval(
                RENEW_COORDINATOR_CONCURRENCY_LUA,
                1,
                self._concurrency_key(user_id),
                coordinator_run_id,
                str(self._clock()),
                str(CONCURRENCY_TTL_SECONDS),
            )
            return bool(int(result))
        except Exception as exc:
            logger.warning(
                "renew_coordinator_concurrency failed (fail-closed): "
                "user_id=%s coordinator_run_id=%s err=%s",
                user_id, coordinator_run_id, exc,
            )
            return False

    async def release_coordinator_quotas(
        self,
        *,
        user_id: str,
        coordinator_run_id: str,
        cost_usd: float = 0.0,
    ) -> None:
        """Release a coordinator quota reservation. Best-effort.

        Always removes the exact concurrency run-id member. If ``cost_usd > 0``,
        additionally rolls back that amount from the daily counter.
        Negative or zero ``cost_usd`` is treated as "no daily rollback
        desired" — the caller signals intent by passing the positive
        cost they previously reserved.

        Each Redis op is wrapped independently: if INCRBYFLOAT raises,
        the exact ZREM still runs (and vice versa). All errors are logged and
        swallowed (matches the existing ``release`` pattern).
        """
        client = self._redis.client
        if cost_usd > 0:
            try:
                await client.incrbyfloat(self._daily_key(user_id), -cost_usd)
            except Exception as exc:
                logger.warning(
                    "release_coordinator_quotas daily rollback failed: "
                    "user_id=%s cost_usd=%s err=%s",
                    user_id, cost_usd, exc,
                )
        try:
            await client.eval(
                RELEASE_COORDINATOR_CONCURRENCY_LUA,
                1,
                self._concurrency_key(user_id),
                coordinator_run_id,
            )
        except Exception as exc:
            logger.warning(
                "release_coordinator_quotas concurrency zrem failed: "
                "user_id=%s coordinator_run_id=%s err=%s",
                user_id, coordinator_run_id, exc,
            )
