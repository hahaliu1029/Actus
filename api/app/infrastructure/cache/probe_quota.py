"""Atomic Redis Lua quota for active subagent research probes per user.

This module also exposes coordinator-scoped daily-cost + concurrency quotas
used by C2 PR-6 §14.3 (#2 per-user daily cost cap, #3 per-user concurrency
cap). Those methods deliberately use plain INCR/INCRBYFLOAT + conditional
rollback rather than a Lua script (see "Coordinator quota v1 trade-off"
below).

Why atomic (probe quota only): a naive HLEN + HSET sequence (even
pipelined) is not atomic — two concurrent acquires can both observe
count=1 and both insert, yielding count=3 when max=2. Lua script runs
server-side as a single atomic unit on Redis main thread.

Coordinator quota v1 trade-off:
  ``acquire_coordinator_daily_cost`` and ``acquire_coordinator_concurrency``
  execute INCR(BYFLOAT) and a conditional rollback as two separate Redis
  commands. This is NOT atomic — a concurrent acquire can briefly observe
  the temporarily-overshoot counter before the rollback lands. We accept
  this because (a) the rollback closes the window in O(ms), (b) the
  coordinator concurrency cap is small (default 2), and (c) overshoot
  tolerance for daily cost is bounded by a single in-flight call's
  ``cost_usd``. If the cap is later raised or strict bounds are required,
  promote to a Lua script.

Concurrency key TTL (codex round 3 P1-5):
  ``acquire_coordinator_concurrency`` issues ``EXPIRE`` with
  ``CONCURRENCY_TTL_SECONDS = 21600`` (6h) after every successful INCR. The
  TTL is a crash-recovery ceiling, NOT a normal lifecycle — the happy
  path releases the slot via ``release_coordinator_quotas`` (DECR) in
  ``reducer_node.finally`` long before the TTL fires. The 6h ceiling
  comfortably exceeds the longest reasonable coordinator run
  (``MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN = 900`` + supervisor backstop
  ``SUBAGENT_RESULT_READY_TIMEOUT_SECONDS = 600``) plus generous
  manual-ops grace, so a pod that crashes between the INCR at dispatch and
  the DECR at reducer cannot permanently leak the user's slot until ops
  manually issues ``DECR`` / ``DEL``. EXPIRE failures are logged but do
  NOT roll back the acquire (the slot is still held; worst case it
  outlives a crash by the 6h window). EXPIRE on every acquire re-arms the
  TTL — multiple concurrent acquires keep the most recent expiry, which
  is fine since the counter is decremented on each release independently.

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
window — a deliberately loose ceiling so transient client crashes don't
permanently leak slots. The graph-level watchdog (D5
`total_timeout_seconds = 600`) caps any single probe run; the 300s gap
absorbs scheduler / cleanup jitter. Callers MUST treat 15 min as the hard
upper bound for a probe's lifetime; long-running probes should
periodically re-acquire (refresh) or get a higher ttl by callers' choice.

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
from typing import Final

from app.infrastructure.storage.redis import RedisClient

logger = logging.getLogger(__name__)

MAX_ACTIVE_PROBES_PER_USER_DEFAULT: Final[int] = 2
PROBE_QUOTA_TTL_SECONDS: Final[int] = 900  # 15 min, exceeds D5 watchdog 600s
# [codex R3 P1-5] 6h crash-recovery ceiling on the per-user concurrency
# counter. See module docstring "Concurrency key TTL" for the full
# rationale; the happy path releases via DECR in reducer_node.finally
# long before the TTL fires.
CONCURRENCY_TTL_SECONDS: Final[int] = 21600  # 6h

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
    ) -> None:
        self._redis = redis_client
        self._max_active = max_active
        self._ttl_seconds = ttl_seconds

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
    # These methods are NOT idempotent — each caller's coordinator_run_id is
    # unique. Callers MUST pair every successful acquire with
    # ``release_coordinator_quotas`` inside a try/finally.

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
        """Per-user concurrency counter key.

        TTL behavior: ``acquire_coordinator_concurrency`` issues
        ``EXPIRE key CONCURRENCY_TTL_SECONDS`` (6h) on every successful
        INCR. The TTL exists solely as a crash-recovery ceiling — the
        normal lifecycle releases via DECR in
        ``release_coordinator_quotas`` (called from
        ``reducer_node.finally``) long before the TTL fires. The 6h
        ceiling deliberately exceeds the longest reasonable coordinator
        run (``MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN = 900`` + supervisor
        backstop ``SUBAGENT_RESULT_READY_TIMEOUT_SECONDS = 600`` + ops
        grace) so a pod crash between dispatch and reducer cannot
        permanently pin the user's slots.

        If DECR drives the value below 0 due to a release-without-
        acquire bug, that is an observability problem surfaced through
        monitoring, not a hidden silent state.
        """
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
        self, *, user_id: str, cap: int
    ) -> bool:
        """Reserve one concurrency slot against the user's cap.

        Semantics: INCR then compare. If the new value strictly exceeds
        ``cap``, roll back with DECR and return False. Reaching the cap
        exactly is allowed (predicate is strict ``>``).

        Fail-closed: any Redis error during the initial INCR → return
        False.

        [codex R3 P1-5] On successful acquire, refresh the concurrency
        key with ``EXPIRE key CONCURRENCY_TTL_SECONDS`` (6h). This is a
        crash-recovery ceiling, NOT a normal lifecycle (the happy path
        releases via DECR in ``release_coordinator_quotas`` long before
        the TTL fires). Without this, a pod that crashes between INCR
        here and the matching DECR inside ``reducer_node.finally`` would
        permanently leak the slot until ops issued a manual ``DECR`` /
        ``DEL``. EXPIRE failure is logged but does NOT roll back the
        acquire — the slot is still legitimately held, worst case it
        outlives a crash by the 6h window.
        """
        key = self._concurrency_key(user_id)
        client = self._redis.client
        try:
            new_val = await client.incr(key)
        except Exception as exc:
            logger.warning(
                "acquire_coordinator_concurrency failed (fail-closed): "
                "user_id=%s err=%s",
                user_id, exc,
            )
            return False

        if int(new_val) > cap:
            try:
                await client.decr(key)
            except Exception as exc:
                logger.warning(
                    "acquire_coordinator_concurrency rollback failed "
                    "(counter leak): user_id=%s err=%s",
                    user_id, exc,
                )
            return False

        # Crash-recovery TTL: re-arm the 6h ceiling on every successful
        # acquire. Failure is non-fatal — slot is still held; worst case
        # the key persists slightly beyond the TTL window.
        try:
            await client.expire(key, CONCURRENCY_TTL_SECONDS)
        except Exception as exc:
            logger.warning(
                "acquire_coordinator_concurrency: EXPIRE failed user=%s "
                "— slot still acquired but may not auto-release on "
                "pod crash before reducer DECR runs: %s",
                user_id, exc,
            )
        return True

    async def release_coordinator_quotas(
        self, *, user_id: str, cost_usd: float = 0.0
    ) -> None:
        """Release a coordinator quota reservation. Best-effort.

        Always decrements the concurrency counter. If ``cost_usd > 0``,
        additionally rolls back that amount from the daily counter.
        Negative or zero ``cost_usd`` is treated as "no daily rollback
        desired" — the caller signals intent by passing the positive
        cost they previously reserved.

        Each Redis op is wrapped independently: if INCRBYFLOAT raises,
        the DECR still runs (and vice versa). All errors are logged and
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
            await client.decr(self._concurrency_key(user_id))
        except Exception as exc:
            logger.warning(
                "release_coordinator_quotas concurrency decr failed: "
                "user_id=%s err=%s",
                user_id, exc,
            )
