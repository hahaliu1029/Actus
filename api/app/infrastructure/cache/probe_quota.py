"""Atomic Redis Lua quota for active subagent research probes per user.

Why atomic: a naive HLEN + HSET sequence (even pipelined) is not atomic —
two concurrent acquires can both observe count=1 and both insert,
yielding count=3 when max=2. Lua script runs server-side as a single
atomic unit on Redis main thread.

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

import logging
import time
from typing import Final

from app.infrastructure.storage.redis import RedisClient

logger = logging.getLogger(__name__)

MAX_ACTIVE_PROBES_PER_USER_DEFAULT: Final[int] = 2
PROBE_QUOTA_TTL_SECONDS: Final[int] = 900  # 15 min, exceeds D5 watchdog 600s

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
