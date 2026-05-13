"""Redis Lua scripts for B3 ExecutionSupervisor admission accounting."""

from __future__ import annotations

import hashlib
import logging
from typing import Sequence

from redis.exceptions import NoScriptError

logger = logging.getLogger(__name__)


LUA_ADMIT_SOURCE = """
local sys_key = KEYS[1]
local user_key = KEYS[2]
local hot_key = KEYS[3]
local bg_zset_key = KEYS[4]
local session_id = ARGV[1]
local expires_at = ARGV[2]
local max_sys = tonumber(ARGV[3])
local max_user = tonumber(ARGV[4])
local activity_at = ARGV[5] or expires_at

local expires_score = tonumber(expires_at)
if expires_score == nil then
    return redis.error_reply('LUA_ADMIT: invalid ARGV[2] expires_at_unix')
end

if tonumber(activity_at) == nil then
    return redis.error_reply('LUA_ADMIT: invalid ARGV[5] activity_at_unix')
end

if redis.call('HEXISTS', user_key, session_id) == 1 then
    redis.call('HSET', user_key, session_id, expires_at)
    redis.call('EXPIRE', user_key, 86400)
    redis.call('ZADD', bg_zset_key, expires_score, session_id)
    redis.call('EXPIRE', bg_zset_key, 86400)
    redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
    redis.call('HSET', hot_key, 'last_activity_at', activity_at)
    redis.call('EXPIRE', hot_key, 300)
    return 3
end

local user_count = redis.call('HLEN', user_key)
if user_count >= max_user then
    return 2
end

local sys_count = tonumber(redis.call('GET', sys_key) or '0')
if sys_count >= max_sys then
    return 1
end

redis.call('HSET', user_key, session_id, expires_at)
redis.call('EXPIRE', user_key, 86400)
redis.call('INCR', sys_key)
redis.call('ZADD', bg_zset_key, expires_score, session_id)
redis.call('EXPIRE', bg_zset_key, 86400)
redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
redis.call('HSET', hot_key, 'last_activity_at', activity_at)
redis.call('EXPIRE', hot_key, 300)
return 0
"""


LUA_REVOKE_SOURCE = """
local sys_key = KEYS[1]
local user_key = KEYS[2]
local bg_zset_key = KEYS[3]
local session_id = ARGV[1]

local existed = redis.call('HDEL', user_key, session_id)
redis.call('ZREM', bg_zset_key, session_id)
if existed == 1 then
    local sys_count = tonumber(redis.call('GET', sys_key) or '0')
    if sys_count > 0 then
        redis.call('DECR', sys_key)
    end
end
return existed
"""


LUA_SWEEP_EXPIRED_SOURCE = """
local zset_key = KEYS[1]
local now = tonumber(ARGV[1])
if now == nil then
    return redis.error_reply('LUA_SWEEP_EXPIRED: invalid ARGV[1] now_unix')
end

local expired = redis.call('ZRANGEBYSCORE', zset_key, '-inf', now)
return expired
"""


LUA_ADMIT_SHA = hashlib.sha1(LUA_ADMIT_SOURCE.encode("utf-8")).hexdigest()
LUA_REVOKE_SHA = hashlib.sha1(LUA_REVOKE_SOURCE.encode("utf-8")).hexdigest()
LUA_SWEEP_EXPIRED_SHA = hashlib.sha1(
    LUA_SWEEP_EXPIRED_SOURCE.encode("utf-8")
).hexdigest()

# Short aliases for later supervisor code.
LUA_ADMIT = LUA_ADMIT_SOURCE
LUA_REVOKE = LUA_REVOKE_SOURCE
LUA_SWEEP_EXPIRED = LUA_SWEEP_EXPIRED_SOURCE


async def run_lua_with_fallback(
    redis_client,
    *,
    source: str,
    sha: str,
    keys: Sequence[str],
    args: Sequence[str | int | float],
    meter=None,
) -> object:
    """Run EVALSHA and reload the script on NOSCRIPT."""
    try:
        return await redis_client.evalsha(sha, len(keys), *keys, *args)
    except NoScriptError:
        logger.warning("Redis Lua NOSCRIPT; reloading script sha=%s", sha)
        if meter is not None:
            try:
                meter.create_counter(
                    "actus_supervisor_lua_noscript_fallback_total"
                ).add(1)
            except Exception:
                logger.debug("failed to record NOSCRIPT fallback metric", exc_info=True)
        new_sha = await redis_client.script_load(source)
        return await redis_client.evalsha(new_sha, len(keys), *keys, *args)
