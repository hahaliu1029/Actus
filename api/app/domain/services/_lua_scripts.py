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
local generation_key = KEYS[5]
local system_members_key = KEYS[6]
local reconcile_marker_key = KEYS[7]
local reconcile_due_key = KEYS[8]
local session_id = ARGV[1]
local expires_at = ARGV[2]
local max_sys = tonumber(ARGV[3])
local max_user = tonumber(ARGV[4])
local activity_at = ARGV[5] or expires_at
local generation = tonumber(ARGV[6])
local authoritative_repair = tonumber(ARGV[7]) or 0
local membership_value = ARGV[8] or tostring(generation)
local slot_ttl = tonumber(ARGV[9]) or 86400
local expires_score = tonumber(expires_at)

if expires_score == nil then
    return redis.error_reply('LUA_ADMIT: invalid ARGV[2] expires_at_unix')
end
if tonumber(activity_at) == nil then
    return redis.error_reply('LUA_ADMIT: invalid ARGV[5] activity_at_unix')
end
if generation == nil then
    return redis.error_reply('LUA_ADMIT: invalid ARGV[6] generation')
end
if slot_ttl == nil then
    return redis.error_reply('LUA_ADMIT: invalid membership or ttl')
end

local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end

local function extend_ttl(key, ttl)
    local current = redis.call('TTL', key)
    if current < ttl then
        redis.call('EXPIRE', key, ttl)
    end
end

local function clear_reconcile_marker_through(generation_limit)
    local raw = redis.call('HGET', reconcile_marker_key, session_id)
    if raw == false then
        return
    end
    local encoded = string.match(tostring(raw), '^v1|[^|]+|([^|]+)|')
    local marker_generation = tonumber(encoded)
    if marker_generation ~= nil and marker_generation <= generation_limit then
        redis.call('HDEL', reconcile_marker_key, session_id)
        redis.call('ZREM', reconcile_due_key, session_id)
    end
end

local raw_reconcile_marker = redis.call(
    'HGET', reconcile_marker_key, session_id
)
local marker_phase, marker_generation_raw, marker_user = string.match(
    tostring(raw_reconcile_marker or ''),
    '^v1|([^|]+)|([^|]+)|[^|]*|[^|]*|[^|]*|[^|]*|(.*)$'
)
local marker_generation = tonumber(marker_generation_raw)
local membership_user = string.match(membership_value, '^v1|[^|]+|(.*)$')
local released_marker_matches = (
    marker_phase == 'released'
    and marker_generation ~= nil
    and generation ~= nil
    and marker_generation <= generation
    and (marker_user == '' or marker_user == membership_user)
)

local function admit_released_residue()
    -- A released marker proves the old reservation was already decremented.
    -- The surviving user hash field is residue, not a counted slot.
    if authoritative_repair ~= 1 then
        local other_user_count = redis.call('HLEN', user_key) - 1
        if other_user_count >= max_user then
            return 2
        end
        local sys_count = tonumber(redis.call('GET', sys_key) or '0')
        if sys_count >= max_sys then
            return 1
        end
    end
    redis.call('INCR', sys_key)
    redis.call('HSET', user_key, session_id, expires_at)
    extend_ttl(user_key, slot_ttl)
    redis.call('ZADD', bg_zset_key, expires_score, session_id)
    extend_ttl(bg_zset_key, slot_ttl)
    redis.call('HSET', generation_key, session_id, generation)
    extend_ttl(generation_key, slot_ttl)
    redis.call('HSET', system_members_key, session_id, membership_value)
    redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
    redis.call('HSET', hot_key, 'last_activity_at', activity_at)
    redis.call('EXPIRE', hot_key, 300)
    clear_reconcile_marker_through(generation)
    return 0
end

if redis.call('HEXISTS', user_key, session_id) == 1 then
    local current_generation = tonumber(redis.call('HGET', generation_key, session_id))
    local member_generation = membership_generation(redis.call('HGET', system_members_key, session_id))
    if current_generation == nil then
        if member_generation == nil and released_marker_matches then
            return admit_released_residue()
        end
        if authoritative_repair ~= 1 then
            return 6
        end
        if member_generation ~= nil and member_generation > generation then
            return 4
        end
        -- The user slot proves this legacy projection was already counted.
        -- Upgrade it atomically so no revoke/readmit gap can delete or exceed
        -- quota while a newer generation races this repair.
        redis.call('HSET', user_key, session_id, expires_at)
        extend_ttl(user_key, slot_ttl)
        redis.call('ZADD', bg_zset_key, expires_score, session_id)
        extend_ttl(bg_zset_key, slot_ttl)
        redis.call('HSET', generation_key, session_id, generation)
        extend_ttl(generation_key, slot_ttl)
        redis.call('HSET', system_members_key, session_id, membership_value)
        redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
        redis.call('HSET', hot_key, 'last_activity_at', activity_at)
        redis.call('EXPIRE', hot_key, 300)
        clear_reconcile_marker_through(generation)
        return 3
    end
    if generation < current_generation then
        return 4
    end
    if member_generation ~= nil and member_generation > generation then
        return 4
    end
    if member_generation == nil and released_marker_matches then
        return admit_released_residue()
    end
    if generation == current_generation then
        if member_generation == nil or member_generation < generation then
            if authoritative_repair ~= 1 then
                return 7
            end
            redis.call('HSET', system_members_key, session_id, membership_value)
        end
        clear_reconcile_marker_through(generation)
        return 3
    end
    -- generation > current_generation.  This is a replacement reservation.
    -- The existing user slot was already counted; update all generation
    -- projections together without incrementing the system counter.
    redis.call('HSET', user_key, session_id, expires_at)
    extend_ttl(user_key, slot_ttl)
    redis.call('ZADD', bg_zset_key, expires_score, session_id)
    extend_ttl(bg_zset_key, slot_ttl)
    redis.call('HSET', generation_key, session_id, generation)
    extend_ttl(generation_key, slot_ttl)
    redis.call('HSET', system_members_key, session_id, membership_value)
    redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
    redis.call('HSET', hot_key, 'last_activity_at', activity_at)
    redis.call('EXPIRE', hot_key, 300)
    clear_reconcile_marker_through(generation)
    return 5
end

local user_count = redis.call('HLEN', user_key)
if user_count >= max_user then
    return 2
end

local sys_count = tonumber(redis.call('GET', sys_key) or '0')
local member_generation = membership_generation(redis.call('HGET', system_members_key, session_id))
if member_generation ~= nil and member_generation > generation then
    return 4
end
if member_generation == nil and authoritative_repair == 1 then
    return 8
end
if member_generation == nil and sys_count >= max_sys and authoritative_repair ~= 1 then
    return 1
end

redis.call('HSET', user_key, session_id, expires_at)
extend_ttl(user_key, slot_ttl)
if member_generation == nil then
    redis.call('INCR', sys_key)
end
redis.call('HSET', system_members_key, session_id, membership_value)
redis.call('ZADD', bg_zset_key, expires_score, session_id)
extend_ttl(bg_zset_key, slot_ttl)
redis.call('HSET', generation_key, session_id, generation)
extend_ttl(generation_key, slot_ttl)
redis.call('HSETNX', hot_key, 'admitted_at', expires_at)
redis.call('HSET', hot_key, 'last_activity_at', activity_at)
redis.call('EXPIRE', hot_key, 300)
clear_reconcile_marker_through(generation)
return 0
"""


LUA_REVOKE_SOURCE = """
local sys_key = KEYS[1]
local user_key = KEYS[2]
local bg_zset_key = KEYS[3]
local generation_key = KEYS[4]
local system_members_key = KEYS[5]
local reconcile_marker_key = KEYS[6]
local reconcile_due_key = KEYS[7]
local session_id = ARGV[1]
local expected_generation = tonumber(ARGV[2])
local allow_legacy = tonumber(ARGV[3]) or 0
local expected_marker = ARGV[4] or ''
local released_marker = ARGV[5] or ''
local released_due = tonumber(ARGV[6])

local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end
if expected_generation == nil then
    return redis.error_reply('LUA_REVOKE: invalid expected generation')
end

local raw_marker = redis.call('HGET', reconcile_marker_key, session_id)
local marker_phase = nil
if raw_marker ~= false then
    marker_phase = string.match(tostring(raw_marker), '^v1|([^|]+)|')
end
if marker_phase == 'held' then
    if tostring(raw_marker) ~= expected_marker or released_marker == '' or released_due == nil then
        return -2
    end
end

local current_generation = redis.call('HGET', generation_key, session_id)
if current_generation == false then
    if allow_legacy ~= 1 then
        return 0
    end
elseif tonumber(current_generation) ~= expected_generation then
    return 0
end

local member_generation = redis.call('HGET', system_members_key, session_id)
if member_generation == false and current_generation ~= false and allow_legacy ~= 1 then
    return -1
end
if member_generation ~= false then
    local member_generation_number = membership_generation(member_generation)
    if member_generation_number == nil then
        return -1
    end
    if member_generation_number > expected_generation then
        return 0
    end
    if member_generation_number < expected_generation and allow_legacy ~= 1 then
        return -1
    end
end

local existed = redis.call('HDEL', user_key, session_id)
redis.call('HDEL', generation_key, session_id)
local member_existed = redis.call('HDEL', system_members_key, session_id)
redis.call('ZREM', bg_zset_key, session_id)
if member_existed == 1 or (marker_phase ~= 'released' and allow_legacy == 1 and existed == 1) then
    local sys_count = tonumber(redis.call('GET', sys_key) or '0')
    if sys_count > 0 then
        redis.call('DECR', sys_key)
    end
end
if marker_phase == 'held' and tostring(raw_marker) == expected_marker then
    redis.call('HSET', reconcile_marker_key, session_id, released_marker)
    redis.call('ZADD', reconcile_due_key, released_due, session_id)
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


LUA_SYNC_BACKGROUND_EXPIRY_SOURCE = """
local user_key = KEYS[1]
local bg_zset_key = KEYS[2]
local generation_key = KEYS[3]
local system_members_key = KEYS[4]
local reconcile_marker_key = KEYS[5]
local reconcile_due_key = KEYS[6]
local session_id = ARGV[1]
local expires_at = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local generation = tonumber(ARGV[4])
local membership_value = ARGV[5] or tostring(generation)
if expires_at == nil or ttl == nil or generation == nil then
    return redis.error_reply('LUA_SYNC_BACKGROUND_EXPIRY: invalid arguments')
end
local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end
local function extend_ttl(key, ttl)
    local current = redis.call('TTL', key)
    if current < ttl then
        redis.call('EXPIRE', key, ttl)
    end
end
local function clear_reconcile_marker_through(generation_limit)
    local raw = redis.call('HGET', reconcile_marker_key, session_id)
    if raw == false then
        return
    end
    local encoded = string.match(tostring(raw), '^v1|[^|]+|([^|]+)|')
    local marker_generation = tonumber(encoded)
    if marker_generation ~= nil and marker_generation <= generation_limit then
        redis.call('HDEL', reconcile_marker_key, session_id)
        redis.call('ZREM', reconcile_due_key, session_id)
    end
end
local current_generation = tonumber(redis.call('HGET', generation_key, session_id))
if current_generation ~= nil and current_generation > generation then
    return 0
end
local member_generation = membership_generation(redis.call('HGET', system_members_key, session_id))
if member_generation == nil then
    return -1
end
if member_generation > generation then
    return 0
end
if member_generation < generation then
    return -2
end
if current_generation ~= nil and current_generation < generation then
    return -2
end
redis.call('HSET', user_key, session_id, ARGV[2])
extend_ttl(user_key, ttl)
redis.call('ZADD', bg_zset_key, expires_at, session_id)
extend_ttl(bg_zset_key, ttl)
redis.call('HSET', generation_key, session_id, generation)
extend_ttl(generation_key, ttl)
redis.call('HSET', system_members_key, session_id, membership_value)
clear_reconcile_marker_through(generation)
return 1
"""


LUA_MARK_BACKGROUND_RECONCILE_HELD_SOURCE = """
local system_members_key = KEYS[1]
local marker_key = KEYS[2]
local marker_due_key = KEYS[3]
local session_id = ARGV[1]
local expected_generation = tonumber(ARGV[2])
local marker_value = ARGV[3]
local marker_due = tonumber(ARGV[4])

local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end

if expected_generation == nil or marker_due == nil then
    return redis.error_reply('LUA_MARK_BACKGROUND_RECONCILE_HELD: invalid arguments')
end
if membership_generation(redis.call('HGET', system_members_key, session_id)) ~= expected_generation then
    return 0
end
if redis.call('HEXISTS', marker_key, session_id) == 1 then
    return 2
end
redis.call('HSET', marker_key, session_id, marker_value)
redis.call('ZADD', marker_due_key, marker_due, session_id)
return 1
"""


LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SOURCE = """
local sys_key = KEYS[1]
local user_key = KEYS[2]
local bg_zset_key = KEYS[3]
local generation_key = KEYS[4]
local system_members_key = KEYS[5]
local marker_key = KEYS[6]
local marker_due_key = KEYS[7]
local session_id = ARGV[1]
local expected_generation = tonumber(ARGV[2])
local expected_marker = ARGV[3]
local released_marker = ARGV[4]
local released_due = tonumber(ARGV[5])
local full_revoke = tonumber(ARGV[6]) or 0
local require_generation = tonumber(ARGV[7]) or 0
local allow_missing_membership = tonumber(ARGV[8]) or 0

local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end

if expected_generation == nil or released_due == nil then
    return redis.error_reply('LUA_RELEASE_HELD_BACKGROUND_RECONCILE: invalid arguments')
end
if redis.call('HGET', marker_key, session_id) ~= expected_marker then
    return 0
end
local raw_membership = redis.call('HGET', system_members_key, session_id)
local member_generation = membership_generation(raw_membership)
if member_generation == nil then
    if allow_missing_membership ~= 1 then
        return -1
    end
elseif member_generation ~= expected_generation then
    return -2
end
if full_revoke == 1 then
    local current_generation = redis.call('HGET', generation_key, session_id)
    if current_generation == false then
        if require_generation == 1 then
            return -3
        end
    elseif tonumber(current_generation) ~= expected_generation then
        return -3
    end
end

if full_revoke == 1 then
    redis.call('HDEL', user_key, session_id)
    redis.call('ZREM', bg_zset_key, session_id)
    redis.call('HDEL', generation_key, session_id)
end
local member_existed = redis.call('HDEL', system_members_key, session_id)
if member_existed == 1 then
    local sys_count = tonumber(redis.call('GET', sys_key) or '0')
    if sys_count > 0 then
        redis.call('DECR', sys_key)
    end
end
redis.call('HSET', marker_key, session_id, released_marker)
redis.call('ZADD', marker_due_key, released_due, session_id)
return 1
"""


LUA_RESTORE_BACKGROUND_FROM_MARKER_SOURCE = """
local sys_key = KEYS[1]
local user_key = KEYS[2]
local bg_zset_key = KEYS[3]
local generation_key = KEYS[4]
local system_members_key = KEYS[5]
local marker_key = KEYS[6]
local marker_due_key = KEYS[7]
local session_id = ARGV[1]
local expires_at = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])
local generation = tonumber(ARGV[4])
local membership_value = ARGV[5]
local expected_marker = ARGV[6]

local function membership_generation(raw)
    if raw == false or raw == nil then
        return nil
    end
    local encoded = string.match(tostring(raw), '^v1|([^|]+)|')
    if encoded ~= nil then
        return tonumber(encoded)
    end
    return tonumber(raw)
end
local function extend_ttl(key, requested)
    local current = redis.call('TTL', key)
    if current < requested then
        redis.call('EXPIRE', key, requested)
    end
end

if expires_at == nil or ttl == nil or generation == nil then
    return redis.error_reply('LUA_RESTORE_BACKGROUND_FROM_MARKER: invalid arguments')
end
if redis.call('HGET', marker_key, session_id) ~= expected_marker then
    return 0
end
local marker_generation_raw = string.match(
    expected_marker, '^v1|[^|]+|([^|]+)|'
)
local marker_generation = tonumber(marker_generation_raw)
if marker_generation == nil or generation < marker_generation then
    return -1
end
local current_generation = tonumber(redis.call('HGET', generation_key, session_id))
if current_generation ~= nil and current_generation > generation then
    return -1
end
local raw_membership = redis.call('HGET', system_members_key, session_id)
local member_generation = membership_generation(raw_membership)
if member_generation ~= nil and member_generation > generation then
    return -1
end
local created_reservation = member_generation == nil
if created_reservation then
    redis.call('INCR', sys_key)
end
redis.call('HSET', user_key, session_id, ARGV[2])
extend_ttl(user_key, ttl)
redis.call('ZADD', bg_zset_key, expires_at, session_id)
extend_ttl(bg_zset_key, ttl)
redis.call('HSET', generation_key, session_id, generation)
extend_ttl(generation_key, ttl)
redis.call('HSET', system_members_key, session_id, membership_value)
redis.call('HDEL', marker_key, session_id)
redis.call('ZREM', marker_due_key, session_id)
if created_reservation then
    return 1
end
return 2
"""


LUA_GC_BACKGROUND_RECONCILE_MARKER_SOURCE = """
local marker_key = KEYS[1]
local marker_due_key = KEYS[2]
local session_id = ARGV[1]
local expected_marker = ARGV[2]
if redis.call('HGET', marker_key, session_id) ~= expected_marker then
    return 0
end
redis.call('HDEL', marker_key, session_id)
redis.call('ZREM', marker_due_key, session_id)
return 1
"""


LUA_ADMIT_SHA = hashlib.sha1(LUA_ADMIT_SOURCE.encode("utf-8")).hexdigest()
LUA_REVOKE_SHA = hashlib.sha1(LUA_REVOKE_SOURCE.encode("utf-8")).hexdigest()
LUA_SWEEP_EXPIRED_SHA = hashlib.sha1(
    LUA_SWEEP_EXPIRED_SOURCE.encode("utf-8")
).hexdigest()
LUA_SYNC_BACKGROUND_EXPIRY_SHA = hashlib.sha1(
    LUA_SYNC_BACKGROUND_EXPIRY_SOURCE.encode("utf-8")
).hexdigest()
LUA_MARK_BACKGROUND_RECONCILE_HELD_SHA = hashlib.sha1(
    LUA_MARK_BACKGROUND_RECONCILE_HELD_SOURCE.encode("utf-8")
).hexdigest()
LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SHA = hashlib.sha1(
    LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SOURCE.encode("utf-8")
).hexdigest()
LUA_RESTORE_BACKGROUND_FROM_MARKER_SHA = hashlib.sha1(
    LUA_RESTORE_BACKGROUND_FROM_MARKER_SOURCE.encode("utf-8")
).hexdigest()
LUA_GC_BACKGROUND_RECONCILE_MARKER_SHA = hashlib.sha1(
    LUA_GC_BACKGROUND_RECONCILE_MARKER_SOURCE.encode("utf-8")
).hexdigest()

# Short aliases for later supervisor code.
LUA_ADMIT = LUA_ADMIT_SOURCE
LUA_REVOKE = LUA_REVOKE_SOURCE
LUA_SWEEP_EXPIRED = LUA_SWEEP_EXPIRED_SOURCE
LUA_SYNC_BACKGROUND_EXPIRY = LUA_SYNC_BACKGROUND_EXPIRY_SOURCE
LUA_MARK_BACKGROUND_RECONCILE_HELD = LUA_MARK_BACKGROUND_RECONCILE_HELD_SOURCE
LUA_RELEASE_HELD_BACKGROUND_RECONCILE = (
    LUA_RELEASE_HELD_BACKGROUND_RECONCILE_SOURCE
)
LUA_RESTORE_BACKGROUND_FROM_MARKER = LUA_RESTORE_BACKGROUND_FROM_MARKER_SOURCE
LUA_GC_BACKGROUND_RECONCILE_MARKER = LUA_GC_BACKGROUND_RECONCILE_MARKER_SOURCE


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
