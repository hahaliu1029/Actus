from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from redis.asyncio import Redis

ZSET_KEY = "confirmation_deadlines"
SWEEP_LOCK_KEY = "confirmation_sweep_lock"


@dataclass
class ConfirmationDetail:
    session_id: str
    tool_call_id: str
    user_id: str
    tool_name: str
    tool_args: dict[str, Any]
    risk_level: str
    arg_digest: str
    primary_arg: str
    dir_arg: str | None
    matched_patterns: list[str]
    deadline_ts: float
    status: str = "pending"


class ConfirmationManager:
    """Manages confirmation deadlines via Redis Sorted Set + Hash.
    State machine: PENDING -> PROCESSING -> CLEANED"""

    def __init__(self, redis: "Redis", timeout_seconds: int = 300):
        self._redis = redis
        self._timeout_seconds = timeout_seconds

    def _hash_key(self, session_id: str, tool_call_id: str) -> str:
        return f"confirmation_detail:{session_id}:{tool_call_id}"

    def _member(self, session_id: str, tool_call_id: str) -> str:
        return f"{session_id}:{tool_call_id}"

    async def store(self, detail: ConfirmationDetail) -> None:
        member = self._member(detail.session_id, detail.tool_call_id)
        await self._redis.zadd(ZSET_KEY, {member: detail.deadline_ts})
        await self._redis.hset(self._hash_key(detail.session_id, detail.tool_call_id), mapping={
            "session_id": detail.session_id,
            "tool_call_id": detail.tool_call_id,
            "user_id": detail.user_id,
            "tool_name": detail.tool_name,
            "tool_args_json": json.dumps(detail.tool_args, default=str),
            "risk_level": detail.risk_level,
            "arg_digest": detail.arg_digest,
            "primary_arg": detail.primary_arg,
            "dir_arg": detail.dir_arg or "",
            "matched_patterns_json": json.dumps(detail.matched_patterns),
            "deadline_ts": str(detail.deadline_ts),
            "status": "pending",
        })

    async def read(self, session_id: str, tool_call_id: str) -> ConfirmationDetail | None:
        data = await self._redis.hgetall(self._hash_key(session_id, tool_call_id))
        if not data:
            return None
        return ConfirmationDetail(
            session_id=data["session_id"],
            tool_call_id=data["tool_call_id"],
            user_id=data["user_id"],
            tool_name=data["tool_name"],
            tool_args=json.loads(data["tool_args_json"]),
            risk_level=data["risk_level"],
            arg_digest=data["arg_digest"],
            primary_arg=data["primary_arg"],
            dir_arg=data["dir_arg"] or None,
            matched_patterns=json.loads(data["matched_patterns_json"]),
            deadline_ts=float(data["deadline_ts"]),
            status=data.get("status", "pending"),
        )

    async def mark_processing(self, session_id: str, tool_call_id: str) -> None:
        """Unconditionally set status=processing (legacy; no CAS semantics).

        **Warning**: 无 CAS 保证。R5b-3 新代码请用 ``mark_processing_if_pending``
        做原子 single-flight claim；此方法保留仅供向后兼容 / sweep 路径使用。
        """
        await self._redis.hset(self._hash_key(session_id, tool_call_id), "status", "processing")

    # R5b-3 fix (Codex CRIT): Redis Lua 原子 CAS——仅当 status=='pending' 时改写
    # 成 'processing'。用于 scope='once' 的 single-flight claim：两个并发 /resume
    # 只有一个能拿到 True，另一个拿 False → 上层返 409。
    # KEYS[1] = hash_key；ARGV[1] = new_status；ARGV[2] = expected_status
    # 返 1 表示 CAS 成功；0 表示 expected 不匹配或 key 不存在。
    _MARK_PROCESSING_IF_PENDING_LUA = (
        "local s = redis.call('HGET', KEYS[1], 'status') "
        "if s == ARGV[2] then "
        "    redis.call('HSET', KEYS[1], 'status', ARGV[1]) "
        "    return 1 "
        "end "
        "return 0"
    )

    async def mark_processing_if_pending(
        self, session_id: str, tool_call_id: str
    ) -> bool:
        """Atomic CAS: status='pending' → 'processing'. 返 True 表示本调用赢得了
        single-flight claim；False 表示已被其他请求/worker 占用（or 不存在）。

        scope='once' 的并发 /resume 防重依赖此方法；I1/I4 的 race invariant 由
        这里的 Redis Lua 脚本原子性保证。
        """
        result = await self._redis.eval(
            self._MARK_PROCESSING_IF_PENDING_LUA,
            1,
            self._hash_key(session_id, tool_call_id),
            "processing",
            "pending",
        )
        return bool(int(result or 0))

    async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
        """Roll back from processing to pending (on resume failure)."""
        await self._redis.hset(self._hash_key(session_id, tool_call_id), "status", "pending")

    async def cleanup(self, session_id: str, tool_call_id: str) -> None:
        await self._redis.zrem(ZSET_KEY, self._member(session_id, tool_call_id))
        await self._redis.delete(self._hash_key(session_id, tool_call_id))

    async def has_pending_for_session(self, session_id: str) -> bool:
        """Check if any pending (non-expired) confirmation exists for a session."""
        members = await self._redis.zrangebyscore(ZSET_KEY, "-inf", "+inf")
        for member in members:
            if member.startswith(f"{session_id}:"):
                parts = member.split(":", 1)
                if len(parts) == 2:
                    detail = await self.read(parts[0], parts[1])
                    if detail and detail.status == "pending":
                        return True
        return False

    async def find_expired(self) -> list[ConfirmationDetail]:
        now = time.time()
        members = await self._redis.zrangebyscore(ZSET_KEY, "-inf", now)
        results = []
        for member in members:
            parts = member.split(":", 1)
            if len(parts) != 2:
                continue
            detail = await self.read(parts[0], parts[1])
            if detail and detail.status != "processing":
                results.append(detail)
        return results

    async def acquire_sweep_lock(self, worker_id: str, ttl: int = 30) -> bool:
        return bool(await self._redis.set(SWEEP_LOCK_KEY, worker_id, nx=True, ex=ttl))
