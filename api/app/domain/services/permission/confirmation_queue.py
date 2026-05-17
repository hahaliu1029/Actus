from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

logger = logging.getLogger(__name__)

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
    claim_nonce: str | None = None              # PE-0: 16-byte hex set by
                                                # mark_processing_if_pending
                                                # when called with claim_nonce
    processing_started_at: datetime | None = None  # PE-0: timestamp used by
                                                    # find_orphaned_processing
                                                    # to rescue stale processing
    # Round 36 P1#1: SSM mode_revision captured AT preflight_resume time.
    # commit_resume compares this against the current SSM revision to catch
    # the RUNNING -> TAKEOVER (rev=Y) -> RUNNING (rev=Z) round-trip race
    # window between preflight (sees rev=X) and the EvaluationContext read
    # used by the existing round-34 P1#1 check (which only carries the
    # post-interrupt revision, NOT the preflight one).
    session_mode_revision_at_preflight: int | None = None


# PE-0 extended Lua CAS: atomically writes status + claim_nonce + processing_started_at
# KEYS[1] = hash_key
# ARGV[1] = new_status ("processing")
# ARGV[2] = expected_status ("pending")
# ARGV[3] = claim_nonce (empty string if absent)
# ARGV[4] = processing_started_at ISO string (empty string if absent)
_CAS_LUA = (
    "local s = redis.call('HGET', KEYS[1], 'status') "
    "if s == ARGV[2] then "
    "    redis.call('HSET', KEYS[1], 'status', ARGV[1], "
    "                                'claim_nonce', ARGV[3], "
    "                                'processing_started_at', ARGV[4]) "
    "    return 1 "
    "else "
    "    return 0 "
    "end"
)


class ConfirmationQueue:
    """Per-(session_id, tool_call_id) approval claim queue (Redis ZSET + Hash).

    Renamed from ConfirmationManager (PE-0). Semantics: PE-internal queue;
    public callers should go through PermissionEngine.preflight_resume /
    commit_resume, NOT call this directly. Sweeper background path still
    calls this directly via acquire_sweep_lock + find_expired +
    find_orphaned_processing.
    """

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
        await self._redis.hset(
            self._hash_key(detail.session_id, detail.tool_call_id),
            mapping={
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
                "status": detail.status,
                # PE-0 additive — empty string means absent
                "claim_nonce": detail.claim_nonce or "",
                "processing_started_at": (
                    detail.processing_started_at.isoformat()
                    if detail.processing_started_at is not None else ""
                ),
                # Round 36 P1#1 additive — empty string means absent
                "session_mode_revision": (
                    str(detail.session_mode_revision_at_preflight)
                    if detail.session_mode_revision_at_preflight is not None
                    else ""
                ),
            },
        )

    async def read(self, session_id: str, tool_call_id: str) -> ConfirmationDetail | None:
        data = await self._redis.hgetall(self._hash_key(session_id, tool_call_id))
        if not data:
            return None
        claim_raw = data.get("claim_nonce", "")
        proc_raw = data.get("processing_started_at", "")
        processing_started_at: datetime | None = None
        if proc_raw:
            try:
                processing_started_at = datetime.fromisoformat(proc_raw)
            except ValueError:
                logger.warning(
                    "ConfirmationQueue.read: corrupt processing_started_at %r for %s:%s",
                    proc_raw, data.get("session_id"), data.get("tool_call_id"),
                )
                # leave processing_started_at as None so the sweeper treats it as
                # legacy-pending rather than crashing.
        # Issue 3: normalize naive datetimes to UTC so find_orphaned_processing
        # comparison with datetime.now(timezone.utc) never raises TypeError.
        if processing_started_at is not None and processing_started_at.tzinfo is None:
            processing_started_at = processing_started_at.replace(tzinfo=timezone.utc)
        # Round 36 P1#1: hydrate session_mode_revision_at_preflight (written by
        # store() and/or set_preflight_mode_revision() below). Defensive int()
        # — if a stale/corrupt entry has a non-int value, treat as absent so
        # commit_resume falls back to the existing ctx-based revision check.
        mode_rev_raw = data.get("session_mode_revision", "")
        session_mode_revision_at_preflight: int | None = None
        if mode_rev_raw:
            try:
                session_mode_revision_at_preflight = int(mode_rev_raw)
            except (TypeError, ValueError):
                logger.warning(
                    "ConfirmationQueue.read: corrupt session_mode_revision %r for %s:%s",
                    mode_rev_raw, data.get("session_id"), data.get("tool_call_id"),
                )
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
            claim_nonce=claim_raw or None,
            processing_started_at=processing_started_at,
            session_mode_revision_at_preflight=session_mode_revision_at_preflight,
        )

    async def mark_processing(self, session_id: str, tool_call_id: str) -> None:
        """Unconditionally set status=processing (legacy; no CAS semantics).

        **Warning**: 无 CAS 保证。R5b-3 新代码请用 ``mark_processing_if_pending``
        做原子 single-flight claim；此方法保留仅供向后兼容 / sweep 路径使用。
        """
        await self._redis.hset(self._hash_key(session_id, tool_call_id), "status", "processing")

    async def mark_processing_if_pending(
        self,
        session_id: str,
        tool_call_id: str,
        *,
        claim_nonce: str | None = None,
        processing_started_at: datetime | None = None,
    ) -> bool:
        """Atomic CAS: status='pending' → 'processing'. 返 True 表示本调用赢得了
        single-flight claim；False 表示已被其他请求/worker 占用（or 不存在）。

        If claim_nonce is provided, it is written atomically in the same
        Redis EVAL so commit_resume can later prove ownership of the claim.
        Loser sees this method return False (status was already processing
        / expired / missing).

        scope='once' 的并发 /resume 防重依赖此方法；I1/I4 的 race invariant 由
        这里的 Redis Lua 脚本原子性保证。
        """
        nonce_arg = claim_nonce or ""
        proc_arg = (
            processing_started_at.isoformat()
            if processing_started_at is not None else ""
        )
        res = await self._redis.eval(
            _CAS_LUA, 1,
            self._hash_key(session_id, tool_call_id),
            "processing", "pending",
            nonce_arg, proc_arg,
        )
        return bool(int(res or 0))

    async def mark_pending(self, session_id: str, tool_call_id: str) -> None:
        """Roll back from processing to pending (on resume failure); clears claim fields.

        Round 36 P1#1: also clears session_mode_revision so the next preflight
        rewrites it (the preflight rev belongs to the lost claim, not the
        future one).
        """
        await self._redis.hset(
            self._hash_key(session_id, tool_call_id),
            mapping={
                "status": "pending",
                "claim_nonce": "",
                "processing_started_at": "",
                "session_mode_revision": "",
            },
        )

    async def set_preflight_mode_revision(
        self,
        session_id: str,
        tool_call_id: str,
        mode_revision: int,
    ) -> None:
        """Round 36 P1#1: write session_mode_revision separately after the CAS
        in ``mark_processing_if_pending`` has succeeded.

        Single-field HSET, not part of the Lua CAS — that script already locks
        ``status`` + ``claim_nonce`` atomically, and the reader (commit_resume)
        only acts on this value after validating the claim_nonce, so a partial
        write is safe: a losing CAS never reaches this method, and a winning
        CAS owns the field exclusively until cleanup/mark_pending wipes it.
        """
        await self._redis.hset(
            self._hash_key(session_id, tool_call_id),
            "session_mode_revision",
            str(mode_revision),
        )

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

    async def find_orphaned_processing(
        self,
        processing_age_threshold_seconds: int = 60,
    ) -> list[ConfirmationDetail]:
        """Return entries where status='processing' and
        processing_started_at < now - threshold.

        These represent claims where preflight succeeded but commit_resume
        never ran (e.g. worker crash between HTTP response and graph
        resume). Sweeper rescues them by either:
          - mark_pending (re-open claim) when the original session is still
            in RUNNING/WAITING mode, OR
          - cleanup + audit deny when expired by deadline_ts as well.
        Both decisions live in the sweeper caller (Phase 11+), not here.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=processing_age_threshold_seconds,
        )
        out: list[ConfirmationDetail] = []
        # ZSET key holds (session_id:tool_call_id) members; we iterate full
        # set rather than re-keying by status because volume is small
        # (capped by deadline_ts ZSET-driven cleanup).
        members = await self._redis.zrange(ZSET_KEY, 0, -1)
        for m in members:
            member_str = m.decode() if isinstance(m, bytes) else m
            sid, tcid = member_str.split(":", 1)
            d = await self.read(sid, tcid)
            if d is None:
                continue
            if d.status != "processing":
                continue
            if d.processing_started_at is None:
                continue
            if d.processing_started_at < cutoff:
                out.append(d)
        return out

    async def acquire_sweep_lock(self, worker_id: str, ttl: int = 30) -> bool:
        return bool(await self._redis.set(SWEEP_LOCK_KEY, worker_id, nx=True, ex=ttl))
