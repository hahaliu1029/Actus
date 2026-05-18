"""SkillSource — adapter for skill_tool calls. Recomputes risk
(Risk #1 hard rule) under a Redis NX single-flight lock with random
token + Lua compare-and-delete + loser poll for winner result +
45s failure cache.

Spec §3.1 SkillSource; Round 1 P1#7/P1#8; Round 2 P1#10; Round 3 P1#2/P1#3.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from typing import TYPE_CHECKING, ClassVar

from app.domain.services.permission.errors import PEInfrastructureUnavailable
from app.domain.services.permission.source_metadata import SkillCallMetadata
from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.sources.skill_metadata import (
    SkillRiskRefreshResult,
)
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel
from app.domain.services.skill_risk_assessor import SkillRiskAssessor

if TYPE_CHECKING:
    from redis.asyncio import Redis

    from app.domain.services.permission.skill_refresher import SkillRiskRefresher

logger = logging.getLogger(__name__)


# Lua: atomic compare-and-delete. Prevents TTL-expired lock being deleted
# by previous owner. Only the holder of the matching token can release.
_LUA_RELEASE_LOCK = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""


class SkillSource(PermissionSource):
    tool_source: ClassVar[str] = "skill"

    REFRESH_LOCK_TTL_SECONDS: ClassVar[int] = 10
    REFRESH_FAILURE_CACHE_TTL_SECONDS: ClassVar[int] = 45  # 30-60s window per Q2 B+
    LOSER_POLL_INTERVAL_MS: ClassVar[int] = 50
    LOSER_POLL_MAX_WAIT_MS: ClassVar[int] = 3000  # winner refresh budget

    def __init__(self, refresher: "SkillRiskRefresher", redis: "Redis"):
        self._refresher = refresher
        self._redis = redis

    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        meta = call.source_metadata
        if not isinstance(meta, SkillCallMetadata):
            raise ValueError(
                f"skill source requires SkillCallMetadata; got {type(meta).__name__}"
            )

        result = await self._refresh_with_singleflight(meta)

        # Q2 B+ rule: unknown / failed → force HIGH (defense in depth)
        if result.status == "fresh":
            effective_risk = meta.risk_level
        elif result.status == "refreshed":
            effective_risk = result.risk_level or RiskLevel.HIGH
        else:  # unknown | failed
            effective_risk = RiskLevel.HIGH

        # SkillRiskAssessor is stateless — instantiate cheap, no DI needed
        return SkillRiskAssessor().assess(
            tool_name=call.tool_name,
            tool_args=dict(call.tool_args),
            risk_level=effective_risk,
            runtime_type=meta.runtime_type,
            trust_origin=meta.trust_origin,
        )

    # ---------- singleflight ----------

    async def _refresh_with_singleflight(
        self, meta: SkillCallMetadata
    ) -> SkillRiskRefreshResult:
        hash_part = meta.content_hash or "none"
        lock_key = f"pe:skill:refresh_lock:{meta.skill_id}:{hash_part}"
        fail_key = f"pe:skill:refresh_failed:{meta.skill_id}:{hash_part}"
        result_key = f"pe:skill:refresh_result:{meta.skill_id}:{hash_part}"
        lock_token = secrets.token_hex(16)

        # 1. Quick check: was the previous refresh recently failed?
        try:
            cached_fail = await self._redis.get(fail_key)
        except Exception as exc:
            raise PEInfrastructureUnavailable(
                f"redis_get_fail_key: {exc}"
            ) from exc

        if cached_fail:
            return SkillRiskRefreshResult(
                status="failed", error="cached_recent_failure"
            )

        # 2. Try to acquire singleflight lock
        try:
            acquired = await self._redis.set(
                lock_key, lock_token,
                nx=True, ex=self.REFRESH_LOCK_TTL_SECONDS,
            )
        except Exception as exc:
            raise PEInfrastructureUnavailable(
                f"redis_set_lock: {exc}"
            ) from exc

        if not acquired:
            return await self._poll_for_winner_result(result_key, fail_key)

        # 3. We are the winner — run refresher and publish result
        try:
            result = await self._refresher.refresh(meta.tool_name, meta.content_hash)
            await self._cache_result(result, result_key, fail_key)
            return result
        finally:
            # Lua compare-and-delete: only delete if our token still owns the lock.
            # Best-effort; TTL fallback ensures the lock can't pin forever.
            try:
                await self._redis.eval(
                    _LUA_RELEASE_LOCK, 1, lock_key, lock_token,
                )
            except Exception as exc:
                logger.warning(
                    "skill_source: lua lock release best-effort failed for %s: %r",
                    lock_key, exc,
                )

    async def _poll_for_winner_result(
        self, result_key: str, fail_key: str,
    ) -> SkillRiskRefreshResult:
        deadline_ms = self.LOSER_POLL_MAX_WAIT_MS
        elapsed_ms = 0
        while elapsed_ms < deadline_ms:
            try:
                cached = await self._redis.get(result_key)
                if cached:
                    return SkillRiskRefreshResult.from_redis_value(cached)
                cached_fail = await self._redis.get(fail_key)
                if cached_fail:
                    return SkillRiskRefreshResult(
                        status="failed",
                        error="winner_published_failure",
                    )
            except Exception as exc:
                # PE-1 §5.1: loser poll is part of the read path → fail-fast.
                # Surfaces as 503 in HTTP preflight + agent retry chain handles
                # transient outages. Cache publish remains the only best-effort
                # Redis op (_cache_result).
                raise PEInfrastructureUnavailable(
                    f"redis_loser_poll: {exc}"
                ) from exc
            await asyncio.sleep(self.LOSER_POLL_INTERVAL_MS / 1000)
            elapsed_ms += self.LOSER_POLL_INTERVAL_MS
        return SkillRiskRefreshResult(status="failed", error="winner_timeout")

    async def _cache_result(
        self,
        result: SkillRiskRefreshResult,
        result_key: str,
        fail_key: str,
    ) -> None:
        """Best-effort cache publish (Round 2 P1#9 / Round 3 P1#3).

        Winner has already computed the result and SkillSource will return
        it regardless. If Redis is unreachable here, losers will surface
        PEInfrastructureUnavailable from their own _poll_for_winner_result
        read attempt (fail-fast per §5.1), which routes to 503 in HTTP
        preflight + agent retry chain.

        Read path (initial check + lock acquire + loser poll) IS fail-fast
        (raises PEInfrastructureUnavailable). Cache publish is the only
        Redis op where we tolerate failure — and only because the
        alternative (raising) would discard the winner's just-computed
        result.

        fresh / refreshed → short result-key TTL only.
        unknown / failed  → also write fail_key so future calls in the
                            TTL window skip the refresher entirely.
        """
        try:
            await self._redis.set(result_key, result.to_redis_value(), ex=5)
            if result.status in {"failed", "unknown"}:
                await self._redis.set(
                    fail_key, "1",
                    ex=self.REFRESH_FAILURE_CACHE_TTL_SECONDS,
                )
        except Exception as exc:
            logger.warning(
                "skill_source: cache publish best-effort failed "
                "(winner result still returned): %r",
                exc,
            )
