"""PE-1 §3.1 — SkillSource + Redis single-flight matrix.

Coverage:
- assess_risk hash status branching (fresh / refreshed / unknown / failed)
- Caller-prefilled risk override (Risk #1 hard rule)
- Redis NX lock (winner runs refresher; loser polls)
- Lua compare-and-delete (only own token deletes)
- Loser poll observes winner fresh / refreshed / failed / timeout
- Failure cache TTL (45s) + cache-hit skips refresher call
- Redis exceptions raise PEInfrastructureUnavailable
- Cache publish is best-effort (Redis write failure does NOT propagate)
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.permission.errors import PEInfrastructureUnavailable
from app.domain.services.permission.source_metadata import SkillCallMetadata
from app.domain.services.permission.sources.skill_metadata import (
    SkillRiskRefreshResult,
)
from app.domain.services.permission.sources.skill_source import SkillSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_meta(*, risk_level: RiskLevel = RiskLevel.MEDIUM,
               content_hash: str | None = "sha256:hash1",
               skill_id: str = "sk_test") -> SkillCallMetadata:
    return SkillCallMetadata(
        tool_name="myskill_run",
        skill_id=skill_id,
        content_hash=content_hash,
        risk_level=risk_level,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )


def _make_call(meta: SkillCallMetadata, *,
               caller_risk: RiskAssessment | None = None) -> ToolCallSpec:
    return ToolCallSpec(
        tool_name=meta.tool_name,
        tool_args={"x": "y"},
        tool_source="skill",
        user_id="u",
        session_id="s",
        primary_arg="x",
        dir_arg=None,
        arg_digest="d1",
        risk_assessment=caller_risk,
        tool_call_id="tc1",
        source_metadata=meta,
    )


class _FakeRedis:
    """Minimal Redis stub with explicit per-method overrides for failure injection."""

    def __init__(self):
        self.store: dict[str, bytes | str] = {}
        # When set, calling the matching method raises this exception
        self.get_exc: Exception | None = None
        self.set_exc: Exception | None = None
        self.eval_exc: Exception | None = None
        # NX simulation: pretend lock was already acquired
        self.nx_already_held: bool = False
        self.set_calls: list[tuple[str, Any, dict]] = []
        self.eval_calls: list[tuple[str, int, tuple]] = []

    async def get(self, key: str):
        if self.get_exc is not None:
            raise self.get_exc
        return self.store.get(key)

    async def set(self, key: str, value: Any, **kwargs):
        self.set_calls.append((key, value, kwargs))
        if self.set_exc is not None:
            raise self.set_exc
        if kwargs.get("nx") and (self.nx_already_held or key in self.store):
            return None
        # Non-NX: write
        if not kwargs.get("nx"):
            self.store[key] = value
        else:
            self.store[key] = value
        return True

    async def eval(self, script: str, numkeys: int, *args):
        self.eval_calls.append((script, numkeys, args))
        if self.eval_exc is not None:
            raise self.eval_exc
        # Lua compare-and-delete simulation: keys[0] = lock key, args[0] = expected token
        if numkeys == 1:
            key = args[0]
            expected_token = args[1] if len(args) > 1 else None
            existing = self.store.get(key)
            if existing == expected_token:
                del self.store[key]
                return 1
            return 0
        return 0


class _FakeRefresher:
    def __init__(self, *, result: SkillRiskRefreshResult | None = None,
                 exc: Exception | None = None):
        self._result = result
        self._exc = exc
        self.calls: list[tuple[str, str | None]] = []

    async def refresh(self, tool_name: str,
                      expected_content_hash: str | None) -> SkillRiskRefreshResult:
        self.calls.append((tool_name, expected_content_hash))
        if self._exc is not None:
            raise self._exc
        return self._result or SkillRiskRefreshResult(status="fresh")


# ---------- Construction ----------

class TestSkillSourceTypeAndCtor:
    def test_tool_source(self):
        assert SkillSource.tool_source == "skill"

    def test_constants(self):
        assert SkillSource.REFRESH_LOCK_TTL_SECONDS == 10
        assert SkillSource.REFRESH_FAILURE_CACHE_TTL_SECONDS == 45
        assert SkillSource.LOSER_POLL_INTERVAL_MS == 50
        assert SkillSource.LOSER_POLL_MAX_WAIT_MS == 3000


# ---------- Status branching ----------

class TestAssessRiskStatusBranching:
    async def test_fresh_status_uses_meta_risk_level(self):
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        meta = _make_meta(risk_level=RiskLevel.LOW)
        out = await src.assess_risk(_make_call(meta))
        assert out.final_level == RiskLevel.LOW

    async def test_refreshed_status_uses_new_risk_level(self):
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="refreshed", risk_level=RiskLevel.HIGH)
        )
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        meta = _make_meta(risk_level=RiskLevel.LOW)  # caller meta says LOW
        out = await src.assess_risk(_make_call(meta))
        assert out.final_level == RiskLevel.HIGH

    async def test_refreshed_status_with_none_risk_falls_back_to_high(self):
        """Defense-in-depth: refresher said 'refreshed' but forgot risk_level."""
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="refreshed", risk_level=None)
        )
        src = SkillSource(refresher=refresher, redis=_FakeRedis())
        out = await src.assess_risk(_make_call(_make_meta()))
        assert out.final_level == RiskLevel.HIGH

    async def test_unknown_forces_high(self):
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="unknown", error="skill_dir_missing")
        )
        src = SkillSource(refresher=refresher, redis=_FakeRedis())
        out = await src.assess_risk(_make_call(_make_meta()))
        assert out.final_level == RiskLevel.HIGH

    async def test_failed_forces_high(self):
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="failed", error="OSError")
        )
        src = SkillSource(refresher=refresher, redis=_FakeRedis())
        out = await src.assess_risk(_make_call(_make_meta()))
        assert out.final_level == RiskLevel.HIGH


# ---------- Risk #1 hard rule: caller pre-fill ignored ----------

class TestCallerPrefilledRiskIgnored:
    async def test_caller_low_meta_high_returns_high(self):
        """If caller pre-filled risk_assessment=LOW but binding says HIGH,
        SkillSource MUST return HIGH (Risk #1)."""
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        src = SkillSource(refresher=refresher, redis=_FakeRedis())

        caller_low = RiskAssessment(
            tool_name="myskill_run", tool_args={"x": "y"},
            static_level=RiskLevel.LOW, dynamic_level=RiskLevel.NONE,
            final_level=RiskLevel.LOW, risk_reason="caller said low",
            matched_patterns=[], suggested_alternative=None,
            primary_arg="x", dir_arg=None, arg_digest="d1",
        )
        meta = _make_meta(risk_level=RiskLevel.HIGH)
        out = await src.assess_risk(_make_call(meta, caller_risk=caller_low))
        assert out.final_level == RiskLevel.HIGH


# ---------- Source metadata validation ----------

class TestSourceMetadataValidation:
    async def test_raises_when_source_metadata_missing(self):
        src = SkillSource(refresher=_FakeRefresher(), redis=_FakeRedis())
        call = ToolCallSpec(
            tool_name="t", tool_args={}, tool_source="skill",
            user_id="u", session_id="s", tool_call_id="tc",
            source_metadata=None,
        )
        with pytest.raises(ValueError, match="SkillCallMetadata"):
            await src.assess_risk(call)


# ---------- Singleflight: winner / loser ----------

class TestSingleflightWinnerAcquiresLock:
    async def test_winner_acquires_lock_and_calls_refresher(self):
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        assert refresher.calls == [("myskill_run", "sha256:hash1")]
        # set called with nx=True
        nx_calls = [c for c in redis.set_calls if c[2].get("nx") is True]
        assert len(nx_calls) == 1
        assert nx_calls[0][0].startswith("pe:skill:refresh_lock:sk_test:")

    async def test_winner_lock_key_includes_skill_id_and_hash(self):
        refresher = _FakeRefresher()
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(
            _make_meta(skill_id="sk_xyz", content_hash="sha256:abc")
        ))
        lock_call = [c for c in redis.set_calls if c[2].get("nx") is True][0]
        assert lock_call[0] == "pe:skill:refresh_lock:sk_xyz:sha256:abc"

    async def test_winner_lock_uses_random_token_and_ttl(self):
        refresher = _FakeRefresher()
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        lock_call = [c for c in redis.set_calls if c[2].get("nx") is True][0]
        # Token = secrets.token_hex(16) → 32-char hex
        assert isinstance(lock_call[1], str)
        assert len(lock_call[1]) == 32
        assert lock_call[2].get("ex") == SkillSource.REFRESH_LOCK_TTL_SECONDS


class TestSingleflightLuaRelease:
    async def test_lua_eval_uses_only_own_token(self):
        refresher = _FakeRefresher()
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        assert len(redis.eval_calls) == 1
        script, numkeys, args = redis.eval_calls[0]
        assert "redis.call" in script
        assert "del" in script.lower()
        assert numkeys == 1
        # args = (lock_key, lock_token)
        assert args[0].startswith("pe:skill:refresh_lock:")
        assert isinstance(args[1], str) and len(args[1]) == 32

    async def test_lua_eval_failure_swallowed_best_effort(self):
        refresher = _FakeRefresher()
        redis = _FakeRedis()
        redis.eval_exc = RuntimeError("lua disabled")
        src = SkillSource(refresher=refresher, redis=redis)
        # Should NOT raise even though eval failed
        out = await src.assess_risk(_make_call(_make_meta()))
        assert out is not None


class TestSingleflightLoserPolls:
    async def test_loser_observes_winner_fresh(self):
        redis = _FakeRedis()
        redis.nx_already_held = True
        winner_result = SkillRiskRefreshResult(status="fresh")
        redis.store["pe:skill:refresh_result:sk_test:sha256:hash1"] = (
            winner_result.to_redis_value()
        )
        refresher = _FakeRefresher()
        src = SkillSource(refresher=refresher, redis=redis)
        meta = _make_meta(risk_level=RiskLevel.MEDIUM)
        out = await src.assess_risk(_make_call(meta))
        # Loser did NOT call refresher
        assert refresher.calls == []
        assert out.final_level == RiskLevel.MEDIUM  # from fresh + meta

    async def test_loser_observes_winner_refreshed_high(self):
        redis = _FakeRedis()
        redis.nx_already_held = True
        winner_result = SkillRiskRefreshResult(
            status="refreshed", risk_level=RiskLevel.HIGH
        )
        redis.store["pe:skill:refresh_result:sk_test:sha256:hash1"] = (
            winner_result.to_redis_value()
        )
        refresher = _FakeRefresher()
        src = SkillSource(refresher=refresher, redis=redis)
        out = await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))
        assert out.final_level == RiskLevel.HIGH

    async def test_loser_observes_winner_failure_via_fail_key(self):
        redis = _FakeRedis()
        redis.nx_already_held = True
        redis.store["pe:skill:refresh_failed:sk_test:sha256:hash1"] = "1"
        refresher = _FakeRefresher()
        src = SkillSource(refresher=refresher, redis=redis)
        out = await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))
        assert out.final_level == RiskLevel.HIGH
        assert refresher.calls == []

    async def test_loser_timeout_returns_failed_forces_high(self, monkeypatch):
        """Loser polls but neither result_key nor fail_key appear within
        LOSER_POLL_MAX_WAIT_MS → force-HIGH."""
        redis = _FakeRedis()
        redis.nx_already_held = True

        # Speed up the test by patching poll constants
        monkeypatch.setattr(SkillSource, "LOSER_POLL_INTERVAL_MS", 10)
        monkeypatch.setattr(SkillSource, "LOSER_POLL_MAX_WAIT_MS", 50)

        refresher = _FakeRefresher()
        src = SkillSource(refresher=refresher, redis=redis)
        out = await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))
        assert out.final_level == RiskLevel.HIGH

    async def test_loser_poll_redis_read_failure_raises_infra_unavailable(self, monkeypatch):
        """PE-1 §5.1: loser poll Redis read failure is fail-fast (read path).
        Was originally drafted as force-HIGH; corrected post-codex review."""
        redis = _FakeRedis()
        redis.nx_already_held = True
        original_get = redis.get
        call_count = {"n": 0}

        async def flaky_get(key: str):
            call_count["n"] += 1
            if call_count["n"] > 1:
                raise ConnectionError("redis down")
            return await original_get(key)

        redis.get = flaky_get  # type: ignore[method-assign]
        monkeypatch.setattr(SkillSource, "LOSER_POLL_INTERVAL_MS", 5)
        monkeypatch.setattr(SkillSource, "LOSER_POLL_MAX_WAIT_MS", 50)

        src = SkillSource(refresher=_FakeRefresher(), redis=redis)
        with pytest.raises(PEInfrastructureUnavailable, match="redis_loser_poll"):
            await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))


# ---------- Failure cache ----------

class TestFailureCache:
    async def test_cached_failure_skips_refresh_call(self):
        redis = _FakeRedis()
        redis.store["pe:skill:refresh_failed:sk_test:sha256:hash1"] = "1"
        refresher = _FakeRefresher()
        src = SkillSource(refresher=refresher, redis=redis)
        out = await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))
        assert refresher.calls == []  # no refresher call
        assert out.final_level == RiskLevel.HIGH

    async def test_failed_result_writes_failure_cache_with_ttl(self):
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="failed", error="OSError")
        )
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        # Find the fail_key set with ex=REFRESH_FAILURE_CACHE_TTL_SECONDS
        fail_writes = [
            c for c in redis.set_calls
            if c[0].startswith("pe:skill:refresh_failed:")
            and c[2].get("ex") == SkillSource.REFRESH_FAILURE_CACHE_TTL_SECONDS
        ]
        assert len(fail_writes) == 1

    async def test_unknown_result_writes_failure_cache(self):
        refresher = _FakeRefresher(
            result=SkillRiskRefreshResult(status="unknown", error="no_bundle")
        )
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        fail_writes = [
            c for c in redis.set_calls
            if c[0].startswith("pe:skill:refresh_failed:")
        ]
        assert len(fail_writes) == 1

    async def test_fresh_result_does_not_write_failure_cache(self):
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        fail_writes = [
            c for c in redis.set_calls
            if c[0].startswith("pe:skill:refresh_failed:")
        ]
        assert fail_writes == []

    async def test_result_published_for_loser_poll(self):
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        redis = _FakeRedis()
        src = SkillSource(refresher=refresher, redis=redis)
        await src.assess_risk(_make_call(_make_meta()))
        result_writes = [
            c for c in redis.set_calls
            if c[0].startswith("pe:skill:refresh_result:")
        ]
        assert len(result_writes) == 1
        # 5s TTL
        assert result_writes[0][2].get("ex") == 5


# ---------- Redis exceptions → PEInfrastructureUnavailable ----------

class TestRedisExceptionsRaiseInfra:
    async def test_redis_get_fail_key_exception_raises(self):
        redis = _FakeRedis()
        redis.get_exc = ConnectionError("network unreachable")
        src = SkillSource(refresher=_FakeRefresher(), redis=redis)
        with pytest.raises(PEInfrastructureUnavailable, match="redis_get_fail_key"):
            await src.assess_risk(_make_call(_make_meta()))

    async def test_redis_set_lock_exception_raises(self):
        # First get succeeds (returns None), then set raises
        redis = _FakeRedis()
        redis.set_exc = ConnectionError("write fail")
        src = SkillSource(refresher=_FakeRefresher(), redis=redis)
        with pytest.raises(PEInfrastructureUnavailable, match="redis_set_lock"):
            await src.assess_risk(_make_call(_make_meta()))


# ---------- Best-effort cache publish ----------

class TestCachePublishBestEffort:
    async def test_cache_set_failure_does_not_propagate(self):
        """After winner has refresher result, cache publish failure must NOT
        raise — caller still receives the winner's result."""
        refresher = _FakeRefresher(result=SkillRiskRefreshResult(status="fresh"))
        redis = _FakeRedis()

        # Allow initial get (fail_key check) + set (NX lock), then fail
        # on the result_key publish.
        original_set = redis.set
        call_count = {"n": 0}

        async def flaky_set(key: str, value: Any, **kwargs):
            call_count["n"] += 1
            if call_count["n"] > 1:  # second set = result_key publish
                raise ConnectionError("publish fail")
            return await original_set(key, value, **kwargs)

        redis.set = flaky_set  # type: ignore[method-assign]

        src = SkillSource(refresher=refresher, redis=redis)
        # MUST NOT raise — best-effort by design
        out = await src.assess_risk(_make_call(_make_meta(risk_level=RiskLevel.LOW)))
        assert out.final_level == RiskLevel.LOW
