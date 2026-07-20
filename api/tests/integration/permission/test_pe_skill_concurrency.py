"""PE-1 §6.3 — Redis single-flight concurrency.

5 concurrent assess_risk calls for the same (skill_id, content_hash):
- One winner runs the refresher exactly once.
- Four losers poll the result key and observe the same status.
- Without the singleflight, the refresher would run 5x and Asked would
  storm the user with 5 confirm events.

Markers: @pytest.mark.integration  requires Redis.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


class _SlowRefresher:
    def __init__(self, *, sleep_seconds: float, status: str = "refreshed"):
        from app.domain.services.permission.sources.skill_metadata import (
            SkillRiskRefreshResult,
        )
        from app.domain.services.risk_assessor import RiskLevel
        self._sleep = sleep_seconds
        self._result = (
            SkillRiskRefreshResult(status="refreshed", risk_level=RiskLevel.HIGH)
            if status == "refreshed"
            else SkillRiskRefreshResult(status="failed", error="injected")
        )
        self.calls = 0

    async def refresh(self, tool_name: str, expected_content_hash: str | None):
        self.calls += 1
        await asyncio.sleep(self._sleep)
        return self._result


def _make_meta(skill_id_prefix: str = "sk_concurrency"):
    """Unique skill_id + content_hash per call so the 45s failure cache
    cannot poison reruns within the same module."""
    from uuid import uuid4
    from app.domain.models.skill import SkillRuntimeType
    from app.domain.services.permission.source_metadata import SkillCallMetadata
    from app.domain.services.risk_assessor import RiskLevel
    unique = uuid4().hex[:8]
    return SkillCallMetadata(
        tool_name="t_concurrency",
        skill_id=f"{skill_id_prefix}_{unique}",
        content_hash=f"sha256:hash_{unique}",
        risk_level=RiskLevel.LOW,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )


def _make_call(meta: Any):
    from app.domain.services.permission.tool_call_spec import ToolCallSpec
    return ToolCallSpec(
        tool_name=meta.tool_name, tool_args={"q": 1}, tool_source="skill",
        user_id="u", session_id="s", tool_call_id="tc_conc",
        source_metadata=meta,
    )


async def test_5_concurrent_winner_runs_refresher_once_losers_observe_result(
    redis_client,
):
    """Winner refresh succeeds  losers see status='refreshed' via result_key.
    Refresher invoked exactly once (count == 1)."""
    from app.domain.services.permission.sources.skill_source import SkillSource
    from app.domain.services.risk_assessor import RiskLevel

    raw_redis = getattr(redis_client, "client", redis_client)
    # Generous slow refresher so all losers definitely overlap
    refresher = _SlowRefresher(sleep_seconds=0.5, status="refreshed")
    src = SkillSource(refresher=refresher, redis=raw_redis)
    meta = _make_meta()

    results = await asyncio.gather(*[
        src.assess_risk(_make_call(meta)) for _ in range(5)
    ])

    assert refresher.calls == 1
    # Winner returns HIGH (refreshed); losers also return HIGH via result_key
    levels = {r.final_level for r in results}
    assert levels == {RiskLevel.HIGH}


async def test_5_concurrent_winner_failure_losers_observe_fail_force_high(
    redis_client,
):
    """Winner refresh fails  losers see fail_key  all force-HIGH (one
    confirm storm is unavoidable when winner fails, but losers must not
    *each* run the refresher)."""
    from app.domain.services.permission.sources.skill_source import SkillSource
    from app.domain.services.risk_assessor import RiskLevel

    raw_redis = getattr(redis_client, "client", redis_client)
    refresher = _SlowRefresher(sleep_seconds=0.2, status="failed")
    src = SkillSource(refresher=refresher, redis=raw_redis)
    meta = _make_meta(skill_id_prefix="sk_fail_storm")

    results = await asyncio.gather(*[
        src.assess_risk(_make_call(meta)) for _ in range(5)
    ])
    assert refresher.calls == 1
    levels = {r.final_level for r in results}
    assert levels == {RiskLevel.HIGH}
