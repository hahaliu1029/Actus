"""PE-1 §2.4 — DefaultPermissionEngine step 5.5 (uniform source.assess_risk)
+ step 9 reason.type per-source dispatch + Risk #1 caller-override hard rule.

Uses lightweight in-memory fakes so we exercise the dispatch path without
spinning up Postgres/Redis.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.models.skill import SkillRuntimeType
from app.domain.models.tool_result import Asked, AllowSuccess, Denied
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.errors import UnsupportedSource
from app.domain.services.permission.source_metadata import SkillCallMetadata
from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _ra(level: RiskLevel, *, reason: str = "r") -> RiskAssessment:
    return RiskAssessment(
        tool_name="t", tool_args={}, static_level=level,
        dynamic_level=RiskLevel.NONE, final_level=level,
        risk_reason=reason, matched_patterns=[], suggested_alternative=None,
        primary_arg="", dir_arg=None, arg_digest="d",
    )


def _ctx() -> EvaluationContext:
    return EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        retry_count=0,
    )


class _RecordingSource(PermissionSource):
    def __init__(self, tool_source: str, returns: RiskAssessment):
        type(self).tool_source = tool_source  # set ClassVar dynamically for fake
        self.tool_source = tool_source
        self._returns = returns
        self.calls: list[ToolCallSpec] = []

    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        self.calls.append(call)
        return self._returns


def _make_engine(sources: dict[str, PermissionSource]) -> DefaultPermissionEngine:
    class _U:
        async def __aenter__(self):
            self.user_tool_approval_policy = AsyncMock()
            self.user_tool_approval_policy.get = AsyncMock(return_value=None)
            return self

        async def __aexit__(self, *a):
            return None

    def _uow_factory():
        return _U()

    writer = AsyncMock()
    queue = AsyncMock()
    queue.read = AsyncMock(return_value=None)
    queue.store = AsyncMock(return_value=None)
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 1)
    )
    reader = AsyncMock()
    reader.check = AsyncMock(return_value="no_match")

    return DefaultPermissionEngine(
        uow_factory=_uow_factory,
        writer=writer,
        queue=queue,
        session_machine=ssm,
        reader=reader,
        escalation_registry={},  # no SmartApprove → falls through to Asked enqueue
        sources=sources,
    )


def _native_call(level: RiskLevel) -> ToolCallSpec:
    return ToolCallSpec(
        tool_name="file_write", tool_args={"path": "/tmp/x"},
        tool_source="native", user_id="u", session_id="s",
        risk_assessment=_ra(level), tool_call_id="tc",
    )


def _skill_call(*, meta_risk: RiskLevel = RiskLevel.LOW,
                caller_risk: RiskLevel | None = None) -> ToolCallSpec:
    meta = SkillCallMetadata(
        tool_name="my_skill",
        skill_id="sk_x", content_hash="h",
        risk_level=meta_risk,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )
    return ToolCallSpec(
        tool_name="my_skill", tool_args={"q": 1}, tool_source="skill",
        user_id="u", session_id="s",
        risk_assessment=_ra(caller_risk) if caller_risk is not None else None,
        tool_call_id="tc",
        source_metadata=meta,
    )


# ---------- Source dispatch ----------

class TestSourceDispatch:
    async def test_native_call_routes_through_native_source(self):
        native = _RecordingSource("native", _ra(RiskLevel.LOW))
        skill = _RecordingSource("skill", _ra(RiskLevel.HIGH))
        pe = _make_engine({"native": native, "skill": skill})
        await pe.evaluate(_native_call(RiskLevel.LOW), _ctx())
        assert len(native.calls) == 1
        assert len(skill.calls) == 0

    async def test_skill_call_routes_through_skill_source(self):
        native = _RecordingSource("native", _ra(RiskLevel.LOW))
        skill = _RecordingSource("skill", _ra(RiskLevel.HIGH))
        pe = _make_engine({"native": native, "skill": skill})
        await pe.evaluate(_skill_call(meta_risk=RiskLevel.HIGH), _ctx())
        assert len(skill.calls) == 1
        assert len(native.calls) == 0

    async def test_unsupported_source_raises_unsupported_source(self):
        native = _RecordingSource("native", _ra(RiskLevel.NONE))
        pe = _make_engine({"native": native})
        bogus = ToolCallSpec(
            tool_name="t", tool_args={}, tool_source="mystery",
            user_id="u", session_id="s",
        )
        with pytest.raises(UnsupportedSource) as exc_info:
            await pe.evaluate(bogus, _ctx())
        assert exc_info.value.source == "mystery"


# ---------- Risk #1 — SkillSource recompute overrides caller-prefilled risk ----------

class TestRisk1CallerOverride:
    async def test_skill_high_overrides_caller_low(self):
        """End-to-end: caller pre-filled LOW, source recomputed HIGH → Asked."""
        # SkillSource returns HIGH; PE step 6 reader=no_match → step 7b
        # (no SmartApprove provider in registry) → Asked.
        skill = _RecordingSource("skill", _ra(RiskLevel.HIGH))
        native = _RecordingSource("native", _ra(RiskLevel.LOW))
        pe = _make_engine({"native": native, "skill": skill})
        out = await pe.evaluate(
            _skill_call(meta_risk=RiskLevel.HIGH, caller_risk=RiskLevel.LOW), _ctx(),
        )
        assert isinstance(out, Asked)
        # reason.code carries 'ask:high' (lowercased final_level name)
        assert out.reason.code == "ask:high"


# ---------- Step 9 reason.type per-source dispatch ----------

class TestReasonTypeDispatch:
    async def test_skill_asked_uses_risk_enforce_reason_type(self):
        skill = _RecordingSource("skill", _ra(RiskLevel.HIGH, reason="skill is spicy"))
        native = _RecordingSource("native", _ra(RiskLevel.LOW))
        pe = _make_engine({"native": native, "skill": skill})
        out = await pe.evaluate(_skill_call(meta_risk=RiskLevel.HIGH), _ctx())
        assert isinstance(out, Asked)
        assert out.reason.type == "risk_enforce"
        assert out.reason.message == "skill is spicy"

    async def test_native_asked_uses_approval_policy_reason_type(self):
        skill = _RecordingSource("skill", _ra(RiskLevel.LOW))
        native = _RecordingSource("native", _ra(RiskLevel.HIGH, reason="native danger"))
        pe = _make_engine({"native": native, "skill": skill})
        out = await pe.evaluate(_native_call(RiskLevel.HIGH), _ctx())
        assert isinstance(out, Asked)
        assert out.reason.type == "approval_policy"
        assert out.reason.message == "user confirmation required"


# ---------- Confirmation detail risk level uses post-source assessment ----------

class TestConfirmationDetailRiskLevel:
    async def test_skill_asked_stores_high_risk_in_detail(self):
        """PE step 8: ConfirmationDetail.risk_level comes from the
        recomputed assessment (post step 5.5), NOT caller pre-fill."""
        skill = _RecordingSource("skill", _ra(RiskLevel.HIGH))
        pe = _make_engine({"native": _RecordingSource("native", _ra(RiskLevel.LOW)),
                           "skill": skill})
        out = await pe.evaluate(
            _skill_call(meta_risk=RiskLevel.HIGH, caller_risk=RiskLevel.LOW), _ctx(),
        )
        assert isinstance(out, Asked)
        # Inspect the ConfirmationDetail that was stored
        stored = pe._queue.store.call_args.args[0]
        assert stored.risk_level == "high"


class TestRegisterSourcePostInit:
    async def test_register_source_adds_to_sources_mapping(self):
        from app.domain.services.permission.sources import NativeSource
        skill = _RecordingSource("skill", _ra(RiskLevel.LOW))
        pe = _make_engine({"native": NativeSource()})
        assert "skill" not in pe._sources
        pe.register_source("skill", skill)
        assert pe._sources["skill"] is skill

    async def test_register_source_rejects_double_registration(self):
        from app.domain.services.permission.sources import NativeSource
        pe = _make_engine({"native": NativeSource()})
        pe.register_source("skill", _RecordingSource("skill", _ra(RiskLevel.LOW)))
        with pytest.raises(ValueError, match="already registered"):
            pe.register_source("skill", _RecordingSource("skill", _ra(RiskLevel.LOW)))
