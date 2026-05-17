"""SmartApprove string -> ToolOutcome adapter; timeout/exception -> Asked."""

import asyncio
from unittest.mock import AsyncMock

from app.domain.models.session import SessionStatus
from app.domain.models.tool_result import AllowSuccess, Asked, Denied
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.smart_approve_provider import (
    SmartApproveProvider,
)
from app.domain.services.permission.tool_call_spec import ToolCallSpec


def _call():
    return ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/x"},
        tool_source="native",
        user_id="u",
        session_id="s",
        tool_call_id="tc1",
    )


def _ctx():
    return EvaluationContext(
        session_mode=SessionStatus.RUNNING, session_mode_revision=0,
    )


def _run(coro):
    return asyncio.run(coro)


def test_approve_string_maps_to_allow_success():
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="approve")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0)
    out = _run(provider.resolve(_call(), _ctx(), None))
    assert isinstance(out, AllowSuccess)
    assert "smart_approve" in out.data.get("via", "")


def test_deny_string_maps_to_denied():
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="deny")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0)
    out = _run(provider.resolve(_call(), _ctx(), None))
    assert isinstance(out, Denied)
    assert out.reason.type == "smart_approve"


def test_escalate_string_maps_to_asked_with_confirmation_id():
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="escalate")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0)
    out = _run(provider.resolve(_call(), _ctx(), None))
    assert isinstance(out, Asked)
    assert out.confirmation_id == "s:tc1"
    assert out.reason.type == "smart_approve"


def test_timeout_falls_through_to_asked_not_denied():
    """codex P2-11: timeout MUST fall through to user, NOT deny."""

    async def slow(*a, **kw):
        await asyncio.sleep(10)
        return "approve"

    inner = type("Inner", (), {"evaluate": slow})()
    provider = SmartApproveProvider(inner, timeout_seconds=0.1)
    out = _run(provider.resolve(_call(), _ctx(), None))
    assert isinstance(out, Asked)
    assert out.reason.code == "smart_approve_unavailable_falling_through"


def test_exception_falls_through_to_asked_not_denied():
    inner = AsyncMock()
    inner.evaluate = AsyncMock(side_effect=RuntimeError("oops"))
    provider = SmartApproveProvider(inner, timeout_seconds=1.0)
    out = _run(provider.resolve(_call(), _ctx(), None))
    assert isinstance(out, Asked)
    assert out.reason.code == "smart_approve_unavailable_falling_through"


# ---------------------------------------------------------------------------
# P1#4: medium_only=True tests
# ---------------------------------------------------------------------------


def _call_with_risk(risk: str) -> ToolCallSpec:
    """Build a ToolCallSpec with a RiskAssessment at the given level."""
    from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

    level = RiskLevel[risk.upper()]
    assessment = RiskAssessment(
        tool_name="file_write",
        tool_args={"path": "/x"},
        static_level=level,
        dynamic_level=level,
        final_level=level,
        risk_reason="test",
        matched_patterns=[],
        suggested_alternative=None,
        primary_arg="/x",
        dir_arg=None,
        arg_digest="d1",
    )
    return ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/x"},
        tool_source="native",
        user_id="u",
        session_id="s",
        tool_call_id="tc1",
        risk_assessment=assessment,
    )


def test_medium_only_true_high_risk_skips_llm():
    """P1#4: medium_only=True + HIGH risk → Asked without calling inner.evaluate."""
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="approve")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0, medium_only=True)
    out = _run(provider.resolve(_call_with_risk("high"), _ctx(), None))
    assert isinstance(out, Asked)
    assert out.reason.code == "smart_approve_medium_only_high_skip"
    inner.evaluate.assert_not_awaited()


def test_medium_only_true_medium_risk_calls_llm():
    """P1#4: medium_only=True + MEDIUM risk → LLM is still called."""
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="approve")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0, medium_only=True)
    out = _run(provider.resolve(_call_with_risk("medium"), _ctx(), None))
    assert isinstance(out, AllowSuccess)
    inner.evaluate.assert_awaited_once()


def test_medium_only_false_high_risk_calls_llm():
    """P1#4: medium_only=False (default) + HIGH risk → LLM is called as normal."""
    inner = AsyncMock()
    inner.evaluate = AsyncMock(return_value="approve")
    provider = SmartApproveProvider(inner, timeout_seconds=1.0, medium_only=False)
    out = _run(provider.resolve(_call_with_risk("high"), _ctx(), None))
    assert isinstance(out, AllowSuccess)
    inner.evaluate.assert_awaited_once()
