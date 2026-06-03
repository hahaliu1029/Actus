"""PE-3 §3.2 — A2aSource: stateless constant-LOW adapter (behavior parity)."""

from __future__ import annotations

import pytest

from app.domain.services.permission.sources import A2aSource
from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

# Explicit anyio marker — this repo has no asyncio_mode in pytest.ini, so async
# tests without the marker are silently not awaited (false GREEN).
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_call(**overrides) -> ToolCallSpec:
    base = dict(
        tool_name="call_remote_agent",
        tool_args={"id": "agent-1", "query": "do a thing"},
        tool_source="a2a",
        user_id="u",
        session_id="s",
        tool_call_id="tc-1",
    )
    base.update(overrides)
    return ToolCallSpec(**base)


class TestA2aSourceContract:
    def test_tool_source_class_var_is_a2a(self):
        assert A2aSource.tool_source == "a2a"

    def test_is_a_permission_source(self):
        assert issubclass(A2aSource, PermissionSource)

    def test_construct_with_no_args(self):
        """Stateless: no refresher, no redis, no live A2ATool.manager dep —
        A2aSource() takes no required args (registers at PE construction like
        McpSource/NativeSource)."""
        A2aSource()  # must not raise — no constructor dependencies

    async def test_assess_risk_returns_constant_low(self):
        src = A2aSource()
        assessment = await src.assess_risk(_make_call())
        assert isinstance(assessment, RiskAssessment)
        assert assessment.final_level == RiskLevel.LOW
        assert assessment.static_level == RiskLevel.LOW
        assert assessment.dynamic_level == RiskLevel.LOW

    async def test_assess_risk_populates_all_required_fields(self):
        """RiskAssessment has 11 required fields, no defaults — all must be set."""
        src = A2aSource()
        a = await src.assess_risk(
            _make_call(primary_arg=None, dir_arg=None, arg_digest=None)
        )
        assert a.tool_name == "call_remote_agent"
        assert a.tool_args == {"id": "agent-1", "query": "do a thing"}
        assert a.matched_patterns == []
        assert a.suggested_alternative is None
        # top-level args absent on a2a calls → coerced to "" / None (tool-level authz)
        assert a.primary_arg == ""
        assert a.dir_arg is None
        assert a.arg_digest == ""
        assert "a2a baseline" in a.risk_reason

    async def test_assess_risk_is_stateless_across_calls(self):
        src = A2aSource()
        a1 = await src.assess_risk(_make_call(tool_name="call_remote_agent"))
        a2 = await src.assess_risk(_make_call(tool_name="get_remote_agent_cards"))
        assert a1.final_level == a2.final_level == RiskLevel.LOW
        assert a1.tool_name == "call_remote_agent"
        assert a2.tool_name == "get_remote_agent_cards"
