"""PE-2 §3.2 — McpSource: stateless constant-LOW adapter (behavior parity)."""

from __future__ import annotations

import pytest

from app.domain.services.permission.sources import McpSource
from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

# This repo runs async tests via EXPLICIT anyio markers (no asyncio_mode in
# pytest.ini); mirror test_native_source.py:11-16 or the async tests below are
# silently not awaited (false GREEN). (codex R1 P1#1)
pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_call(**overrides) -> ToolCallSpec:
    base = dict(
        tool_name="mcp_my_server_search",
        tool_args={"q": "hello"},
        tool_source="mcp",
        user_id="u",
        session_id="s",
        tool_call_id="tc-1",
    )
    base.update(overrides)
    return ToolCallSpec(**base)


class TestMcpSourceContract:
    def test_tool_source_class_var_is_mcp(self):
        assert McpSource.tool_source == "mcp"

    def test_is_a_permission_source(self):
        assert issubclass(McpSource, PermissionSource)

    def test_construct_with_no_args(self):
        """Stateless: no refresher, no redis, no live _mcp_tool dep — McpSource()
        takes no required args (registers at PE construction like NativeSource).
        (codex R1 P1#2: don't introspect __init__ — McpSource inherits
        object.__init__ whose signature is (self, *args, **kwargs); just
        construct it.)"""
        McpSource()  # must not raise — no constructor dependencies

    async def test_assess_risk_returns_constant_low(self):
        src = McpSource()
        assessment = await src.assess_risk(_make_call())
        assert isinstance(assessment, RiskAssessment)
        assert assessment.final_level == RiskLevel.LOW
        assert assessment.static_level == RiskLevel.LOW
        assert assessment.dynamic_level == RiskLevel.LOW

    async def test_assess_risk_populates_all_required_fields(self):
        """RiskAssessment has 11 required fields, no defaults — all must be set."""
        src = McpSource()
        a = await src.assess_risk(
            _make_call(primary_arg=None, dir_arg=None, arg_digest=None)
        )
        assert a.tool_name == "mcp_my_server_search"
        assert a.tool_args == {"q": "hello"}
        assert a.matched_patterns == []
        assert a.suggested_alternative is None
        # top-level args absent on MCP calls → coerced to "" / None (tool-level authz)
        assert a.primary_arg == ""
        assert a.dir_arg is None
        assert a.arg_digest == ""
        assert "mcp baseline" in a.risk_reason

    async def test_assess_risk_is_stateless_across_calls(self):
        src = McpSource()
        a1 = await src.assess_risk(_make_call(tool_name="mcp_a_x"))
        a2 = await src.assess_risk(_make_call(tool_name="mcp_b_x"))
        assert a1.final_level == a2.final_level == RiskLevel.LOW
        assert a1.tool_name == "mcp_a_x"
        assert a2.tool_name == "mcp_b_x"
