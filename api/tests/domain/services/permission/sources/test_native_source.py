"""PE-1 §3.1 — NativeSource passthrough."""

from __future__ import annotations

import pytest

from app.domain.services.permission.sources.native_source import NativeSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_call(assessment: RiskAssessment | None) -> ToolCallSpec:
    return ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/tmp/x", "content": "y"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
        primary_arg="/tmp/x",
        dir_arg="/tmp",
        arg_digest="abc123",
        risk_assessment=assessment,
        tool_call_id="tc1",
    )


def _sample_assessment() -> RiskAssessment:
    return RiskAssessment(
        tool_name="file_write",
        tool_args={"path": "/tmp/x", "content": "y"},
        static_level=RiskLevel.MEDIUM,
        dynamic_level=RiskLevel.NONE,
        final_level=RiskLevel.MEDIUM,
        risk_reason="path under /tmp",
        matched_patterns=["tmp_write"],
        suggested_alternative=None,
        primary_arg="/tmp/x",
        dir_arg="/tmp",
        arg_digest="abc123",
    )


class TestNativeSource:
    def test_tool_source_class_attr(self):
        assert NativeSource.tool_source == "native"

    async def test_passthrough_returns_caller_assessment_object(self):
        src = NativeSource()
        a = _sample_assessment()
        out = await src.assess_risk(_make_call(a))
        assert out is a  # passthrough — same object, no copy

    async def test_raises_if_caller_omits_risk_assessment(self):
        src = NativeSource()
        with pytest.raises(ValueError, match="pre-filled risk_assessment"):
            await src.assess_risk(_make_call(None))
