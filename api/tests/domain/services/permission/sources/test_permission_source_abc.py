"""PE-1 §2.3 — PermissionSource ABC contract."""

from __future__ import annotations

import inspect
from typing import ClassVar, get_type_hints

import pytest

from app.domain.services.permission.sources.base import PermissionSource


class TestPermissionSourceABC:
    def test_is_abstract(self):
        with pytest.raises(TypeError):
            PermissionSource()  # type: ignore[abstract]

    def test_class_var_tool_source_declared(self):
        hints = get_type_hints(PermissionSource, include_extras=True)
        assert "tool_source" in hints

    def test_assess_risk_is_async(self):
        method = PermissionSource.assess_risk
        assert inspect.iscoroutinefunction(method)

    def test_assess_risk_signature(self):
        sig = inspect.signature(PermissionSource.assess_risk)
        params = list(sig.parameters)
        # self + call
        assert params == ["self", "call"]
        # return annotation = RiskAssessment
        from app.domain.services.risk_assessor import RiskAssessment
        assert sig.return_annotation is RiskAssessment

    def test_subclass_with_only_tool_source_still_abstract(self):
        """tool_source alone doesn't satisfy ABC — assess_risk must be defined."""

        class _Half(PermissionSource):
            tool_source: ClassVar[str] = "half"

        with pytest.raises(TypeError):
            _Half()  # type: ignore[abstract]

    def test_concrete_subclass_instantiable(self):
        from app.domain.services.permission.tool_call_spec import ToolCallSpec
        from app.domain.services.risk_assessor import (
            RiskAssessment,
            RiskLevel,
        )

        class _Echo(PermissionSource):
            tool_source: ClassVar[str] = "echo"

            async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
                return RiskAssessment(
                    tool_name=call.tool_name,
                    tool_args=dict(call.tool_args),
                    static_level=RiskLevel.NONE,
                    dynamic_level=RiskLevel.NONE,
                    final_level=RiskLevel.NONE,
                    risk_reason="echo",
                    matched_patterns=[],
                    suggested_alternative=None,
                    primary_arg="",
                    dir_arg=None,
                    arg_digest="",
                )

        e = _Echo()
        assert e.tool_source == "echo"
