"""NativeSource — passthrough adapter.

Per spec §3.1 / Q3.3 S1: native still pre-fills (existing tool_node /
RiskAssessor pipeline); the source adapter merely returns the
caller-prefilled ``RiskAssessment``. Skill source is the one that
recomputes (Risk #1 hard rule).
"""

from __future__ import annotations

from typing import ClassVar

from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment


class NativeSource(PermissionSource):
    tool_source: ClassVar[str] = "native"

    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        if call.risk_assessment is None:
            raise ValueError(
                "native source requires pre-filled risk_assessment "
                "(set by RiskAssessor.assess in tool_node before pe.evaluate)"
            )
        return call.risk_assessment
