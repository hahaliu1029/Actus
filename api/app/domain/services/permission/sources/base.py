"""PermissionSource ABC — source-specific risk derivation only.

Per spec §2.3 / §2.4:
- SourceMetadata alias is NOT defined here (lives in
  ``permission/source_metadata.py`` to avoid the
  ``tool_call_spec → sources.base → tool_call_spec`` import cycle).
- ``assess_risk`` returns ``RiskAssessment`` only — NEVER ``ToolOutcome``
  (queue / writer / SSM are owned by ``DefaultPermissionEngine``; INV-1b/2/3
  static scans enforce this).
- NativeSource passes through ``call.risk_assessment``; SkillSource
  recomputes ignoring caller pre-fill (Risk #1 hard rule).

Note: We intentionally do NOT use ``from __future__ import annotations`` here
so that ``inspect.signature(PermissionSource.assess_risk).return_annotation``
resolves to the live ``RiskAssessment`` class (identity-comparable). The ABC
contract test in ``tests/.../test_permission_source_abc.py`` relies on this.
"""

from abc import ABC, abstractmethod
from typing import ClassVar

from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment


class PermissionSource(ABC):
    """Per-source RiskAssessment derivation. ONE method only."""

    tool_source: ClassVar[str]

    @abstractmethod
    async def assess_risk(self, call: ToolCallSpec) -> RiskAssessment:
        """Return the canonical RiskAssessment for this source.

        MUST NOT touch writer / queue / SSM mutators (INV-1b/2/3 AST scan
        enforces). MUST NOT return ToolOutcome.
        """
