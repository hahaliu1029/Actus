"""PE-1 source adapter subpackage. PermissionSource ABC + per-source adapters."""

from app.domain.services.permission.source_metadata import (
    SkillCallMetadata,
    SourceMetadata,
)
from app.domain.services.permission.sources.base import PermissionSource
from app.domain.services.permission.sources.gate_helper import (
    PE_SUPPORTED_SOURCES_AFTER_PE_1,
    is_pe_eligible_tool_source,
    is_pe_enabled_for_source,
)
from app.domain.services.permission.sources.native_source import NativeSource
from app.domain.services.permission.sources.skill_metadata import (
    SkillRiskRefreshResult,
    build_skill_call_metadata,
)
from app.domain.services.permission.sources.skill_source import SkillSource

__all__ = [
    "SkillCallMetadata",
    "SourceMetadata",
    "PermissionSource",
    "NativeSource",
    "SkillSource",
    "SkillRiskRefreshResult",
    "build_skill_call_metadata",
    "PE_SUPPORTED_SOURCES_AFTER_PE_1",
    "is_pe_enabled_for_source",
    "is_pe_eligible_tool_source",
]
