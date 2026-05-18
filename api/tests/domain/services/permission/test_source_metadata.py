"""PE-1 §3.1 — no-cycle SkillCallMetadata DTO tests."""

from __future__ import annotations

import dataclasses

import pytest

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.permission.source_metadata import (
    SkillCallMetadata,
    SourceMetadata,
)
from app.domain.services.risk_assessor import RiskLevel


def test_skill_call_metadata_is_frozen_dataclass():
    meta = SkillCallMetadata(
        tool_name="myskill_install",
        skill_id="sk_abc123",
        content_hash="sha256:deadbeef",
        risk_level=RiskLevel.HIGH,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )
    assert meta.tool_name == "myskill_install"
    assert meta.skill_id == "sk_abc123"
    assert meta.content_hash == "sha256:deadbeef"
    assert meta.risk_level == RiskLevel.HIGH
    assert meta.runtime_type == SkillRuntimeType.NATIVE
    assert meta.trust_origin == "user_installed"
    assert meta.scan_verdict == "safe"

    # frozen — replace not assign
    with pytest.raises(dataclasses.FrozenInstanceError):
        meta.skill_id = "sk_other"  # type: ignore[misc]


def test_skill_call_metadata_content_hash_optional():
    """content_hash can be None when the skill was never scanned
    (consistent with skill.py:177-181 missing-hash fallback)."""
    meta = SkillCallMetadata(
        tool_name="myskill",
        skill_id="sk_abc",
        content_hash=None,
        risk_level=RiskLevel.LOW,
        runtime_type=SkillRuntimeType.MCP,
        trust_origin="builtin",
        scan_verdict="unscanned",
    )
    assert meta.content_hash is None


def test_source_metadata_type_alias_resolves_to_skill_call_metadata():
    """PE-1 only registers skill; SourceMetadata = SkillCallMetadata for now.
    PE-2 will extend to Union (SkillCallMetadata | McpCallMetadata)."""
    meta: SourceMetadata = SkillCallMetadata(  # type: ignore[assignment]
        tool_name="t",
        skill_id="s",
        content_hash=None,
        risk_level=RiskLevel.NONE,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="builtin",
        scan_verdict="safe",
    )
    assert isinstance(meta, SkillCallMetadata)


def test_source_metadata_module_has_no_internal_pe_deps():
    """no-cycle invariant — this module MUST NOT import from sources/ or tool_call_spec."""
    import app.domain.services.permission.source_metadata as mod
    import inspect

    src = inspect.getsource(mod)
    assert "from app.domain.services.permission.sources" not in src
    assert "from app.domain.services.permission.tool_call_spec" not in src
