"""PE-1 §3.2 — ToolCallSpec.source_metadata new field."""

from __future__ import annotations

import dataclasses

import pytest

from app.domain.models.skill import SkillRuntimeType
from app.domain.services.permission.source_metadata import SkillCallMetadata
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskLevel


def test_tool_call_spec_is_frozen():
    spec = ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/workspace/notes.md", "content": "hi"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tool_name = "x"


def test_tool_call_spec_optional_fields_default():
    spec = ToolCallSpec(
        tool_name="shell_execute",
        tool_args={"command": "ls"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
    )
    assert spec.primary_arg is None
    assert spec.dir_arg is None
    assert spec.arg_digest is None
    assert spec.risk_assessment is None


def test_tool_call_spec_tool_source_str_only():
    # str typed; PE-1+ will validate the union at construction time
    spec = ToolCallSpec(
        tool_name="x", tool_args={}, tool_source="native",
        user_id="u", session_id="s",
    )
    assert spec.tool_source == "native"


def test_default_source_metadata_is_none():
    call = ToolCallSpec(
        tool_name="t", tool_args={}, tool_source="native",
        user_id="u", session_id="s",
    )
    assert call.source_metadata is None


def test_construct_with_skill_call_metadata():
    meta = SkillCallMetadata(
        tool_name="my_skill",
        skill_id="sk1",
        content_hash="h",
        risk_level=RiskLevel.HIGH,
        runtime_type=SkillRuntimeType.NATIVE,
        trust_origin="user_installed",
        scan_verdict="safe",
    )
    call = ToolCallSpec(
        tool_name="my_skill", tool_args={}, tool_source="skill",
        user_id="u", session_id="s",
        source_metadata=meta,
    )
    assert call.source_metadata is meta


def test_dataclasses_replace_preserves_source_metadata():
    meta = SkillCallMetadata(
        tool_name="t", skill_id="s", content_hash=None,
        risk_level=RiskLevel.LOW, runtime_type=SkillRuntimeType.MCP,
        trust_origin="builtin", scan_verdict="safe",
    )
    call = ToolCallSpec(
        tool_name="t", tool_args={}, tool_source="skill",
        user_id="u", session_id="s",
        source_metadata=meta,
    )
    # Replacing risk_assessment must keep source_metadata intact (used by
    # default_engine.evaluate step 5.5)
    from app.domain.services.risk_assessor import RiskAssessment

    new_assessment = RiskAssessment(
        tool_name="t", tool_args={}, static_level=RiskLevel.HIGH,
        dynamic_level=RiskLevel.NONE, final_level=RiskLevel.HIGH,
        risk_reason="r", matched_patterns=[], suggested_alternative=None,
        primary_arg="", dir_arg=None, arg_digest="",
    )
    call2 = dataclasses.replace(call, risk_assessment=new_assessment)
    assert call2.source_metadata is meta
    assert call2.risk_assessment is new_assessment
