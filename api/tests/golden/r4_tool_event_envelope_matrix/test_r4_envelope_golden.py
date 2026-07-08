"""R4 CS3 envelope golden matrix runner.

26 golden JSON fixtures: 21 R2 non-Asked artifacts + 5 legacy shapes.

Assertion:
1. fixture JSON → ToolEvent.model_validate → projector → envelope
2. envelope.model_dump(mode="json", by_alias=True) 与 expected 逐字节相等 (I-R4.6)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.application.services.tool_event_envelope_v1 import project_tool_event_to_envelope_v1
from app.domain.models.event import ToolEvent

FIXTURES_DIR = Path(__file__).parent / "fixtures"
EXPECTED_DIR = Path(__file__).parent / "expected"

FIXTURE_NAMES = [
    # native (9)
    "native_allow_success",
    "native_allow_error_exception",
    "native_allow_error_timeout",
    "native_denied_ast_validator",
    "native_denied_approval_policy",
    "native_denied_smart_approve",
    "native_passthrough_image",
    "native_passthrough_pdf",
    "native_passthrough_text",
    # mcp (4)
    "mcp_allow_success",
    "mcp_allow_error_exception",
    "mcp_allow_error_timeout",
    "mcp_denied_approval_policy",
    # a2a (3)
    "a2a_allow_success",
    "a2a_allow_error_exception",
    "a2a_denied_approval_policy",
    # skill (3)
    "skill_allow_success",
    "skill_allow_error_exception",
    "skill_denied_approval_policy",
    # mixed / extra (2)
    "native_passthrough_mixed",
    "native_passthrough_image_mediatype",
    "native_passthrough_video_mediatype",
    "native_passthrough_pdf_docpreview",
    "native_passthrough_extraction_docpreview",
    "native_passthrough_mixed_docpreview",
    "skill_denied_risk_enforce",
    # legacy (5)
    "legacy_allow_success",
    "legacy_failure",
    "legacy_calling_only",
    "legacy_no_result",
    "legacy_pre_r1_no_tool_source",
]


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text())


def _load_expected(name: str) -> dict[str, Any]:
    return json.loads((EXPECTED_DIR / f"{name}_expected.json").read_text())


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
def test_envelope_transform_binary_equality(fixture_name: str) -> None:
    """Each fixture's projector output must binary-equal its expected wire JSON."""
    fixture = _load_fixture(fixture_name)
    expected = _load_expected(fixture_name)

    evt = ToolEvent.model_validate(fixture)
    envelope = project_tool_event_to_envelope_v1(evt)
    actual = envelope.model_dump(mode="json", by_alias=True)

    assert actual == expected, (
        f"[{fixture_name}] envelope wire drift:\n"
        f"actual:   {json.dumps(actual, indent=2, sort_keys=True, ensure_ascii=False)}\n"
        f"expected: {json.dumps(expected, indent=2, sort_keys=True, ensure_ascii=False)}"
    )


def test_fixture_count_matches_expected() -> None:
    assert len(FIXTURE_NAMES) == 31
    for name in FIXTURE_NAMES:
        assert (FIXTURES_DIR / f"{name}.json").exists(), f"missing fixture: {name}"
        assert (EXPECTED_DIR / f"{name}_expected.json").exists(), f"missing expected: {name}"
