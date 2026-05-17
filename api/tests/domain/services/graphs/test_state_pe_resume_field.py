"""State schema: pe_resume_outcomes is added alongside approved_tool_call_ids
(legacy field kept for PE-unwired fail-open until PE-3 cleanup).

Phase 10, Task 10.1.
"""
from __future__ import annotations

from app.domain.services.graphs.state import ReactGraphState


def test_state_has_pe_resume_outcomes_field_typed():
    annotations = ReactGraphState.__annotations__
    assert "pe_resume_outcomes" in annotations


def test_state_keeps_legacy_approved_tool_call_ids_for_fail_open():
    annotations = ReactGraphState.__annotations__
    assert "approved_tool_call_ids" in annotations, (
        "PE-0 keeps the legacy field alongside pe_resume_outcomes; "
        "PE-3 cleanup removes approved_tool_call_ids after all 4 sources migrate."
    )
