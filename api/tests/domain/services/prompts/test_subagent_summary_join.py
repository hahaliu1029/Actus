"""Tests for summary join prompt template + deterministic validator."""
from app.domain.services.prompts.subagent_summary_join import (
    build_summary_prompt,
    validate_joined_summary,
)


class _StubChildResult:
    def __init__(self, child_id, outcome, final_answer=None, error_summary=None):
        self.child_id = child_id
        self.outcome = outcome
        self.final_answer = final_answer
        self.error_summary = error_summary


def test_validator_passes_clean_summary():
    completed = [
        _StubChildResult(child_id="c-abcdef01", outcome="completed", final_answer="A"),
        _StubChildResult(child_id="c-12345678", outcome="completed", final_answer="B"),
    ]
    summary = (
        "背景：研究 X。\n"
        "关键发现：发现 1 [[C1:c-abcdef]]；发现 2 [[C2:c-123456]]。\n"
        "整合判断：综合两点 [[C1:c-abcdef]] [[C2:c-123456]]。"
    )
    ok, errors = validate_joined_summary(summary, completed, dropped_children=[])
    assert ok is True, f"unexpected errors: {errors}"


def test_validator_catches_missing_citation():
    completed = [
        _StubChildResult(child_id="c-abcdef01", outcome="completed"),
        _StubChildResult(child_id="c-12345678", outcome="completed"),
    ]
    summary = "Some text with [[C1:c-abcdef]] only."
    ok, errors = validate_joined_summary(summary, completed, dropped_children=[])
    assert ok is False
    assert any("c-12345678" in e or "C2" in e for e in errors)


def test_validator_catches_fabricated_marker():
    completed = [_StubChildResult(child_id="c-abcdef01", outcome="completed")]
    summary = "Text [[C1:c-abcdef]] and [[C99:deadbeef]] (fabricated)."
    ok, errors = validate_joined_summary(summary, completed, dropped_children=[])
    assert ok is False
    assert any("fabricated" in e.lower() or "C99" in e for e in errors)


def test_validator_length_cap():
    completed = [_StubChildResult(child_id="c-abcdef01", outcome="completed")]
    summary = "[[C1:c-abcdef]] " + "x" * 1300
    ok, errors = validate_joined_summary(summary, completed, dropped_children=[])
    assert ok is False
    assert any("cap" in e.lower() or "1200" in e for e in errors)


def test_validator_dropped_outcome_consistency():
    completed = [_StubChildResult(child_id="c-abcdef01", outcome="completed")]
    summary = "Cited [[C1:c-abcdef]]."
    dropped = [{"child_id": "c-x", "outcome": "unknown_state"}]
    ok, errors = validate_joined_summary(summary, completed, dropped_children=dropped)
    assert ok is False
    assert any("dropped" in e.lower() or "unknown_state" in e for e in errors)


def test_build_summary_prompt_includes_all_completed_inputs():
    completed = [
        _StubChildResult(child_id="c-abcdef01", outcome="completed", final_answer="A"),
        _StubChildResult(child_id="c-12345678", outcome="completed", final_answer="B"),
    ]
    dropped = []
    prompts = ["Q1 text", "Q2 text"]
    out = build_summary_prompt(prompts, completed, dropped)
    assert "Q1 text" in out
    assert "Q2 text" in out
    assert "A" in out
    assert "B" in out
    assert "c-abcdef" in out
