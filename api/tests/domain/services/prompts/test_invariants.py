"""B5 C1: regex pattern + dangling skill ref invariant tests."""
from __future__ import annotations

import pytest

from app.domain.services.prompts.errors import ToolBindingInvariantError
from app.domain.services.prompts.invariants import (
    _SKILL_TOOL_PATTERN,
    _assert_no_dangling_skill_tool_refs,
)


# ---- Regex pattern coverage --------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    [
        "skill_foo_bar",
        "skill_foo_bar_baz",
        "skill_foo__bar_tool",  # double underscore (real edge case)
        "skill_foo___bar",  # triple underscore
        "skill_a_b",
        "skill_1_2_3",  # numeric
        "skill_foo",  # single segment, ends in alphanumeric
    ],
)
def test_regex_matches_valid_skill_names(text: str) -> None:
    matches = _SKILL_TOOL_PATTERN.findall(text)
    assert text in matches, f"Expected to match {text!r}, got {matches}"


@pytest.mark.parametrize(
    "text",
    [
        "skill_",  # only prefix
        "skill_foo_",  # trailing underscore
        "skill_FOO",  # uppercase (normalized to lower)
        "Skill_foo_bar",  # uppercase prefix
    ],
)
def test_regex_does_not_match_invalid(text: str) -> None:
    matches = _SKILL_TOOL_PATTERN.findall(text)
    assert text not in matches, f"Expected NOT to match {text!r}, got {matches}"


def test_regex_extracts_from_prose() -> None:
    """Realistic case: skill names appear inline in markdown text."""
    text = "Use `skill_foo_bar` to do X. Or call `skill_baz__qux_action`."
    matches = set(_SKILL_TOOL_PATTERN.findall(text))
    assert "skill_foo_bar" in matches
    assert "skill_baz__qux_action" in matches


def test_regex_no_false_positives_on_pure_prose() -> None:
    """Words that look skill-ish but lack the prefix structure."""
    text = "The skill is great. The skills work. skillful coding."
    matches = _SKILL_TOOL_PATTERN.findall(text)
    assert matches == [], f"Unexpected matches: {matches}"


# ---- _assert_no_dangling_skill_tool_refs -------------------------------- #


def test_invariant_passes_when_text_has_no_skill_refs() -> None:
    _assert_no_dangling_skill_tool_refs(
        rendered_text="No skill tool references here at all.",
        declared_bound_tool_names=frozenset(),
        section_id="test_section",
    )


def test_invariant_passes_when_referenced_tool_is_bound() -> None:
    _assert_no_dangling_skill_tool_refs(
        rendered_text="Use `skill_foo_bar` for the task.",
        declared_bound_tool_names=frozenset({"skill_foo_bar"}),
        section_id="test_section",
    )


def test_invariant_fails_when_referenced_tool_is_unbound() -> None:
    with pytest.raises(ToolBindingInvariantError) as exc_info:
        _assert_no_dangling_skill_tool_refs(
            rendered_text="Use `skill_dangling_ref`.",
            declared_bound_tool_names=frozenset({"skill_other"}),
            section_id="my_section",
        )
    assert "my_section" in str(exc_info.value)
    assert "skill_dangling_ref" in str(exc_info.value)


def test_invariant_double_underscore_tool_name_passes() -> None:
    """Edge case: skill_{slug}_{tool} with slug containing repeated separators."""
    _assert_no_dangling_skill_tool_refs(
        rendered_text="Tool `skill_foo__bar_action` is available.",
        declared_bound_tool_names=frozenset({"skill_foo__bar_action"}),
        section_id="test_section",
    )


def test_invariant_double_underscore_unbound_fails() -> None:
    with pytest.raises(ToolBindingInvariantError) as exc_info:
        _assert_no_dangling_skill_tool_refs(
            rendered_text="Tool `skill_foo__bar_action` is available.",
            declared_bound_tool_names=frozenset({"skill_other"}),
            section_id="test",
        )
    assert "skill_foo__bar_action" in str(exc_info.value)


def test_invariant_lists_all_missing_tools_sorted() -> None:
    with pytest.raises(ToolBindingInvariantError) as exc_info:
        _assert_no_dangling_skill_tool_refs(
            rendered_text="Try `skill_zeta_one`, `skill_alpha_two`, `skill_beta_three`.",
            declared_bound_tool_names=frozenset(),
            section_id="test",
        )
    msg = str(exc_info.value)
    # Should be sorted alphabetically in the error
    assert msg.index("skill_alpha_two") < msg.index("skill_beta_three") < msg.index(
        "skill_zeta_one"
    )


def test_invariant_partial_overlap() -> None:
    """Some refs are bound, some aren't — only the unbound ones get reported."""
    with pytest.raises(ToolBindingInvariantError) as exc_info:
        _assert_no_dangling_skill_tool_refs(
            rendered_text="`skill_bound_one` and `skill_unbound_two` are referenced.",
            declared_bound_tool_names=frozenset({"skill_bound_one"}),
            section_id="test",
        )
    msg = str(exc_info.value)
    assert "skill_unbound_two" in msg
    assert "skill_bound_one" not in msg.split("not in declared")[1].split(".")[0]
