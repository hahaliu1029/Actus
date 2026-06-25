import pytest

from app.application.services.coordinator_child_runner_starter import (
    _decode_member_skill_slugs,
    _decode_member_skill_tools,
)


def test_missing_key_tolerant():
    assert _decode_member_skill_tools({}) == frozenset()
    assert _decode_member_skill_slugs({}) == ()


def test_present_list_of_str_decoded():
    assert _decode_member_skill_tools({"member_skill_tools": ["skill_a_x"]}) == frozenset({"skill_a_x"})
    assert _decode_member_skill_slugs({"member_skill_slugs": ["repo-map"]}) == ("repo-map",)


def test_bare_string_rejected_not_iterated_into_chars():
    # the corrupt-bind-set trap: frozenset("abc") → {'a','b','c'}
    with pytest.raises(ValueError):
        _decode_member_skill_tools({"member_skill_tools": "skill_a_x"})
    with pytest.raises(ValueError):
        _decode_member_skill_slugs({"member_skill_slugs": "repo-map"})


def test_non_str_elements_rejected():
    with pytest.raises(ValueError):
        _decode_member_skill_tools({"member_skill_tools": ["ok", 7]})
