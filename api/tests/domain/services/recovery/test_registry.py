import pytest

from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.recovery._registry import (
    RECOVERY_RULES,
    _validate_rules_unique,
    match_rule,
    register_rule,
)


class _FakeAction:
    code = "fake_action"

    async def apply(self, messages, kwargs, ctx):
        return messages, kwargs


class _OtherAction:
    code = "other"

    async def apply(self, messages, kwargs, ctx):
        return messages, kwargs


@pytest.fixture(autouse=True)
def _clear_rules():
    saved = dict(RECOVERY_RULES)
    RECOVERY_RULES.clear()
    yield
    RECOVERY_RULES.clear()
    RECOVERY_RULES.update(saved)


def test_register_rule_stores_key_value():
    register_rule(
        ("p1", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp1"),
        (_FakeAction(),),
    )
    assert len(RECOVERY_RULES) == 1


def test_register_rule_idempotent_same_value_overwrite_ok():
    a = _FakeAction()
    register_rule(("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"), (a,))
    register_rule(("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"), (a,))
    assert len(RECOVERY_RULES) == 1


def test_register_rule_conflict_raises():
    register_rule(
        ("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"),
        (_FakeAction(),),
    )
    with pytest.raises(Exception):
        register_rule(
            ("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"),
            (_OtherAction(),),
        )


def test_match_rule_exact_match():
    register_rule(
        ("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"),
        (_FakeAction(),),
    )
    matched = match_rule("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp")
    assert matched is not None
    assert matched[0].code == "fake_action"


def test_match_rule_none_fingerprint_code_falls_back_to_wildcard():
    register_rule(
        ("*", "*", ErrorClass.CONTEXT_OVERFLOW, "*"),
        (_FakeAction(),),
    )
    matched = match_rule("anyprof", "chat_completions", ErrorClass.CONTEXT_OVERFLOW, None)
    assert matched is not None


def test_match_rule_no_match_returns_none():
    register_rule(
        ("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"),
        (_FakeAction(),),
    )
    assert match_rule("other", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp") is None


def test_match_rule_priority_exact_over_wildcard():
    exact_action = _FakeAction()

    class _Wild:
        code = "wild"

        async def apply(self, m, k, c):
            return m, k

    register_rule(("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"), (exact_action,))
    register_rule(("*", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"), (_Wild(),))
    matched = match_rule("p", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp")
    assert matched[0].code == "fake_action"


def test_validate_rules_unique_same_priority_duplicate_raises():
    class _A:
        code = "a"

        async def apply(self, m, k, c):
            return m, k

    class _B:
        code = "b"

        async def apply(self, m, k, c):
            return m, k

    # Two single-wildcard keys both match ("p", "chat_completions", COMPAT_QUIRK, "fp")
    register_rule(("*", "chat_completions", ErrorClass.COMPAT_QUIRK, "fp"), (_A(),))
    register_rule(("p", "*", ErrorClass.COMPAT_QUIRK, "fp"), (_B(),))
    with pytest.raises(Exception):
        _validate_rules_unique()
