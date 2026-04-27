"""PR-2: rules registered at package import time (T27b / T36 / T37)."""
from app.domain.services.provider_profiles._base import ErrorClass


def test_T36_recovery_package_import_triggers_register_all():
    import app.domain.services.recovery  # noqa: F401
    from app.domain.services.recovery._registry import RECOVERY_RULES
    assert len(RECOVERY_RULES) >= 4


def test_R1_registered():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    key = ("dashscope_qwen", "chat_completions", ErrorClass.COMPAT_QUIRK, "json_mode_with_thinking")
    assert key in RECOVERY_RULES
    codes = [a.code for a in RECOVERY_RULES[key]]
    assert codes == ["strip_response_format", "disable_thinking"]


def test_R2_registered():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    key = ("dashscope_qwen", "chat_completions", ErrorClass.COMPAT_QUIRK, "tool_choice_string_forbidden")
    assert key in RECOVERY_RULES


def test_R3_registered():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    key = ("anthropic_compat", "chat_completions", ErrorClass.COMPAT_QUIRK, "thinking_forbidden_with_tool_choice")
    assert key in RECOVERY_RULES


def test_R4_wildcard_registered():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    key = ("*", "*", ErrorClass.CONTEXT_OVERFLOW, "*")
    assert key in RECOVERY_RULES


def test_T31_R4_wildcard_matches_any_profile_api_mode():
    """Audit Round 6 P2 #2 fix: spec T31 requires match_rule to actually
    resolve the wildcard R4 key across any profile_id, the 'responses' api_mode
    (§2 N5 the sole cross-protocol exception), AND fingerprint_code=None
    (classify_error_diagnostic falling through to status-code fallback).
    """
    from app.domain.services.recovery._registry import match_rule
    from app.domain.services.recovery._actions import TriggerRecompact

    # Arbitrary profile_id + Chat api_mode + concrete fingerprint_code
    matched = match_rule(
        "dashscope_qwen", "chat_completions", ErrorClass.CONTEXT_OVERFLOW, "range_of_input_length",
    )
    assert matched is not None and any(isinstance(a, TriggerRecompact) for a in matched)

    # Arbitrary profile_id + Responses api_mode (cross-protocol R4 exception)
    matched = match_rule(
        "openai_official", "responses", ErrorClass.CONTEXT_OVERFLOW, "context_length_exceeded",
    )
    assert matched is not None and any(isinstance(a, TriggerRecompact) for a in matched)

    # fingerprint_code=None (diagnostic had no fingerprint hit) must still wildcard-match
    matched = match_rule("anonymous_profile", "chat_completions", ErrorClass.CONTEXT_OVERFLOW, None)
    assert matched is not None and any(isinstance(a, TriggerRecompact) for a in matched)


def test_T27b_no_anthropic_thinking_rejected_rule_registered():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    for (profile_id, _api_mode, _err, fp_code) in RECOVERY_RULES:
        assert not (profile_id == "anthropic_compat" and fp_code == "thinking_rejected"), (
            "No rule for anthropic_compat + thinking_rejected (Opus 4.7 routes to "
            "generic_openai — spec §13 P3 follow-up)."
        )


def test_T37_register_all_idempotent():
    from app.domain.services.recovery._rules import register_all
    register_all()  # already called by __init__
    register_all()  # second call must not raise
