from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._rewrites import resolve_tool_choice


def _kimi() -> ProviderProfile:
    return ProviderProfile(
        provider_id="kimi_k2", human_name="Kimi K2",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
        tool_choice_any_alias="auto",
        tool_choice_forbidden_when_thinking=frozenset({"required"}),
        supports_thinking=True, thinking_always_on=True,
    )


def _deepseek_reasoner() -> ProviderProfile:
    return ProviderProfile(
        provider_id="deepseek_reasoner", human_name="DeepSeek Reasoner",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
        tool_choice_any_alias="required",
        tool_choice_forbidden_when_thinking=frozenset(),
        supports_thinking=True, thinking_always_on=True,
    )


def test_per_call_any_step_a_normalization() -> None:
    v, w = resolve_tool_choice("any", None, _kimi(), thinking_enabled=True)
    assert v == "auto"
    assert w == []


def test_per_call_required_step_b_rewrite_with_warning() -> None:
    v, w = resolve_tool_choice("required", None, _kimi(), thinking_enabled=True)
    assert v == "auto"
    assert len(w) == 1
    assert w[0].code == "tool_choice_forbidden/required"
    assert w[0].context == {"original": "required", "rewritten_to": "auto"}
    assert w[0].level == "warning"


def test_bound_required_triggers_same_rule() -> None:
    v, w = resolve_tool_choice(None, "required", _kimi(), thinking_enabled=True)
    assert v == "auto"
    assert len(w) == 1
    assert w[0].code == "tool_choice_forbidden/required"


def test_per_call_wins_over_bound_when_not_forbidden() -> None:
    v, w = resolve_tool_choice("none", "required", _kimi(), thinking_enabled=True)
    assert v == "none"
    assert w == []


def test_both_none_returns_none() -> None:
    v, w = resolve_tool_choice(None, None, _kimi(), thinking_enabled=True)
    assert v is None
    assert w == []


def test_deepseek_required_passes_through() -> None:
    v, w = resolve_tool_choice(None, "required", _deepseek_reasoner(),
                                thinking_enabled=True)
    assert v == "required"
    assert w == []


def test_dict_form_passthrough() -> None:
    d = {"type": "function", "function": {"name": "foo"}}
    v, w = resolve_tool_choice(d, None, _kimi(), thinking_enabled=True)
    assert v == d
    assert w == []


def test_dict_form_via_bound_passthrough() -> None:
    d = {"type": "function", "function": {"name": "foo"}}
    v, w = resolve_tool_choice(None, d, _kimi(), thinking_enabled=True)
    assert v == d
    assert w == []


def test_multiple_calls_same_warning_code_nondedup() -> None:
    for _ in range(3):
        _, w = resolve_tool_choice("required", None, _kimi(), thinking_enabled=True)
        assert len(w) == 1
        assert w[0].code == "tool_choice_forbidden/required"


def test_resolve_tool_choice_with_real_kimi_profile_required() -> None:
    """T1b (c): bound 'required' + 真实 Kimi profile → 触发 forbidden-when-thinking"""
    profile = get_profile("kimi_k2")
    v, w = resolve_tool_choice(None, "required", profile, thinking_enabled=True)
    assert v == "auto"
    assert len(w) == 1
    assert w[0].code == "tool_choice_forbidden/required"
    assert w[0].context == {"original": "required", "rewritten_to": "auto"}


def test_resolve_tool_choice_with_real_k2_6_profile() -> None:
    """K2.6 同规则"""
    profile = get_profile("kimi_k2_6")
    v, w = resolve_tool_choice(None, "required", profile, thinking_enabled=True)
    assert v == "auto"
    assert len(w) == 1
