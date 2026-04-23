from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._rewrites import build_sdk_params


def _profile(**overrides) -> ProviderProfile:
    defaults = dict(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    defaults.update(overrides)
    return ProviderProfile(**defaults)


def test_stripped_keys_beat_adapter_defaults() -> None:
    p = _profile(
        provider_id="deepseek_reasoner",
        silently_ignored_sampling_params=frozenset(
            {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
        ),
    )
    params = build_sdk_params(
        rewritten_kwargs={},
        profile=p,
        adapter_defaults={"temperature": 0.7, "max_tokens": 8192},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
    )
    assert "temperature" not in params
    assert params["max_tokens"] == 8192


def test_rewritten_kwargs_overrides_defaults() -> None:
    p = _profile()
    params = build_sdk_params(
        rewritten_kwargs={"max_tokens": 4096},
        profile=p,
        adapter_defaults={"temperature": 0.7, "max_tokens": 8192},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
    )
    assert params["max_tokens"] == 4096


def test_response_format_set_when_resolved() -> None:
    p = _profile()
    params = build_sdk_params(
        rewritten_kwargs={}, profile=p, adapter_defaults={},
        base_params={"model": "m", "messages": []},
        resolved_response_format={"type": "json_object"},
    )
    assert params["response_format"] == {"type": "json_object"}

    params2 = build_sdk_params(
        rewritten_kwargs={}, profile=p, adapter_defaults={},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
    )
    assert "response_format" not in params2


def test_resolved_tool_choice_set_when_not_none() -> None:
    p = _profile()
    params = build_sdk_params(
        rewritten_kwargs={}, profile=p, adapter_defaults={},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
        resolved_tool_choice="auto",
    )
    assert params["tool_choice"] == "auto"

    params2 = build_sdk_params(
        rewritten_kwargs={}, profile=p, adapter_defaults={},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
        resolved_tool_choice=None,
    )
    assert "tool_choice" not in params2


def test_no_strip_profile_defaults_flow_through() -> None:
    p = _profile()
    params = build_sdk_params(
        rewritten_kwargs={}, profile=p,
        adapter_defaults={"temperature": 0.7, "max_tokens": 8192},
        base_params={"model": "m", "messages": []},
        resolved_response_format=None,
    )
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 8192
