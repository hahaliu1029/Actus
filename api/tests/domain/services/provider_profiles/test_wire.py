from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._wire import inject_reasoning_into_wire_entry


def _profile(**overrides) -> ProviderProfile:
    defaults = dict(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    defaults.update(overrides)
    return ProviderProfile(**defaults)


def test_inject_kimi_k2() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning_content")
    entry = {"role": "assistant", "content": "hi", "tool_calls": []}
    result = inject_reasoning_into_wire_entry(
        entry, {"reasoning_content": "think"}, p,
        is_chat_completions_api=True,
    )
    assert result["reasoning_content"] == "think"


def test_inject_kimi_k2_6_maps_to_reasoning() -> None:
    p = _profile(supports_thinking=True, reasoning_content_field_name="reasoning")
    entry = {"role": "assistant", "content": "hi"}
    result = inject_reasoning_into_wire_entry(
        entry, {"reasoning_content": "think"}, p,
        is_chat_completions_api=True,
    )
    assert result["reasoning"] == "think"
    assert "reasoning_content" not in result


def test_inject_skipped_when_not_thinking() -> None:
    p = _profile(supports_thinking=False)
    entry = {"role": "assistant", "content": "hi"}
    result = inject_reasoning_into_wire_entry(
        entry, {"reasoning_content": "think"}, p,
        is_chat_completions_api=True,
    )
    assert "reasoning_content" not in result
    assert "reasoning" not in result


def test_inject_skipped_when_additional_kwargs_empty() -> None:
    p = _profile(supports_thinking=True)
    entry = {"role": "assistant", "content": "hi"}
    result = inject_reasoning_into_wire_entry(entry, {}, p, is_chat_completions_api=True)
    assert "reasoning_content" not in result


def test_inject_skipped_for_responses_api() -> None:
    p = _profile(supports_thinking=True)
    entry = {"role": "assistant", "content": "hi"}
    result = inject_reasoning_into_wire_entry(
        entry, {"reasoning_content": "think"}, p,
        is_chat_completions_api=False,
    )
    assert "reasoning_content" not in result
    assert "reasoning" not in result
