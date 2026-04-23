from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.services.provider_profiles._base import ProviderProfile
from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites
from app.application.errors.exceptions import InternalError


def _profile(**overrides) -> ProviderProfile:
    defaults = dict(
        provider_id="x", human_name="X",
        default_api_mode="chat_completions", api_mode_fallback_enabled=False,
    )
    defaults.update(overrides)
    return ProviderProfile(**defaults)


def test_apply_outbound_rewrites_returns_tuple_of_three() -> None:
    result = apply_outbound_rewrites(
        [HumanMessage("hi")], {}, _profile(), is_chat_completions_api=True,
    )
    assert len(result) == 3
    messages, kwargs, warnings = result
    assert isinstance(messages, list)
    assert isinstance(kwargs, dict)
    assert isinstance(warnings, list)


def test_deep_copy_messages_not_mutated() -> None:
    """R3: do not pollute LangGraph state."""
    msg = AIMessage(content="a", additional_kwargs={"reasoning_content": "think"})
    original = deepcopy(msg)
    messages, _, _ = apply_outbound_rewrites(
        [HumanMessage("q"), msg],
        {}, _profile(supports_thinking=True, reasoning_echo_across_user_turns=False),
        is_chat_completions_api=True,
    )
    assert msg.additional_kwargs == original.additional_kwargs


def test_strip_forbidden_sampling_params_emits_warning() -> None:
    p = _profile(forbidden_sampling_params=frozenset({"logprobs"}))
    _, kwargs, warnings = apply_outbound_rewrites(
        [HumanMessage("q")], {"logprobs": True}, p, is_chat_completions_api=True,
    )
    assert "logprobs" not in kwargs
    codes = [w.code for w in warnings]
    assert "sampling_forbidden/logprobs" in codes
    w = next(w for w in warnings if w.code == "sampling_forbidden/logprobs")
    assert w.level == "warning"


def test_strip_silently_ignored_emits_debug_warning() -> None:
    p = _profile(silently_ignored_sampling_params=frozenset({"temperature"}))
    _, kwargs, warnings = apply_outbound_rewrites(
        [HumanMessage("q")], {"temperature": 0.7}, p, is_chat_completions_api=True,
    )
    assert "temperature" not in kwargs
    w = next(w for w in warnings if w.code == "sampling_ignored/temperature")
    assert w.level == "debug"


def test_strip_cross_turn_reasoning_content_for_deepseek_style() -> None:
    """T5 helper-level: DeepSeek style — echo_across_user_turns=False 剥离跨轮 reasoning_content."""
    p = _profile(supports_thinking=True, reasoning_echo_across_user_turns=False)
    msg1 = AIMessage(content="a1", additional_kwargs={"reasoning_content": "think1"})
    msgs = [HumanMessage("q1"), msg1, HumanMessage("q2")]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    deep_ai = rewritten[1]
    assert "reasoning_content" not in deep_ai.additional_kwargs


def test_keep_reasoning_content_in_tool_loop() -> None:
    """T6 helper-level: tool-loop 内保留."""
    p = _profile(supports_thinking=True, reasoning_echo_across_user_turns=False)
    msg_a = AIMessage(
        content="", additional_kwargs={"reasoning_content": "think1"},
        tool_calls=[{"id": "c1", "name": "f", "args": {}}],
    )
    msg_b = AIMessage(
        content="a1", additional_kwargs={"reasoning_content": "think2"},
        tool_calls=[{"id": "c2", "name": "f", "args": {}}],
    )
    msgs = [
        HumanMessage("q1"),
        msg_a,
        ToolMessage(content="r1", tool_call_id="c1"),
        msg_b,
        ToolMessage(content="r2", tool_call_id="c2"),
    ]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    assert rewritten[1].additional_kwargs.get("reasoning_content") == "think1"
    assert rewritten[3].additional_kwargs.get("reasoning_content") == "think2"


def test_keep_reasoning_when_echo_across_user_turns_true() -> None:
    """T2 Kimi style: echo_across_user_turns=True 全部保留."""
    p = _profile(supports_thinking=True, reasoning_echo_across_user_turns=True)
    msg = AIMessage(content="a", additional_kwargs={"reasoning_content": "think"})
    msgs = [HumanMessage("q1"), msg, HumanMessage("q2")]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    assert rewritten[1].additional_kwargs["reasoning_content"] == "think"


def test_responses_api_skip_reasoning_strip() -> None:
    """T19 helper-level: is_chat_completions_api=False 跳过 reasoning_content 处理."""
    p = _profile(supports_thinking=True, reasoning_echo_across_user_turns=False)
    msg = AIMessage(content="a", additional_kwargs={"reasoning_content": "think"})
    msgs = [HumanMessage("q1"), msg, HumanMessage("q2")]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=False)
    assert rewritten[1].additional_kwargs.get("reasoning_content") == "think"


def test_raise_internal_error_on_https_url_when_profile_forbids() -> None:
    """T3b: rewrite 层 fail-fast, not log silently."""
    p = _profile(accepts_image_url=False)
    msg = HumanMessage(content=[
        {"type": "text", "text": "see this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/pic.png"}},
    ])
    with pytest.raises(InternalError, match="accepts_image_url=False"):
        apply_outbound_rewrites([msg], {}, p, is_chat_completions_api=True)


def test_data_url_passes_assertion() -> None:
    """base64 data: URL 不触发断言."""
    p = _profile(accepts_image_url=False)
    msg = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ])
    messages, _, _ = apply_outbound_rewrites([msg], {}, p, is_chat_completions_api=True)
    assert messages is not None
