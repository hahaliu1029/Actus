from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites
import app.domain.services.provider_profiles.deepseek_reasoner  # noqa
import app.domain.services.provider_profiles.deepseek_chat  # noqa


# ---------- Task 3.1 profile declarations ----------


def test_deepseek_reasoner_profile_declarations() -> None:
    p = get_profile("deepseek_reasoner")
    assert p.supports_thinking is True
    assert p.thinking_always_on is True
    assert p.reasoning_content_field_name == "reasoning_content"
    assert p.reasoning_echo_in_tool_loop is True
    assert p.reasoning_echo_across_user_turns is False
    assert p.supports_thinking_with_tools is True
    assert p.accepts_image_url is False
    assert p.accepts_image_base64 is False
    assert p.supports_vision is False
    assert p.silently_ignored_sampling_params == frozenset(
        {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
    )
    assert p.forbidden_sampling_params == frozenset({"logprobs", "top_logprobs"})
    assert p.supports_response_format_json_object is True
    assert p.supports_response_format_json_schema is False
    assert p.tool_choice_any_alias == "required"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    assert p.downgrade_targets == ("deepseek_chat",)
    # P3: downgrade_targets ids must resolve through the registry.
    from app.domain.services.provider_profiles._registry import get_profile as _gp
    for target_id in p.downgrade_targets:
        assert _gp(target_id).provider_id == target_id
    substrs = [fp.body_substring for fp in p.error_fingerprints
               if fp.body_substring]
    assert any("Missing reasoning_content" in s for s in substrs)


def test_deepseek_chat_profile() -> None:
    p = get_profile("deepseek_chat")
    assert p.supports_thinking is False
    assert p.reasoning_echo_in_tool_loop is False
    assert p.reasoning_echo_across_user_turns is False
    assert p.silently_ignored_sampling_params == frozenset()
    assert p.forbidden_sampling_params == frozenset({"logprobs", "top_logprobs"})
    assert p.downgrade_targets == ()


# ---------- Task 3.2 cross-turn strip + tool-loop preserve ----------


def test_deepseek_reasoner_strips_cross_turn_reasoning_content() -> None:
    """T5: 跨用户轮 AIMessage 的 reasoning_content 必须剥离"""
    p = get_profile("deepseek_reasoner")
    msgs = [
        HumanMessage("q1"),
        AIMessage(content="a1", additional_kwargs={"reasoning_content": "think1"}),
        HumanMessage("q2"),
    ]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    assert "reasoning_content" not in rewritten[1].additional_kwargs


def test_deepseek_reasoner_keeps_tool_loop_reasoning() -> None:
    """T6: 同一用户轮内 tool-loop 的多条 AIMessage 都保留 reasoning_content"""
    p = get_profile("deepseek_reasoner")
    msgs = [
        HumanMessage("q1"),
        AIMessage(content="", additional_kwargs={"reasoning_content": "think1"},
                  tool_calls=[{"id": "c1", "name": "f", "args": {}}]),
        ToolMessage(content="r1", tool_call_id="c1"),
        AIMessage(content="a1", additional_kwargs={"reasoning_content": "think2"},
                  tool_calls=[{"id": "c2", "name": "f", "args": {}}]),
        ToolMessage(content="r2", tool_call_id="c2"),
    ]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    assert rewritten[1].additional_kwargs["reasoning_content"] == "think1"
    assert rewritten[3].additional_kwargs["reasoning_content"] == "think2"


# ---------- Task 3.3 forbidden logprobs + silently_ignored temperature ----------


def test_deepseek_reasoner_strips_forbidden_logprobs() -> None:
    """T7: logprobs → strip + RewriteWarning(level=warning)"""
    p = get_profile("deepseek_reasoner")
    _, kwargs, warnings = apply_outbound_rewrites(
        [HumanMessage("q")], {"logprobs": True}, p, is_chat_completions_api=True,
    )
    assert "logprobs" not in kwargs
    matches = [w for w in warnings if w.code == "sampling_forbidden/logprobs"]
    assert len(matches) == 1
    assert matches[0].level == "warning"


def test_deepseek_reasoner_silently_ignores_temperature() -> None:
    """T8: temperature → strip + RewriteWarning(level=debug)"""
    p = get_profile("deepseek_reasoner")
    _, kwargs, warnings = apply_outbound_rewrites(
        [HumanMessage("q")], {"temperature": 0.3}, p, is_chat_completions_api=True,
    )
    assert "temperature" not in kwargs
    matches = [w for w in warnings if w.code == "sampling_ignored/temperature"]
    assert len(matches) == 1
    assert matches[0].level == "debug"


# Kept imported for symmetry with other profile test modules that reference ErrorClass
_ = ErrorClass
