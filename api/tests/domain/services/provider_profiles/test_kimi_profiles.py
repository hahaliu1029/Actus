from dataclasses import replace

from langchain_core.messages import AIMessage, HumanMessage

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._parse import parse_chat_completion_message
from app.domain.services.provider_profiles._rewrites import apply_outbound_rewrites
from app.domain.services.provider_profiles._wire import inject_reasoning_into_wire_entry
import app.domain.services.provider_profiles.kimi_k2  # triggers register
import app.domain.services.provider_profiles.kimi_k2_6  # triggers register


def test_kimi_k2_profile_declarations() -> None:
    """T1 (a)-(c): spec §4.5 Kimi K2 profile"""
    p = get_profile("kimi_k2")
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset({"required"})
    assert p.thinking_always_on is True
    assert p.supports_thinking is True
    assert p.reasoning_content_field_name == "reasoning_content"
    assert p.reasoning_echo_in_tool_loop is True
    assert p.reasoning_echo_across_user_turns is True
    assert p.accepts_image_url is False
    assert p.accepts_image_base64 is True
    assert p.image_max_bytes == 100 * 1024 * 1024
    assert p.api_mode_fallback_enabled is False
    # T4 (a) baseline
    assert p.forbidden_sampling_params == frozenset()
    assert p.silently_ignored_sampling_params == frozenset()
    # T9: fingerprint 覆盖 Missing reasoning_content
    codes = [fp for fp in p.error_fingerprints
             if fp.body_substring and "reasoning_content" in fp.body_substring.lower()]
    assert any(fp.error_class == ErrorClass.COMPAT_QUIRK for fp in codes)


def test_kimi_k2_6_profile_renames_reasoning_field() -> None:
    """K2.6 replace(): reasoning_content_field_name="reasoning" 其他字段同 K2"""
    p = get_profile("kimi_k2_6")
    assert p.provider_id == "kimi_k2_6"
    assert p.reasoning_content_field_name == "reasoning"
    # 其他 Kimi 语义保持
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset({"required"})
    assert p.accepts_image_url is False
    assert p.reasoning_echo_across_user_turns is True


def test_kimi_k2_reasoning_roundtrip() -> None:
    """T2: echo_across_user_turns=True → reasoning_content 跨轮保留"""
    p = get_profile("kimi_k2")
    msg = AIMessage(content="a", additional_kwargs={"reasoning_content": "think"})
    msgs = [HumanMessage("q1"), msg, HumanMessage("q2")]
    rewritten, _, _ = apply_outbound_rewrites(msgs, {}, p, is_chat_completions_api=True)
    assert rewritten[1].additional_kwargs["reasoning_content"] == "think"


def test_kimi_k2_inbound_parse() -> None:
    """T21 with real profile: wire key == 'reasoning_content'"""
    p = get_profile("kimi_k2")
    raw = {"content": "a", "reasoning_content": "think"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {"reasoning_content": "think"}


def test_kimi_k2_6_wire_field_rename() -> None:
    """T22 with real profile: wire 'reasoning' → internal 'reasoning_content'"""
    p = get_profile("kimi_k2_6")
    raw = {"content": "a", "reasoning": "think"}
    kw = parse_chat_completion_message(raw, p)
    assert kw == {"reasoning_content": "think"}


def test_kimi_k2_6_outbound_wire_inject() -> None:
    """T26: K2.6 出 wire 使用 'reasoning' key；inbound→outbound 闭环"""
    p = get_profile("kimi_k2_6")
    entry = {"role": "assistant", "content": "a"}
    inject_reasoning_into_wire_entry(
        entry, {"reasoning_content": "think"}, p, is_chat_completions_api=True,
    )
    assert entry["reasoning"] == "think"
    assert "reasoning_content" not in entry


def test_kimi_k2_forbidden_sampling_baseline() -> None:
    """T4 (a): Moonshot 2026-04 官方未记录限制 — 两集合为空是 baseline 反映"""
    p = get_profile("kimi_k2")
    assert p.forbidden_sampling_params == frozenset()
    assert p.silently_ignored_sampling_params == frozenset()


def test_forbidden_sampling_hook_via_fixture_profile() -> None:
    """T4 (b): 合成 fixture profile（不绑 Kimi）验证 hook 行为"""
    base = get_profile("kimi_k2")
    fixture = replace(base, forbidden_sampling_params=frozenset({"logprobs"}))
    _, kwargs, warnings = apply_outbound_rewrites(
        [HumanMessage("q")], {"logprobs": True}, fixture, is_chat_completions_api=True,
    )
    assert "logprobs" not in kwargs
    codes = [w.code for w in warnings]
    assert "sampling_forbidden/logprobs" in codes
