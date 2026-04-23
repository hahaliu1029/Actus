"""DeepSeek Reasoner profile. spec §4.5.

关键语义差异：
- reasoning_echo_in_tool_loop=True：工具轮内多条 AIMessage 都保留 reasoning_content
- reasoning_echo_across_user_turns=False：跨用户轮必须剥离（V3.2 回归后严格要求）
- silently_ignored_sampling_params: temperature/top_p/presence/frequency 会被静默忽略
- forbidden_sampling_params: logprobs/top_logprobs 返回 400
- supports_response_format_json_schema=False: 只有 /beta 路径支持 schema
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


DEEPSEEK_REASONER_PROFILE = ProviderProfile(
    provider_id="deepseek_reasoner",
    human_name="DeepSeek Reasoner",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,
    tool_choice_any_alias="required",
    tool_choice_forbidden_when_thinking=frozenset(),
    emits_tool_calls_in_content=False,
    supports_thinking=True,
    thinking_always_on=True,
    thinking_toggle_style="none",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=True,
    reasoning_echo_across_user_turns=False,
    supports_thinking_with_tools=True,
    accepts_image_url=False,
    accepts_image_base64=False,
    supports_vision=False,
    silently_ignored_sampling_params=frozenset(
        {"temperature", "top_p", "presence_penalty", "frequency_penalty"}
    ),
    forbidden_sampling_params=frozenset({"logprobs", "top_logprobs"}),
    supports_response_format_json_object=True,
    supports_response_format_json_schema=False,
    error_fingerprints=(
        ErrorFingerprint(400, "Missing reasoning_content", ErrorClass.COMPAT_QUIRK),
        ErrorFingerprint(400, "reasoning_content", ErrorClass.COMPAT_QUIRK),
    ),
    default_context_window=128_000,
    default_max_output_tokens=32_768,
    # Must match a registered provider_id (underscore, not hyphen).
    downgrade_targets=("deepseek_chat",),
)

register_profile(DEEPSEEK_REASONER_PROFILE)
