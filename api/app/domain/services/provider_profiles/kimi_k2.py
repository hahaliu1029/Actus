"""Moonshot Kimi K2 / K2.5 profile. spec §4.5 (A7 review P0).

A7 review P2 baseline: Moonshot 2026-04 官方文档未记录采样参数限制，两集合均为空是
baseline 反映而非遗漏。T4 (a) 段显式断言这一 baseline；若未来文档更新，更新 profile
会自动激活 T4 (b) 段的合成 fixture 行为到 Kimi 直测路径。
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


KIMI_K2_PROFILE = ProviderProfile(
    provider_id="kimi_k2",
    human_name="Moonshot Kimi K2/K2.5",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,
    tool_choice_any_alias="auto",
    tool_choice_forbidden_when_thinking=frozenset({"required"}),
    emits_tool_calls_in_content=False,
    supports_thinking=True,
    thinking_always_on=True,
    thinking_toggle_style="none",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=True,
    reasoning_echo_across_user_turns=True,
    supports_thinking_with_tools=True,
    accepts_image_url=False,
    accepts_image_base64=True,
    image_max_bytes=100 * 1024 * 1024,
    supports_vision=True,
    silently_ignored_sampling_params=frozenset(),
    forbidden_sampling_params=frozenset(),
    supports_response_format_json_object=True,
    supports_response_format_json_schema=True,
    error_fingerprints=(
        ErrorFingerprint(
            code="kimi_missing_reasoning_content",
            status_code=400,
            body_substring="Missing reasoning_content",
            error_class=ErrorClass.COMPAT_QUIRK,
        ),
        ErrorFingerprint(
            code="kimi_rate_limit_reached",
            status_code=429,
            body_substring="rate_limit_reached_for_model",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
    ),
    default_context_window=128_000,
    default_max_output_tokens=32_768,
    downgrade_targets=(),
)

register_profile(KIMI_K2_PROFILE)
