"""DashScope Qwen text-flagship profile (Spec §5.1).

Target models: qwen-plus / qwen-turbo / qwen-flash / qwen3-max
  (hybrid thinking, togglable, default-off; text-only — no vision.)

不覆盖（heuristic 路由到 generic_openai）：
- qwen-max* (non-thinking only, Round 4 Fact #1)
- QwQ / qwen3-*-thinking-* (always-on)
- qwen3.5-* (hybrid default-on)

Docs source:
- OpenAI-compat reference: https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope
- Deep thinking: https://www.alibabacloud.com/help/en/model-studio/deep-thinking
- Function calling: https://www.alibabacloud.com/help/en/model-studio/qwen-function-calling
- Error codes: https://www.alibabacloud.com/help/en/model-studio/error-code
- Qwen structured output: https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output
- JSON mode: https://www.alibabacloud.com/help/en/model-studio/json-mode
- pydantic-ai#1265 (tool_choice=required 400): https://github.com/pydantic/pydantic-ai/issues/1265
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


DASHSCOPE_QWEN_PROFILE = ProviderProfile(
    provider_id="dashscope_qwen",
    human_name="DashScope Qwen (text)",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,  # Responses-compat endpoint is a different URL path
    tool_choice_any_alias="auto",     # 400 on "required" per pydantic-ai#1265
    tool_choice_forbidden_when_thinking=frozenset(),  # no documented constraint (QwQ would differ)
    emits_tool_calls_in_content=False,
    supports_thinking=True,
    thinking_always_on=False,
    thinking_toggle_style="extra_body_enable_thinking",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=False,
    reasoning_echo_across_user_turns=False,
    supports_thinking_with_tools=True,
    accepts_image_url=False,          # text flagship — vision dispatched to qwen_vl profile
    accepts_image_base64=False,
    image_max_bytes=0,
    supports_vision=False,
    supports_pdf_input=False,
    silently_ignored_sampling_params=frozenset({"logit_bias", "logprobs", "top_logprobs"}),
    forbidden_sampling_params=frozenset(),
    # 官方 structured output FAQ 明确:"Models in thinking mode do not support
    # structured output" —— 以下两个 True **仅对 non-thinking mode 成立**。
    # thinking 模式下传 response_format 会 400 "Json mode response is not
    # supported when enable_thinking is true" → 由 F02 fingerprint 捕获为
    # COMPAT_QUIRK。Finding 1 non-goal: 不扩 schema 加 response_format_
    # forbidden_when_thinking；B2 recovery chain 基于 F02 做 typed strip 恢复。
    supports_response_format_json_object=True,   # non-thinking mode only
    supports_response_format_json_schema=True,   # non-thinking mode only
    response_format_silently_ignored=False,      # thinking mode 返回 400 via F02, 非 silent
    error_fingerprints=(
        ErrorFingerprint(400, "tool_choice is one of the strings", ErrorClass.COMPAT_QUIRK),
        ErrorFingerprint(400, "Json mode response is not supported when enable_thinking is true", ErrorClass.COMPAT_QUIRK),
        ErrorFingerprint(400, "Range of input length should be", ErrorClass.CONTEXT_OVERFLOW),
        ErrorFingerprint(429, "Requests throttling triggered", ErrorClass.TRANSIENT_RATE_LIMIT),
        ErrorFingerprint(429, "Allocated quota exceeded", ErrorClass.PERMANENT_4XX),
        ErrorFingerprint(401, "Invalid API-key provided", ErrorClass.TRANSIENT_AUTH),
        ErrorFingerprint(500, "An internal error has occured", ErrorClass.TRANSIENT_CONNECTION),
    ),
    default_context_window=131_072,
    default_max_output_tokens=8_192,
    downgrade_targets=(),
)

register_profile(DASHSCOPE_QWEN_PROFILE)
