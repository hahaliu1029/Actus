"""Google Gemini OpenAI-compat profile (Spec §5.4).

Target models: gemini-2.5-flash 家族 (allowlist prefix covers -8b / -latest / dated)
  — thinking togglable via reasoning_effort="none".

不覆盖（heuristic 路由到 generic_openai）：
- Gemini 2.5 Pro — thinking 不可关 (future gemini_compat_pro P2)
- Gemini 3 家族 — reasoning_effort="medium" 400 + delta.index=None streaming bug
  (future gemini_3_compat P2 with Thinking MEDIUM fingerprint)

Docs source:
- OpenAI compatibility: https://ai.google.dev/gemini-api/docs/openai
- Thinking: https://ai.google.dev/gemini-api/docs/thinking
- Gemini 3 Dev Guide: https://ai.google.dev/gemini-api/docs/gemini-3
- Gemini 3 Preview MEDIUM bug: https://discuss.ai.google.dev/t/gemini-3-preview-openai-compatible-rejects-reasoning-effort-medium/112648
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


GEMINI_COMPAT_PROFILE = ProviderProfile(
    provider_id="gemini_compat",
    human_name="Google Gemini (OpenAI-compat)",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,   # Responses endpoint returns 404
    tool_choice_any_alias="auto",      # docs only showcase "auto"; "required" undocumented
    tool_choice_forbidden_when_thinking=frozenset(),
    emits_tool_calls_in_content=False,
    supports_thinking=True,
    thinking_always_on=False,          # 2.5 Flash can disable via reasoning_effort="none"
    thinking_toggle_style="openai_reasoning_effort",
    reasoning_content_field_name="reasoning_content",   # L confidence: compat layer doesn't commit
    reasoning_echo_in_tool_loop=False,  # Gemini 3 thought_signature out of scope
    reasoning_echo_across_user_turns=False,
    supports_thinking_with_tools=True,
    accepts_image_url=True,
    accepts_image_base64=True,
    image_max_bytes=5 * 1024 * 1024,
    supports_vision=True,
    supports_pdf_input=False,
    silently_ignored_sampling_params=frozenset({
        "logit_bias", "parallel_tool_calls",
        "frequency_penalty", "presence_penalty",
    }),
    forbidden_sampling_params=frozenset({"logprobs", "top_logprobs"}),
    supports_response_format_json_object=True,
    supports_response_format_json_schema=True,
    response_format_silently_ignored=False,
    error_fingerprints=(
        # NOTE: Gemini 3 "Thinking level MEDIUM" fingerprint intentionally
        # removed (codex Fact #3) — model-specific, not 2.5 Flash. Reserved
        # for future gemini_3_compat sub-profile.
        ErrorFingerprint(400, "Unknown name 'logprobs'", ErrorClass.COMPAT_QUIRK),
        ErrorFingerprint(400, "exceeds the maximum number of tokens", ErrorClass.CONTEXT_OVERFLOW),
        ErrorFingerprint(429, "RESOURCE_EXHAUSTED", ErrorClass.TRANSIENT_RATE_LIMIT),
        ErrorFingerprint(503, "UNAVAILABLE", ErrorClass.TRANSIENT_CONNECTION),
        ErrorFingerprint(400, "SAFETY", ErrorClass.PERMANENT_4XX),
    ),
    default_context_window=1_000_000,
    default_max_output_tokens=8_192,
    downgrade_targets=(),
)

register_profile(GEMINI_COMPAT_PROFILE)
