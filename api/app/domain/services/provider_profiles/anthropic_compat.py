"""Anthropic OpenAI-compat profile (Spec §5.3).

Target models: claude-sonnet-4-6 / claude-haiku-4-5 (OpenAI-compat layer).

不覆盖（heuristic 路由到 generic_openai）：
- claude-opus-4-7 — Opus 4.7+ 不接受 manual extra_body.thinking (只 adaptive thinking)
- claude-3-* / claude-4-0..4-5 (legacy, 能力矩阵不一致)

关键（docstring）：
1. 官方文档：Anthropic OpenAI-compat 层 "primarily intended to test and compare
   model capabilities, not a long-term or production-ready solution"；不推荐生产。
2. response_format: Ignored (T24 assumption 确认); PDF / audio / file parts 全部
   silently drop.
3. supports_thinking_with_tools=True (codex Round 2 Fact #1 修正): Anthropic
   官方明确支持 extended thinking + tools，约束仅限 tool_choice (不能用
   required / any / tool-shape)。不要把 "tool_choice 约束" 误当成 "thinking+tools 不支持"。

Docs source:
- OpenAI SDK compatibility: https://platform.claude.com/docs/en/api/openai-sdk
- Extended thinking: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
- Errors: https://platform.claude.com/docs/en/api/errors
- Models overview: https://platform.claude.com/docs/en/about-claude/models/overview
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


ANTHROPIC_COMPAT_PROFILE = ProviderProfile(
    provider_id="anthropic_compat",
    human_name="Anthropic (OpenAI-compat layer)",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,  # no Responses-compat path
    # alias="auto" (not "required"): codex Round 1 — with alias="required" and
    # forbidden_when_thinking={"required"}, resolve_tool_choice's Step B would
    # rewrite "required" back to "required" (loop, no downgrade). alias="auto"
    # lets Step B properly downgrade "required" → "auto" under thinking.
    # Trade-off: outside thinking, "any" input gets aliased to "auto" (loss of
    # OpenAI "must call tool" semantics); explicit "required" still honored
    # outside thinking. Accepted per user D-Q1: safety > precision.
    tool_choice_any_alias="auto",
    tool_choice_forbidden_when_thinking=frozenset({"required"}),
    emits_tool_calls_in_content=False,
    supports_thinking=True,       # Sonnet 4.6 / Haiku 4.5 via extra_body.thinking
    thinking_always_on=False,     # opt-in via extra_body
    thinking_toggle_style="extra_body_thinking",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=False,
    reasoning_echo_across_user_turns=False,
    # codex Round 2 Fact #1: supports_thinking_with_tools=True is correct.
    supports_thinking_with_tools=True,
    accepts_image_url=True,
    accepts_image_base64=True,
    image_max_bytes=5 * 1024 * 1024,  # compat-layer per-image cap undocumented;
                                       # sanitizer 硬上限 5 MiB，此字段实际被 sanitizer 截断
    supports_vision=True,
    supports_pdf_input=False,     # file-type parts: "Ignored" per docs table
    silently_ignored_sampling_params=frozenset({
        "logprobs", "top_logprobs", "frequency_penalty", "presence_penalty",
        "seed", "logit_bias", "service_tier", "store", "user", "modalities",
        "reasoning_effort", "prediction", "metadata", "audio",
    }),
    forbidden_sampling_params=frozenset(),
    supports_response_format_json_object=False,  # silently ignored
    supports_response_format_json_schema=False,  # silently ignored
    response_format_silently_ignored=True,       # T24 confirmed
    # Fingerprint 顺序优先匹配 (tuple-order): 更具体的 thinking/tool_choice 子模式
    # 放在宽 invalid_request_error 前 (codex Round 2 P2-2).
    # Order requirement (Audit Round 1 P2 #6): more specific substring first.
    # R3's F14 body contains BOTH "tool_choice" AND "thinking"; tool_choice
    # must win, so it precedes the bare "thinking" fingerprint below.
    error_fingerprints=(
        ErrorFingerprint(
            code="anthropic_rate_limit",
            status_code=429,
            body_substring="rate_limit_error",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
        ErrorFingerprint(
            code="anthropic_overloaded",
            status_code=529,
            body_substring="overloaded_error",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
        ErrorFingerprint(
            code="anthropic_timeout",
            status_code=504,
            body_substring="timeout_error",
            error_class=ErrorClass.TRANSIENT_CONNECTION,
        ),
        ErrorFingerprint(
            code="anthropic_authentication_error",
            status_code=401,
            body_substring="authentication_error",
            error_class=ErrorClass.TRANSIENT_AUTH,
        ),
        ErrorFingerprint(
            code="anthropic_request_too_large",
            status_code=413,
            body_substring="request_too_large",
            error_class=ErrorClass.CONTEXT_OVERFLOW,
        ),
        # R3 target: matches bodies mentioning tool_choice — reordered before "thinking"
        ErrorFingerprint(
            code="thinking_forbidden_with_tool_choice",
            status_code=400,
            body_substring="tool_choice",
            error_class=ErrorClass.COMPAT_QUIRK,
        ),
        # Pure thinking errors (no tool_choice token) fall through to this line
        ErrorFingerprint(
            code="anthropic_thinking_quirk",
            status_code=400,
            body_substring="thinking",
            error_class=ErrorClass.COMPAT_QUIRK,
        ),
        ErrorFingerprint(
            code="anthropic_invalid_request",
            status_code=400,
            body_substring="invalid_request_error",
            error_class=ErrorClass.PERMANENT_4XX,
        ),
    ),
    default_context_window=200_000,    # Haiku 4.5 floor; Sonnet/Opus reach 1M
    default_max_output_tokens=64_000,  # Sonnet/Haiku; Opus 4.7 allows 128k
    downgrade_targets=(),
)

register_profile(ANTHROPIC_COMPAT_PROFILE)
