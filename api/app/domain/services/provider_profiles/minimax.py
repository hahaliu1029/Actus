"""MiniMax M2 family profile (Spec §5.5).

Target models: MiniMax-M2 / M2.1 / M2.5 / M2.7 (text-only reasoning family).

重要语义：
- base_url 以 api.minimax.io/v1 为准 (.io = 官方 consolidated;
  minimax.chat / minimaxi.com 为历史/域内 mirror) — heuristic
  "minimax" / "minimaxi" substring 覆盖所有三域名。
- OpenAI-compat 端点完全不支持 image / audio 输入
  ("Image/Audio not currently supported").
- reasoning 默认嵌在 content 的 <think>...</think> tags 里; extra_body=
  {"reasoning_split":True} 才能拆到 reasoning_details 字段。本 PR 不注入;
  P1 保留 content-embedded 语义 (D3 决策).
- XML tool calls 在 vLLM/SGLang 后端通常归一化到 tool_calls 字段; 自托管 /
  transformers / TGI 部署下可能 leak 到 content → adapter XML fallback
  parser 已兜底 (emits_tool_calls_in_content=True declarative).

Docs source:
- OpenAI-compat reference: https://platform.minimax.io/docs/api-reference/text-openai-api
- Function call guide: https://platform.minimax.io/docs/guides/text-m2-function-call
- Error codes: https://platform.minimax.io/docs/api-reference/errorcode
- M2.7 tool calling guide: https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/main/docs/tool_calling_guide.md
- M2.5 response_format issue (L-confidence source): https://github.com/MiniMax-AI/MiniMax-M2.5/issues/4
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


MINIMAX_PROFILE = ProviderProfile(
    provider_id="minimax",
    human_name="MiniMax M2",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,
    tool_choice_any_alias="auto",
    tool_choice_forbidden_when_thinking=frozenset(),
    emits_tool_calls_in_content=True,  # declarative: XML fallback unconditional (Finding 1)
    supports_thinking=True,
    thinking_always_on=True,
    thinking_toggle_style="none",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=True,     # docs: "complete model response must be appended"
    reasoning_echo_across_user_turns=True,
    supports_thinking_with_tools=True,
    accepts_image_url=False,
    accepts_image_base64=False,
    image_max_bytes=0,
    supports_vision=False,
    supports_pdf_input=False,
    silently_ignored_sampling_params=frozenset({
        "presence_penalty", "frequency_penalty", "logit_bias",
    }),
    forbidden_sampling_params=frozenset(),
    # OBSERVATION-BASED (L-confidence, codex Fact #4): response_format silent-ignore
    # evidence from MiniMax-M2.5 issue tracker #4 + OpenRouter integration notes,
    # NOT official MiniMax docs. Upgrade to H confidence when official docs update.
    supports_response_format_json_object=False,
    supports_response_format_json_schema=False,
    response_format_silently_ignored=True,
    error_fingerprints=(
        ErrorFingerprint(
            code="minimax_invalid_api_key",
            status_code=401,
            body_substring="invalid api key",
            error_class=ErrorClass.TRANSIENT_AUTH,
        ),
        ErrorFingerprint(
            code="minimax_rate_limit",
            status_code=429,
            body_substring="rate limit",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
        ErrorFingerprint(
            code="minimax_token_limit",
            status_code=400,
            body_substring="token limit",
            error_class=ErrorClass.CONTEXT_OVERFLOW,
        ),
        ErrorFingerprint(
            code="minimax_insufficient_balance",
            status_code=402,
            body_substring="insufficient balance",
            error_class=ErrorClass.PERMANENT_4XX,
        ),
        ErrorFingerprint(
            code="minimax_internal_server_error",
            status_code=500,
            body_substring="internal server error",
            error_class=ErrorClass.TRANSIENT_CONNECTION,
        ),
        ErrorFingerprint(
            code="minimax_gateway_timeout",
            status_code=504,
            body_substring="gateway timeout",
            error_class=ErrorClass.TRANSIENT_CONNECTION,
        ),
    ),
    default_context_window=204_800,
    default_max_output_tokens=16_384,
    downgrade_targets=(),
)

register_profile(MINIMAX_PROFILE)
