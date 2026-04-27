"""Zhipu GLM (bigmodel.cn) minimum-viable profile (Spec §5.6, Finding 2).

Target models: glm-4.5 / 4.6 / 4.5-air / 5.1 / 5v-turbo / 4.5v / 4.6v

设计锚：与 generic_openai **最薄运行时偏离**（Spec §5.6 + §2.3 Finding 1）。
对 _parse.py / _wire.py / reasoning echo / sampling strip / 采样参数 /
image size / response_format json_object 等 runtime-consumed 字段完全匹配
generic_openai 默认 (supports_thinking=False / tool_choice_any_alias=
"required" / forbidden-sampling 空集等)，保留今天 GLM 用户走 silent-fallback
的行为等价。thinking-capable GLM-4.5/4.6/5.1 的专用 profile 留 P2 follow-up
(glm_thinking)。

Runtime-consumed 偏离 (2 条，均 Spec §5.6 保守选择)：
1. api_mode_fallback_enabled=False (vs generic_openai=True)
   — GLM 无 Responses API (actus_fallback_chat_model.py:29 注释锁定)
2. supports_response_format_json_schema=False (vs generic_openai 默认=True)
   — 官方 structured-output docs 只确认 json_object；json_schema 保守 False，
   等 GLM-4.7+ 明确文档再 flip (codex Round 2 P2-3 / Fact #6)。由
   _rewrites.py:91-101 消费：type='json_schema' 时 strip + warning.

Declarative 偏离 (1 条，无 runtime 效果 — Finding 1)：
3. supports_thinking_with_tools=False (vs generic_openai 默认=True, _base.py:84)
   — GLM profile supports_thinking=False，此派生字段显式设 False 保持内部
   一致。目前 runtime 未消费 (grep 无命中)，且在 supports_thinking=False gate
   之下也不生效 — 对用户行为完全无影响。

Docs source:
- Zhipu OpenAI-compat intro: https://docs.bigmodel.cn/cn/guide/develop/openai/introduction
- GLM-4.6 guide (Z.AI): https://docs.z.ai/guides/llm/glm-4.6
- GLM-4.5V guide (Z.AI): https://docs.z.ai/guides/vlm/glm-4.5v
- Thinking mode (Z.AI): https://docs.z.ai/guides/capabilities/thinking-mode
- Structured output (Z.AI): https://docs.z.ai/guides/capabilities/structured-output
- Chat Completions reference (bigmodel.cn): https://docs.bigmodel.cn/api-reference/%E6%A8%A1%E5%9E%8B-api/%E5%AF%B9%E8%AF%9D%E8%A1%A5%E5%85%A8
- API error codes: https://docs.bigmodel.cn/cn/faq/api-code
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass, ErrorFingerprint, ProviderProfile,
)
from app.domain.services.provider_profiles._registry import register_profile


GLM_PROFILE = ProviderProfile(
    provider_id="glm",
    human_name="Zhipu GLM (bigmodel.cn, minimum-viable)",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=False,   # Runtime divergence #1 from generic_openai (no Responses API)
    tool_choice_any_alias="required",  # matches generic_openai default
    tool_choice_forbidden_when_thinking=frozenset(),
    emits_tool_calls_in_content=False,
    # Thinking fields at generic_openai defaults. Thinking-capable GLM
    # (4.5/4.6/5.1) is future glm_thinking sub-profile (P2 follow-up).
    # Setting supports_thinking=True here would change _parse.py:55 /
    # _wire.py:30 behavior for existing glm-5v-turbo users — not a
    # "minimum viable" migration.
    supports_thinking=False,
    thinking_always_on=False,
    thinking_toggle_style="none",
    reasoning_content_field_name="reasoning_content",
    reasoning_echo_in_tool_loop=False,
    reasoning_echo_across_user_turns=False,
    supports_thinking_with_tools=False,
    accepts_image_url=True,
    accepts_image_base64=True,
    image_max_bytes=3_932_160,       # match generic_openai
    supports_vision=True,
    supports_pdf_input=False,
    silently_ignored_sampling_params=frozenset(),
    forbidden_sampling_params=frozenset(),
    supports_response_format_json_object=True,
    # Runtime divergence #2 from generic_openai (codex Round 2 P2-3 / Fact #6):
    # generic_openai 默认 supports_response_format_json_schema=True;
    # GLM 官方 structured-output docs 只确认 json_object，json_schema 无明确
    # OpenAI-compat 层面文档支撑 → 设 False 保守。_rewrites.py:91-101 会在
    # request_rf.type='json_schema' 时 strip + 发 warning。GLM-4.7+ 若原生
    # 支持，用户遇到 1214 再推进 follow-up (glm_thinking 子 profile 或 flip)。
    supports_response_format_json_schema=False,
    response_format_silently_ignored=False,
    error_fingerprints=(
        ErrorFingerprint(
            code="glm_1002_invalid_auth",
            status_code=401,
            body_substring="1002",
            error_class=ErrorClass.TRANSIENT_AUTH,
        ),
        ErrorFingerprint(
            code="glm_1302_rate_limit",
            status_code=429,
            body_substring="1302",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
        # 1214 is a generic "invalid params" code, not response_format-specific
        # (codex Round 1 Fact #7). Categorize as PERMANENT_4XX.
        ErrorFingerprint(
            code="glm_1214_invalid_params",
            status_code=400,
            body_substring="1214",
            error_class=ErrorClass.PERMANENT_4XX,
        ),
        ErrorFingerprint(
            code="glm_internal_error",
            status_code=500,
            body_substring="internal_error",
            error_class=ErrorClass.TRANSIENT_CONNECTION,
        ),
        ErrorFingerprint(
            code="glm_gateway_timeout",
            status_code=504,
            body_substring="gateway_timeout",
            error_class=ErrorClass.TRANSIENT_CONNECTION,
        ),
    ),
    default_context_window=128_000,
    default_max_output_tokens=8_192,
    downgrade_targets=(),
)

register_profile(GLM_PROFILE)
