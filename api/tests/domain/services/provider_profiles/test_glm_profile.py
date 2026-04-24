"""T-P1-6: glm minimum-viable contract tests (Spec §5.6, §6 F27-F31, Finding 2)."""
from __future__ import annotations

import httpx
import openai
import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
import app.domain.services.provider_profiles.glm  # noqa: F401


def _make_exc(status: int, body: str) -> openai.APIError:
    req = httpx.Request("POST", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
    resp = httpx.Response(status, request=req, text=body)
    if status == 429:
        return openai.RateLimitError(body, response=resp, body={"message": body})
    if status == 401:
        return openai.AuthenticationError(body, response=resp, body={"message": body})
    if status in (400, 404, 422):
        return openai.BadRequestError(body, response=resp, body={"message": body})
    return openai.APIStatusError(body, response=resp, body={"message": body})


def test_glm_profile_minimum_viable() -> None:
    """GLM is a minimum-viable migration from generic_openai fallback.

    Field-by-field分类 (Spec §5.6 + §2.3 Finding 1):

    **Runtime-consumed divergences from generic_openai** (2 条，被消费路径 gate)：
      1. api_mode_fallback_enabled=False (generic_openai=True) — GLM 无 Responses API
         (actus_fallback_chat_model.py:29 注释锁定)
      2. supports_response_format_json_schema=False (generic_openai 默认=True)
         — 官方 structured-output docs 只确认 json_object；json_schema 保守 False，
         等 GLM-4.7+ docs 明确 (codex Round 2 P2-3 + Fact #6). 由 _rewrites.py:91-101
         消费：type='json_schema' 时 strip + warning.

    **Declarative divergence (no runtime effect)** (1 条)：
      3. supports_thinking_with_tools=False (generic_openai 默认=True, _base.py:84)
         — GLM profile 将 supports_thinking=False，按 Finding 1 spirit 把相关派生字段
         也显式设 False 保持内部一致；`supports_thinking_with_tools` 目前未被 runtime
         消费 (grep 无命中)，gate 在 supports_thinking=False 下同样不生效，所以对现有
         GLM 用户无行为变化。

    其他 runtime-consumed 字段 (supports_thinking / tool_choice_any_alias="required" /
    reasoning_echo_* / image_max_bytes / silently+forbidden sampling 等) 全部匹配
    generic_openai 默认，确保 silent-fallback → glm profile 的迁移对现有用户行为等价
    (Finding 2).
    """
    p = get_profile("glm")
    assert p.provider_id == "glm"
    # Divergence #1: no Responses API fallback
    assert p.api_mode_fallback_enabled is False
    # These match generic_openai defaults (Finding 2 equivalence):
    assert p.tool_choice_any_alias == "required"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    assert p.supports_thinking is False           # intentional, matches generic_openai
    assert p.thinking_always_on is False
    assert p.thinking_toggle_style == "none"
    assert p.reasoning_echo_in_tool_loop is False
    assert p.reasoning_echo_across_user_turns is False
    assert p.supports_thinking_with_tools is False
    assert p.accepts_image_url is True
    assert p.accepts_image_base64 is True
    assert p.image_max_bytes == 3_932_160         # match generic_openai 3.75MB raw
    assert p.supports_vision is True
    assert p.supports_pdf_input is False
    assert p.silently_ignored_sampling_params == frozenset()
    assert p.forbidden_sampling_params == frozenset()
    # Divergence #2: response_format json_schema conservative False (codex Round 2 P2-3 / Fact #6)
    assert p.supports_response_format_json_object is True
    assert p.supports_response_format_json_schema is False   # ≠ generic_openai default True
    assert p.response_format_silently_ignored is False
    assert p.default_context_window == 128_000
    assert p.default_max_output_tokens == 8_192


@pytest.mark.parametrize("status,body,expected_class", [
    (401, "1002",               ErrorClass.TRANSIENT_AUTH),           # F27
    (429, "1302",               ErrorClass.TRANSIENT_RATE_LIMIT),     # F28
    (400, "1214",               ErrorClass.PERMANENT_4XX),            # F29 (generic param-invalid)
    (500, "internal_error",     ErrorClass.TRANSIENT_CONNECTION),     # F30
    (504, "gateway_timeout",    ErrorClass.TRANSIENT_CONNECTION),     # F31
])
def test_glm_fingerprints(status: int, body: str, expected_class: ErrorClass) -> None:
    p = get_profile("glm")
    exc = _make_exc(status, body)
    assert classify_error(exc, p) == expected_class


def test_glm_registered() -> None:
    assert get_profile("glm").provider_id == "glm"
