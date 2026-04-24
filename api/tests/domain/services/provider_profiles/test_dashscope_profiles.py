"""T-P1-1/2: dashscope_qwen + dashscope_qwen_vl contract tests (Spec §5.1-5.2, §6 F01-F07)."""
from __future__ import annotations

import httpx
import openai
import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
import app.domain.services.provider_profiles.dashscope_qwen  # noqa: F401 triggers register
import app.domain.services.provider_profiles.dashscope_qwen_vl  # noqa: F401 triggers register


def _make_http_exc(status: int, body: str) -> openai.APIError:
    """Build an OpenAI SDK exception with real httpx.Response; classify_error reads body."""
    req = httpx.Request("POST", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body)
    if status == 429:
        return openai.RateLimitError(body, response=resp, body={"message": body})
    if status == 401:
        return openai.AuthenticationError(body, response=resp, body={"message": body})
    if status in (400, 404, 422):
        return openai.BadRequestError(body, response=resp, body={"message": body})
    # 500 / others
    return openai.APIStatusError(body, response=resp, body={"message": body})


# ---------- T-P1-1 declarations ----------


def test_dashscope_qwen_profile_declarations() -> None:
    p = get_profile("dashscope_qwen")
    assert p.provider_id == "dashscope_qwen"
    assert p.default_api_mode == "chat_completions"
    assert p.api_mode_fallback_enabled is False
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    assert p.emits_tool_calls_in_content is False
    assert p.supports_thinking is True
    assert p.thinking_always_on is False
    assert p.thinking_toggle_style == "extra_body_enable_thinking"
    assert p.reasoning_content_field_name == "reasoning_content"
    assert p.reasoning_echo_in_tool_loop is False
    assert p.reasoning_echo_across_user_turns is False
    assert p.supports_thinking_with_tools is True
    # text flagship: no vision
    assert p.accepts_image_url is False
    assert p.accepts_image_base64 is False
    assert p.image_max_bytes == 0
    assert p.supports_vision is False
    assert p.supports_pdf_input is False
    # sampling
    assert p.silently_ignored_sampling_params == frozenset({"logit_bias", "logprobs", "top_logprobs"})
    assert p.forbidden_sampling_params == frozenset()
    # response_format: unconditional True (non-thinking); thinking mode走 F02 fingerprint
    assert p.supports_response_format_json_object is True
    assert p.supports_response_format_json_schema is True
    assert p.response_format_silently_ignored is False
    # context
    assert p.default_context_window == 131_072
    assert p.default_max_output_tokens == 8_192
    assert p.downgrade_targets == ()


# ---------- T-P1-2 fingerprints ----------


@pytest.mark.parametrize("status,body,expected_class", [
    (400, "tool_choice is one of the strings",                         ErrorClass.COMPAT_QUIRK),           # F01
    (400, "Json mode response is not supported when enable_thinking is true", ErrorClass.COMPAT_QUIRK),    # F02
    (400, "Range of input length should be",                           ErrorClass.CONTEXT_OVERFLOW),       # F03
    (429, "Requests throttling triggered",                             ErrorClass.TRANSIENT_RATE_LIMIT),   # F04
    (429, "Allocated quota exceeded",                                  ErrorClass.PERMANENT_4XX),          # F05
    (401, "Invalid API-key provided",                                  ErrorClass.TRANSIENT_AUTH),         # F06
    (500, "An internal error has occured",                             ErrorClass.TRANSIENT_CONNECTION),   # F07
])
def test_dashscope_qwen_fingerprints(status: int, body: str, expected_class: ErrorClass) -> None:
    p = get_profile("dashscope_qwen")
    exc = _make_http_exc(status, body)
    assert classify_error(exc, p) == expected_class


# ---------- T-P1-R registry ----------


def test_dashscope_qwen_registered() -> None:
    p = get_profile("dashscope_qwen")
    assert p.provider_id == "dashscope_qwen"


# ---------- T-P1-2 VL declarations ----------


def test_dashscope_qwen_vl_profile_declarations() -> None:
    p = get_profile("dashscope_qwen_vl")
    assert p.provider_id == "dashscope_qwen_vl"
    assert p.human_name == "DashScope Qwen-VL (vision)"
    # Only diverges from qwen text on image-related fields:
    assert p.accepts_image_url is True
    assert p.accepts_image_base64 is True
    assert p.image_max_bytes == 10 * 1024 * 1024
    assert p.supports_vision is True
    # Inherited from qwen text (spot check a few to prove replace() worked):
    assert p.tool_choice_any_alias == "auto"
    assert p.supports_thinking is True
    assert p.thinking_always_on is False
    assert p.api_mode_fallback_enabled is False
    assert p.default_context_window == 131_072


def test_dashscope_qwen_vl_inherits_fingerprints() -> None:
    """VL profile shares all 7 DashScope fingerprints via replace()."""
    vl = get_profile("dashscope_qwen_vl")
    qwen = get_profile("dashscope_qwen")
    assert vl.error_fingerprints == qwen.error_fingerprints
