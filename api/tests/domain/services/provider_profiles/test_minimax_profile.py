"""T-P1-5: minimax contract tests (Spec §5.5, §6 F21-F26)."""
from __future__ import annotations

import httpx
import openai
import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
import app.domain.services.provider_profiles.minimax  # noqa: F401


def _make_exc(status: int, body: str) -> openai.APIError:
    req = httpx.Request("POST", "https://api.minimax.io/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body)
    if status == 429:
        return openai.RateLimitError(body, response=resp, body={"message": body})
    if status == 401:
        return openai.AuthenticationError(body, response=resp, body={"message": body})
    if status in (400, 402, 404, 422):
        return openai.BadRequestError(body, response=resp, body={"message": body})
    return openai.APIStatusError(body, response=resp, body={"message": body})


def test_minimax_profile_declarations() -> None:
    p = get_profile("minimax")
    assert p.provider_id == "minimax"
    assert p.api_mode_fallback_enabled is False
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    # emits_tool_calls_in_content=True is declarative (adapter XML fallback
    # already runs unconditionally, Finding 1)
    assert p.emits_tool_calls_in_content is True
    # Interleaved thinking is core feature
    assert p.supports_thinking is True
    assert p.thinking_always_on is True
    assert p.thinking_toggle_style == "none"
    assert p.reasoning_echo_in_tool_loop is True
    assert p.reasoning_echo_across_user_turns is True
    # OpenAI-compat endpoint: "Image/Audio not currently supported"
    assert p.accepts_image_url is False
    assert p.accepts_image_base64 is False
    assert p.image_max_bytes == 0
    assert p.supports_vision is False
    assert p.supports_pdf_input is False
    # docs-confirmed silently-ignored; codex Fact #5 removed top_k
    assert p.silently_ignored_sampling_params == frozenset({
        "presence_penalty", "frequency_penalty", "logit_bias",
    })
    assert p.forbidden_sampling_params == frozenset()
    # response_format fields are L-confidence observation-based (codex Fact #4)
    assert p.supports_response_format_json_object is False
    assert p.supports_response_format_json_schema is False
    assert p.response_format_silently_ignored is True
    assert p.default_context_window == 204_800
    assert p.default_max_output_tokens == 16_384


@pytest.mark.parametrize("status,body,expected_class", [
    (401, "invalid api key",         ErrorClass.TRANSIENT_AUTH),           # F21
    (429, "rate limit",              ErrorClass.TRANSIENT_RATE_LIMIT),     # F22
    (400, "token limit",             ErrorClass.CONTEXT_OVERFLOW),         # F23
    (402, "insufficient balance",    ErrorClass.PERMANENT_4XX),            # F24
    (500, "internal server error",   ErrorClass.TRANSIENT_CONNECTION),     # F25
    (504, "gateway timeout",         ErrorClass.TRANSIENT_CONNECTION),     # F26
])
def test_minimax_fingerprints(status: int, body: str, expected_class: ErrorClass) -> None:
    p = get_profile("minimax")
    exc = _make_exc(status, body)
    assert classify_error(exc, p) == expected_class


def test_minimax_registered() -> None:
    assert get_profile("minimax").provider_id == "minimax"
