"""T-P1-3: anthropic_compat contract tests (Spec §5.3, §6 F08-F15)."""
from __future__ import annotations

import httpx
import openai
import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
import app.domain.services.provider_profiles.anthropic_compat  # noqa: F401


def _make_exc(status: int, body: str) -> openai.APIError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body)
    if status == 429:
        return openai.RateLimitError(body, response=resp, body={"message": body})
    if status == 401:
        return openai.AuthenticationError(body, response=resp, body={"message": body})
    if status in (400, 404, 413, 422):
        return openai.BadRequestError(body, response=resp, body={"message": body})
    return openai.APIStatusError(body, response=resp, body={"message": body})


def test_anthropic_compat_profile_declarations() -> None:
    p = get_profile("anthropic_compat")
    assert p.provider_id == "anthropic_compat"
    assert p.default_api_mode == "chat_completions"
    assert p.api_mode_fallback_enabled is False
    # codex Round 1 P1: alias="auto" (not "required") to avoid rewrite loop under thinking
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset({"required"})
    assert p.supports_thinking is True
    assert p.thinking_always_on is False
    assert p.thinking_toggle_style == "extra_body_thinking"
    # codex Round 2 Fact #1: Anthropic extended thinking IS supported with tools
    assert p.supports_thinking_with_tools is True
    assert p.reasoning_echo_in_tool_loop is False
    assert p.reasoning_echo_across_user_turns is False
    # multimodal
    assert p.accepts_image_url is True
    assert p.accepts_image_base64 is True
    assert p.image_max_bytes == 5 * 1024 * 1024
    assert p.supports_vision is True
    assert p.supports_pdf_input is False
    # T24 confirmed: response_format silently ignored
    assert p.supports_response_format_json_object is False
    assert p.supports_response_format_json_schema is False
    assert p.response_format_silently_ignored is True
    assert "logprobs" in p.silently_ignored_sampling_params
    assert "reasoning_effort" in p.silently_ignored_sampling_params
    assert p.forbidden_sampling_params == frozenset()
    assert p.default_context_window == 200_000
    assert p.default_max_output_tokens == 64_000


@pytest.mark.parametrize("status,body,expected_class", [
    (429, "rate_limit_error",                ErrorClass.TRANSIENT_RATE_LIMIT),     # F08
    (529, "overloaded_error",                ErrorClass.TRANSIENT_RATE_LIMIT),     # F09
    (504, "timeout_error",                   ErrorClass.TRANSIENT_CONNECTION),     # F10
    (401, "authentication_error",            ErrorClass.TRANSIENT_AUTH),           # F11
    (413, "request_too_large",               ErrorClass.CONTEXT_OVERFLOW),         # F12
    # narrow sub-patterns match before generic catch-all:
    (400, "thinking is not supported with tool_choice=required", ErrorClass.COMPAT_QUIRK),  # F13
    (400, "tool_choice conflict under thinking", ErrorClass.COMPAT_QUIRK),  # F14
    (400, "invalid_request_error: generic",  ErrorClass.PERMANENT_4XX),           # F15 catch-all
])
def test_anthropic_compat_fingerprints(status: int, body: str, expected_class: ErrorClass) -> None:
    p = get_profile("anthropic_compat")
    exc = _make_exc(status, body)
    assert classify_error(exc, p) == expected_class


def test_anthropic_compat_fingerprint_order() -> None:
    """F13 'thinking' must match before F15 'invalid_request_error' catch-all."""
    p = get_profile("anthropic_compat")
    # Body containing both "thinking" and "invalid_request_error" — F13 should win
    exc = _make_exc(400, "invalid_request_error: thinking+tools conflict")
    assert classify_error(exc, p) == ErrorClass.COMPAT_QUIRK


def test_anthropic_compat_registered() -> None:
    p = get_profile("anthropic_compat")
    assert p.provider_id == "anthropic_compat"
