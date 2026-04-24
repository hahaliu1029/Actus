"""T-P1-4: gemini_compat contract tests (Spec §5.4, §6 F16-F20)."""
from __future__ import annotations

import httpx
import openai
import pytest

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
import app.domain.services.provider_profiles.gemini_compat  # noqa: F401


def _make_exc(status: int, body: str) -> openai.APIError:
    req = httpx.Request("POST", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions")
    resp = httpx.Response(status, request=req, text=body)
    if status == 429:
        return openai.RateLimitError(body, response=resp, body={"message": body})
    if status in (400, 404, 422):
        return openai.BadRequestError(body, response=resp, body={"message": body})
    return openai.APIStatusError(body, response=resp, body={"message": body})


def test_gemini_compat_profile_declarations() -> None:
    p = get_profile("gemini_compat")
    assert p.provider_id == "gemini_compat"
    assert p.api_mode_fallback_enabled is False
    assert p.tool_choice_any_alias == "auto"
    assert p.tool_choice_forbidden_when_thinking == frozenset()
    assert p.supports_thinking is True
    assert p.thinking_always_on is False  # 2.5 Flash can disable via reasoning_effort="none"
    assert p.thinking_toggle_style == "openai_reasoning_effort"
    assert p.supports_thinking_with_tools is True
    assert p.accepts_image_url is True
    assert p.accepts_image_base64 is True
    assert p.supports_vision is True
    assert p.supports_pdf_input is False
    assert "logit_bias" in p.silently_ignored_sampling_params
    assert "parallel_tool_calls" in p.silently_ignored_sampling_params
    # 400 "Unknown name 'logprobs'" observed
    assert p.forbidden_sampling_params == frozenset({"logprobs", "top_logprobs"})
    assert p.supports_response_format_json_object is True
    assert p.supports_response_format_json_schema is True  # auto-maps to native responseJsonSchema
    assert p.response_format_silently_ignored is False
    assert p.default_context_window == 1_000_000
    # v1 removed F13 "Thinking level MEDIUM" — Gemini 3 specific; 5 fingerprints only
    assert len(p.error_fingerprints) == 5


@pytest.mark.parametrize("status,body,expected_class", [
    (400, "Unknown name 'logprobs'",                 ErrorClass.COMPAT_QUIRK),           # F16
    (400, "exceeds the maximum number of tokens",    ErrorClass.CONTEXT_OVERFLOW),       # F17
    (429, "RESOURCE_EXHAUSTED",                      ErrorClass.TRANSIENT_RATE_LIMIT),   # F18
    (503, "UNAVAILABLE",                             ErrorClass.TRANSIENT_CONNECTION),   # F19
    (400, "SAFETY",                                  ErrorClass.PERMANENT_4XX),          # F20
])
def test_gemini_compat_fingerprints(status: int, body: str, expected_class: ErrorClass) -> None:
    p = get_profile("gemini_compat")
    exc = _make_exc(status, body)
    assert classify_error(exc, p) == expected_class


def test_gemini_compat_no_thinking_medium_fingerprint() -> None:
    """v1 intentionally removes Gemini 3 'Thinking level MEDIUM' fingerprint (codex Fact #3).

    Reserved for future gemini_3_compat sub-profile. For gemini_compat (2.5 Flash
    only), MEDIUM body should NOT pre-emptively classify as COMPAT_QUIRK — falls
    through to exception-class fallback.
    """
    p = get_profile("gemini_compat")
    substrs = [fp.body_substring or "" for fp in p.error_fingerprints]
    assert not any("medium" in s.lower() for s in substrs)


def test_gemini_compat_registered() -> None:
    assert get_profile("gemini_compat").provider_id == "gemini_compat"
