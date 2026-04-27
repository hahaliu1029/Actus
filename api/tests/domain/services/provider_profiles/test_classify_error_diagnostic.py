import httpx
import openai
import pytest

from app.domain.services.provider_profiles._base import ErrorClass, ErrorDiagnostic
from app.domain.services.provider_profiles._classify import (
    classify_error,
    classify_error_diagnostic,
)
from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE


def _bad_request(body_str: str, *, status: int = 400) -> openai.BadRequestError:
    """Helper A from plan header — real httpx.Response + flat body dict."""
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req, text=body_str)
    return openai.BadRequestError(message=body_str, response=resp, body={"message": body_str})


def test_classify_error_diagnostic_returns_error_class_and_fingerprint_code():
    diag = classify_error_diagnostic(
        _bad_request("Json mode response is not supported when enable_thinking is true"),
        DASHSCOPE_QWEN_PROFILE,
    )
    assert isinstance(diag, ErrorDiagnostic)
    assert diag.error_class == ErrorClass.COMPAT_QUIRK
    assert diag.fingerprint_code == "json_mode_with_thinking"


def test_classify_error_diagnostic_returns_none_fingerprint_when_no_match():
    # Plain RuntimeError has no httpx response → status=0, body="" → falls
    # through all fingerprints to the exception-class fallback (UNKNOWN).
    diag = classify_error_diagnostic(
        RuntimeError("some unknown error"), GENERIC_OPENAI_PROFILE,
    )
    assert diag.error_class == ErrorClass.UNKNOWN
    assert diag.fingerprint_code is None


def test_classify_error_backward_compat_delegates_to_diagnostic():
    exc = _bad_request("context_length_exceeded")
    assert (
        classify_error(exc, GENERIC_OPENAI_PROFILE)
        == classify_error_diagnostic(exc, GENERIC_OPENAI_PROFILE).error_class
    )


def test_anthropic_F14_body_classifies_as_thinking_forbidden_with_tool_choice():
    """R3 regression gate: F14 error body contains both 'thinking' and
    'tool_choice'. After Task 0.3a reorder, the more-specific tool_choice
    fingerprint must win so R3 will actually fire at runtime.
    """
    from app.domain.services.provider_profiles.anthropic_compat import (
        ANTHROPIC_COMPAT_PROFILE,
    )
    exc = _bad_request(
        '{"error":{"type":"invalid_request_error","message":'
        '"`tool_choice`: any is not supported when thinking is enabled"}}'
    )
    diag = classify_error_diagnostic(exc, ANTHROPIC_COMPAT_PROFILE)
    assert diag.error_class.name == "COMPAT_QUIRK"
    assert diag.fingerprint_code == "thinking_forbidden_with_tool_choice"


def test_anthropic_pure_thinking_body_does_not_hit_R3_code():
    """Regression: a thinking-only error (no tool_choice token) must NOT
    classify as thinking_forbidden_with_tool_choice.
    """
    from app.domain.services.provider_profiles.anthropic_compat import (
        ANTHROPIC_COMPAT_PROFILE,
    )
    exc = _bad_request(
        '{"error":{"type":"invalid_request_error","message":'
        '"thinking.budget_tokens must be >= 1024"}}'
    )
    diag = classify_error_diagnostic(exc, ANTHROPIC_COMPAT_PROFILE)
    assert diag.fingerprint_code != "thinking_forbidden_with_tool_choice"
