import httpx
import openai

from app.domain.services.provider_profiles import get_profile
from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorFingerprint,
    ProviderProfile,
)
from app.domain.services.provider_profiles._classify import classify_error


def _make_profile(fingerprints: tuple[ErrorFingerprint, ...] = ()) -> ProviderProfile:
    return ProviderProfile(
        provider_id="test",
        human_name="Test",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=False,
        error_fingerprints=fingerprints,
    )


def _make_bad_request(body_substr: str, status: int = 400) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://example.com/v1/chat/completions")
    resp = httpx.Response(status, request=req, text=body_substr)
    return openai.BadRequestError(
        message=body_substr,
        response=resp,
        body={"message": body_substr},
    )


def test_profile_fingerprint_takes_priority() -> None:
    profile = _make_profile(
        (
            ErrorFingerprint(
                400, "Missing reasoning_content", ErrorClass.COMPAT_QUIRK
            ),
        )
    )
    exc = _make_bad_request("Missing reasoning_content in assistant message")
    assert classify_error(exc, profile) == ErrorClass.COMPAT_QUIRK


def test_generic_fingerprint_rate_limit() -> None:
    profile = _make_profile()
    req = httpx.Request("POST", "https://x/v1/chat")
    resp = httpx.Response(429, request=req, text="rate limited")
    exc = openai.RateLimitError(
        "rate limited", response=resp, body={"message": "rate"}
    )
    assert classify_error(exc, profile) == ErrorClass.TRANSIENT_RATE_LIMIT


def test_exception_class_fallback_auth() -> None:
    profile = _make_profile()
    req = httpx.Request("POST", "https://x/v1/chat")
    resp = httpx.Response(401, request=req, text="bad key")
    exc = openai.AuthenticationError(
        "bad key", response=resp, body={"message": "bad"}
    )
    assert classify_error(exc, profile) == ErrorClass.TRANSIENT_AUTH


def test_unknown_400_as_permanent_4xx() -> None:
    """T11: 未命中任何指纹 → PERMANENT_4XX（不透传成协议升级）"""
    profile = _make_profile()
    exc = _make_bad_request("unknown model name xyz")
    assert classify_error(exc, profile) == ErrorClass.PERMANENT_4XX


def test_completely_unknown_exception_returns_unknown() -> None:
    profile = _make_profile()
    assert classify_error(RuntimeError("weird"), profile) == ErrorClass.UNKNOWN


def test_classify_kimi_missing_reasoning_content() -> None:
    """T9: Kimi 400 'Missing reasoning_content' → COMPAT_QUIRK"""
    profile = get_profile("kimi_k2")
    exc = _make_bad_request("Missing reasoning_content in assistant message at index 2")
    assert classify_error(exc, profile) == ErrorClass.COMPAT_QUIRK
