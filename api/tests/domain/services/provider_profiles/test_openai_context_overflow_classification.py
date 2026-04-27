import httpx
import openai

from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.provider_profiles._classify import classify_error
from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
from app.domain.services.provider_profiles.openai_official import OPENAI_OFFICIAL_PROFILE


def _bad_request(body_str: str, *, status: int = 400) -> openai.BadRequestError:
    """Helper A from plan header — real httpx.Response + flat body dict."""
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req, text=body_str)
    return openai.BadRequestError(message=body_str, response=resp, body={"message": body_str})


def test_generic_openai_context_length_exceeded_is_context_overflow():
    assert classify_error(
        _bad_request("context_length_exceeded"), GENERIC_OPENAI_PROFILE
    ) == ErrorClass.CONTEXT_OVERFLOW


def test_openai_official_context_length_exceeded_is_context_overflow():
    assert classify_error(
        _bad_request("context_length_exceeded"), OPENAI_OFFICIAL_PROFILE
    ) == ErrorClass.CONTEXT_OVERFLOW


def test_generic_openai_maximum_context_length_substring_matches():
    assert classify_error(
        _bad_request("This model's maximum context length is 128000 tokens"),
        GENERIC_OPENAI_PROFILE,
    ) == ErrorClass.CONTEXT_OVERFLOW
