"""Map provider-returned exceptions to ErrorClass taxonomy.

Priority: profile.error_fingerprints → _registry.GENERIC_FINGERPRINTS →
exception-class fallback → UNKNOWN.
"""
from __future__ import annotations

import httpx
import openai

from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorFingerprint,
    ProviderProfile,
)
from app.domain.services.provider_profiles._registry import GENERIC_FINGERPRINTS


def _match_fingerprint(
    status: int,
    body: str,
    fingerprints: tuple[ErrorFingerprint, ...],
) -> ErrorClass | None:
    for fp in fingerprints:
        if fp.status_code and fp.status_code != status:
            continue
        if fp.body_substring and fp.body_substring.lower() not in body.lower():
            continue
        return fp.error_class
    return None


def _extract_status_body(exc: Exception) -> tuple[int, str]:
    """Extract HTTP status + body text from OpenAI SDK exception. 0,'' if not applicable."""
    status = 0
    body = ""
    resp = getattr(exc, "response", None)
    if isinstance(resp, httpx.Response):
        status = resp.status_code
        try:
            body = resp.text
        except Exception:
            body = ""
    # SDK stores body dict separately in some paths.
    sdk_body = getattr(exc, "body", None)
    if isinstance(sdk_body, dict):
        body = body + " " + str(sdk_body.get("message", ""))
    elif isinstance(sdk_body, str):
        body = body + " " + sdk_body
    return status, body


def classify_error(exc: Exception, profile: ProviderProfile) -> ErrorClass:
    """Classify exception into ErrorClass. Pure function, no side effects."""
    status, body = _extract_status_body(exc)

    # 1. profile-specific fingerprints
    hit = _match_fingerprint(status, body, profile.error_fingerprints)
    if hit is not None:
        return hit

    # 2. generic fingerprints
    hit = _match_fingerprint(status, body, GENERIC_FINGERPRINTS)
    if hit is not None:
        return hit

    # 3. exception-class fallback
    if isinstance(exc, openai.RateLimitError):
        return ErrorClass.TRANSIENT_RATE_LIMIT
    if isinstance(
        exc,
        (
            openai.APITimeoutError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.PoolTimeout,
        ),
    ):
        return ErrorClass.TRANSIENT_CONNECTION
    if isinstance(exc, openai.AuthenticationError):
        return ErrorClass.TRANSIENT_AUTH
    if isinstance(
        exc,
        (
            openai.BadRequestError,
            openai.UnprocessableEntityError,
            openai.NotFoundError,
        ),
    ):
        return ErrorClass.PERMANENT_4XX

    return ErrorClass.UNKNOWN
