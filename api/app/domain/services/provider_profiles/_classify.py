"""Map provider-returned exceptions to ErrorClass taxonomy.

Priority: profile.error_fingerprints → _registry.GENERIC_FINGERPRINTS →
exception-class fallback → UNKNOWN.
"""
from __future__ import annotations

import httpx
import openai

from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorDiagnostic,
    ErrorFingerprint,
    ProviderProfile,
)
from app.domain.services.provider_profiles._registry import GENERIC_FINGERPRINTS


def _match_fingerprint(
    status: int,
    body: str,
    fingerprints: tuple[ErrorFingerprint, ...],
) -> ErrorFingerprint | None:
    body_lower = body.lower()
    for fp in fingerprints:
        if fp.status_code and fp.status_code != status:
            continue
        if fp.body_substring and fp.body_substring.lower() not in body_lower:
            continue
        return fp
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


def _classify_fallback(exc: Exception, status: int) -> ErrorClass:
    """Exception-class / status-based fallback when no fingerprint matched."""
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


def classify_error_diagnostic(
    exc: Exception,
    profile: ProviderProfile,
) -> ErrorDiagnostic:
    """Fine-grained classification: returns (error_class, fingerprint_code).

    Priority: profile.error_fingerprints → GENERIC_FINGERPRINTS → exception-class fallback.
    """
    status, body = _extract_status_body(exc)

    fp = _match_fingerprint(status, body, profile.error_fingerprints)
    if fp is None:
        fp = _match_fingerprint(status, body, GENERIC_FINGERPRINTS)
    if fp is not None:
        return ErrorDiagnostic(error_class=fp.error_class, fingerprint_code=fp.code)

    return ErrorDiagnostic(
        error_class=_classify_fallback(exc, status),
        fingerprint_code=None,
    )


def classify_error(exc: Exception, profile: ProviderProfile) -> ErrorClass:
    """Backward-compat shim. Canonical API is classify_error_diagnostic."""
    return classify_error_diagnostic(exc, profile).error_class
