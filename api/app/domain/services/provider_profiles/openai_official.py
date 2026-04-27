"""OpenAI 官方 API profile。"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorFingerprint,
    ProviderProfile,
)

OPENAI_OFFICIAL_PROFILE = ProviderProfile(
    provider_id="openai_official",
    human_name="OpenAI Official",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=True,
    # 3.75 MB raw ≈ 5 MB base64 — OpenAI Chat API hard limit on the encoded
    # data URL payload. Matches the legacy ``_IMAGE_TARGET_RAW_SIZE`` guard.
    image_max_bytes=3_932_160,
    default_context_window=128_000,
    default_max_output_tokens=8_192,
    error_fingerprints=(
        ErrorFingerprint(
            code="context_length_exceeded",
            status_code=400,
            body_substring="context_length_exceeded",
            error_class=ErrorClass.CONTEXT_OVERFLOW,
        ),
        ErrorFingerprint(
            code="maximum_context_length",
            status_code=400,
            body_substring="maximum context length",
            error_class=ErrorClass.CONTEXT_OVERFLOW,
        ),
        ErrorFingerprint(
            code="rate_limit_exceeded",
            status_code=429,
            body_substring="rate_limit_exceeded",
            error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
        ),
    ),
)
