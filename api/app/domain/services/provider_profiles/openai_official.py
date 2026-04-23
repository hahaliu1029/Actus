"""OpenAI 官方 API profile。"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import ProviderProfile

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
)
