"""未匹配任何启发式的保守兜底 profile。

Non-thinking, no stripping, no forbidden params. api_mode_fallback_enabled=True 允许
chat→responses 升级（因为未知 provider 可能真的实现两套 API）。
"""
from __future__ import annotations

from app.domain.services.provider_profiles._base import ProviderProfile

GENERIC_OPENAI_PROFILE = ProviderProfile(
    provider_id="generic_openai",
    human_name="Generic OpenAI-compatible (fallback)",
    default_api_mode="chat_completions",
    api_mode_fallback_enabled=True,
    # 3.75 MB raw ≈ 5 MB base64 — OpenAI Chat API hard limit on the encoded
    # data URL payload. Matches the legacy ``_IMAGE_TARGET_RAW_SIZE`` guard
    # in agent_task_runner so the base64 fallback path does not silently
    # relax the ceiling relative to pre-A7 behavior.
    image_max_bytes=3_932_160,
)
