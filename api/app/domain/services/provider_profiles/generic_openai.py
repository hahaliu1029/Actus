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
)
