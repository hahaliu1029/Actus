"""Registry: profile lookup by provider_id + heuristic inference by base_url.

GENERIC_FINGERPRINTS: provider-agnostic error classification baseline.
Profile-specific fingerprints take precedence (see _classify.py).
"""
from __future__ import annotations

import logging

from app.application.errors.exceptions import ConfigError
from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorFingerprint,
    ProviderProfile,
)
from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE
from app.domain.services.provider_profiles.openai_official import OPENAI_OFFICIAL_PROFILE

logger = logging.getLogger(__name__)


# Base fingerprints applied after profile-specific ones (see _classify.classify_error).
GENERIC_FINGERPRINTS: tuple[ErrorFingerprint, ...] = (
    ErrorFingerprint(429, None, ErrorClass.TRANSIENT_RATE_LIMIT),
    ErrorFingerprint(503, None, ErrorClass.TRANSIENT_RATE_LIMIT),
    ErrorFingerprint(401, None, ErrorClass.TRANSIENT_AUTH),
)


_REGISTRY: dict[str, ProviderProfile] = {
    GENERIC_OPENAI_PROFILE.provider_id: GENERIC_OPENAI_PROFILE,
    OPENAI_OFFICIAL_PROFILE.provider_id: OPENAI_OFFICIAL_PROFILE,
}


def register_profile(profile: ProviderProfile) -> None:
    """Provider profile 模块 import 时调用以自注册。Idempotent."""
    _REGISTRY[profile.provider_id] = profile


def get_profile(provider_id: str) -> ProviderProfile:
    """Lookup by provider_id. Unknown id raises ConfigError (fail-fast)."""
    if provider_id not in _REGISTRY:
        raise ConfigError(
            f"unknown provider '{provider_id}'. Known: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[provider_id]


def infer_provider_from_base_url(
    base_url: str, *, model_name: str = ""
) -> str:
    """Heuristic inference. Unmatched → generic_openai + WARN.

    Returns provider_id (string), not ProviderProfile — caller pipes through
    get_profile() after. Separated so unit tests can assert string identity
    independently of profile registration state.
    """
    bu = (base_url or "").lower()
    mn = (model_name or "").lower()

    if "moonshot" in bu or "kimi" in bu:
        return "kimi_k2"                # K2 default; K2.6 must be explicit
    if "deepseek" in bu:
        return "deepseek_reasoner" if "reasoner" in mn else "deepseek_chat"
    if "dashscope" in bu or "aliyuncs.com/dashscope" in bu:
        return "dashscope"
    if "api.anthropic.com/v1" in bu:
        return "anthropic_compat"
    if "generativelanguage.googleapis.com" in bu:
        return "gemini_compat"
    if "minimax" in bu or "minimaxi" in bu:
        return "minimax"
    if "bigmodel.cn" in bu or "zhipu" in bu:
        return "glm"
    if "api.openai.com" in bu:
        return "openai_official"

    logger.warning(
        "[A7] provider unspecified and base_url=%s did not match any heuristic; "
        "falling back to generic_openai profile",
        base_url,
    )
    return "generic_openai"
