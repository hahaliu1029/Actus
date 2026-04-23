from app.domain.services.provider_profiles._base import (
    ProviderProfile, ErrorClass, ErrorFingerprint, RewriteWarning,
)
from app.domain.services.provider_profiles._registry import (
    get_profile, infer_provider_from_base_url, register_profile,
)

__all__ = [
    "ProviderProfile", "ErrorClass", "ErrorFingerprint", "RewriteWarning",
    "get_profile", "infer_provider_from_base_url", "register_profile",
]

# Side-effect: register Kimi profiles (P0.2)
from app.domain.services.provider_profiles import kimi_k2 as _kimi_k2  # noqa: F401
from app.domain.services.provider_profiles import kimi_k2_6 as _kimi_k2_6  # noqa: F401

# Side-effect: register DeepSeek profiles (P0.3)
from app.domain.services.provider_profiles import deepseek_reasoner as _deepseek_reasoner  # noqa: F401
from app.domain.services.provider_profiles import deepseek_chat as _deepseek_chat  # noqa: F401
