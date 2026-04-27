from app.domain.services.provider_profiles._base import (
    ProviderProfile, ErrorClass, ErrorFingerprint, RewriteWarning,
)
from app.domain.services.provider_profiles._registry import (
    all_profiles, get_profile, infer_provider_from_base_url, register_profile,
)

__all__ = [
    "ProviderProfile", "ErrorClass", "ErrorFingerprint", "RewriteWarning",
    "all_profiles", "get_profile", "infer_provider_from_base_url", "register_profile",
]

# Side-effect: register Kimi profiles (P0.2)
from app.domain.services.provider_profiles import kimi_k2 as _kimi_k2  # noqa: F401
from app.domain.services.provider_profiles import kimi_k2_6 as _kimi_k2_6  # noqa: F401

# Side-effect: register DeepSeek profiles (P0.3)
from app.domain.services.provider_profiles import deepseek_reasoner as _deepseek_reasoner  # noqa: F401
from app.domain.services.provider_profiles import deepseek_chat as _deepseek_chat  # noqa: F401

# Side-effect: register P1 provider profiles (2026-04-23 A7 P1).
# 严格按字母顺序 (C5 合同)；Python 导入机制保证 dashscope_qwen_vl 的
# `from .dashscope_qwen import DASHSCOPE_QWEN_PROFILE` 会透过递归 import
# 先加载 dashscope_qwen 模块触发其 register_profile()，再执行 replace()，
# 所以顺序不影响注册正确性。
from app.domain.services.provider_profiles import anthropic_compat as _anthropic_compat  # noqa: F401
from app.domain.services.provider_profiles import dashscope_qwen as _dashscope_qwen  # noqa: F401
from app.domain.services.provider_profiles import dashscope_qwen_vl as _dashscope_qwen_vl  # noqa: F401
from app.domain.services.provider_profiles import gemini_compat as _gemini_compat  # noqa: F401
from app.domain.services.provider_profiles import glm as _glm  # noqa: F401
from app.domain.services.provider_profiles import minimax as _minimax  # noqa: F401
