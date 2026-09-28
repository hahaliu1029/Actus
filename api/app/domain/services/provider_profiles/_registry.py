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
    ErrorFingerprint(
        code="generic_rate_limit_429",
        status_code=429,
        body_substring=None,
        error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
    ),
    ErrorFingerprint(
        code="generic_server_5xx",
        status_code=503,
        body_substring=None,
        error_class=ErrorClass.TRANSIENT_RATE_LIMIT,
    ),
    ErrorFingerprint(
        code="generic_invalid_api_key",
        status_code=401,
        body_substring=None,
        error_class=ErrorClass.TRANSIENT_AUTH,
    ),
)


# ---------------------------------------------------------------------------
# A7 P1 pure-allowlist heuristic helpers (Spec §7.2)
#
# 已知在 target 列表的模型 → 返回已注册 profile_id；未命中 → 直接返回
# "generic_openai"（行为与今天等价，无 WARN 噪音）。不预写未来 sub-profile
# 的字段值（codex Round 3 证明预测错的风险高）。
# ---------------------------------------------------------------------------

# Allowlist prefixes for DashScope Qwen text flagship (hybrid default-off, tool-capable).
# Docs: https://www.alibabacloud.com/help/en/model-studio/deep-thinking
# NOT qwen-max* (non-thinking only, Round 4 Fact #1);
# NOT qwq-*, qwen3-*-thinking-* (always-on);
# NOT qwen3.5-* (hybrid default-on).
_DASHSCOPE_QWEN_TEXT_PREFIXES: tuple[str, ...] = (
    "qwen-plus",      # qwen-plus, qwen-plus-latest, qwen-plus-YYYY-MM-DD
    "qwen-turbo",     # qwen-turbo, qwen-turbo-latest
    "qwen-flash",     # qwen-flash, qwen-flash-latest (Round 4 Fact #2)
    "qwen3-max",      # qwen3-max, qwen3-max-preview
)
# Allowlist prefixes for DashScope VL models (Qwen3-VL: hybrid + tool-capable).
# NOT qwen-vl-* (Qwen2.5-VL, no-thinking / no-tools, Round 4 Fact #3);
# NOT qwen3-omni-* / qwen-omni-* / qwen3.5-omni-* / qwen3.5-plus/flash.
_DASHSCOPE_QWEN_VL_PREFIXES: tuple[str, ...] = (
    "qwen3-vl-",      # qwen3-vl-plus, qwen3-vl-flash
)
# Allowlist prefixes for Anthropic OpenAI-compat (thinking togglable models).
# NOT claude-opus-4-7 (no manual thinking);
# NOT claude-3-* / claude-4-0..4-5 (legacy, not yet in target).
_ANTHROPIC_COMPAT_MODEL_PREFIXES: tuple[str, ...] = (
    "claude-sonnet-4-6",    # covers -latest and dated variants
    "claude-haiku-4-5",
)
# Only 2.5 Flash family (thinking togglable via reasoning_effort).
# NOT gemini-2.5-pro (thinking always-on);
# NOT gemini-3-* (Thinking MEDIUM bug).
_GEMINI_COMPAT_MODEL_PREFIXES: tuple[str, ...] = (
    "gemini-2.5-flash",   # -8b, -latest, dated variants
)


def _classify_dashscope_model(mn: str) -> str:
    """Pure allowlist. Unknown models → 'generic_openai' (= today's behavior)."""
    mn = (mn or "").lower().strip()
    if any(mn.startswith(p) for p in _DASHSCOPE_QWEN_TEXT_PREFIXES):
        return "dashscope_qwen"
    if any(mn.startswith(p) for p in _DASHSCOPE_QWEN_VL_PREFIXES):
        return "dashscope_qwen_vl"
    return "generic_openai"


def _classify_anthropic_model(mn: str) -> str:
    mn = (mn or "").lower().strip()
    if any(mn.startswith(p) for p in _ANTHROPIC_COMPAT_MODEL_PREFIXES):
        return "anthropic_compat"
    return "generic_openai"


def _classify_gemini_model(mn: str) -> str:
    mn = (mn or "").lower().strip()
    if any(mn.startswith(p) for p in _GEMINI_COMPAT_MODEL_PREFIXES):
        return "gemini_compat"
    return "generic_openai"


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
        return _classify_dashscope_model(mn)
    if "api.anthropic.com/v1" in bu:
        return _classify_anthropic_model(mn)
    if "generativelanguage.googleapis.com" in bu:
        return _classify_gemini_model(mn)
    # minimax / glm / openai_official 分支保持 base_url-only 匹配
    if "minimax" in bu or "minimaxi" in bu:
        return "minimax"
    if "bigmodel.cn" in bu or "zhipu" in bu or "api.z.ai" in bu:
        if mn == "glm-5.2" or mn.startswith("glm-5.2-"):
            return "glm_5_2_coding" if "/coding/" in bu else "glm_5_2"
        return "glm"
    if "api.openai.com" in bu:
        return "openai_official"

    logger.warning(
        "[A7] provider unspecified and base_url=%s did not match any heuristic; "
        "falling back to generic_openai profile",
        base_url,
    )
    return "generic_openai"


def all_profiles() -> tuple[ProviderProfile, ...]:
    """Snapshot of every registered profile.

    Returns a tuple (not a view) so test iteration is stable even if
    register_profile() is called concurrently.
    """
    return tuple(_REGISTRY.values())
