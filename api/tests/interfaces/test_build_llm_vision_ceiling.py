"""P1 regression — profile.supports_vision acts as a hard ceiling in _build_llm.

Locks in the P1 review finding fix: ProviderProfile.supports_vision=False on
e.g. DeepSeek Reasoner must force the wrapped adapter's supports_vision=False
even when the user's LLMConfig.supports_vision=True. Otherwise downstream
AgentTaskRunner._build_image_blocks still gets handed supports_vision=True
and embeds image blocks into DeepSeek requests, which is exactly the bug we
are fixing.
"""
from __future__ import annotations

import pytest

from app.domain.models.app_config import LLMConfig
from app.interfaces.service_dependencies import _build_llm, _llm_cache, _llm_lock


@pytest.fixture(autouse=True)
def _clear_cache():
    with _llm_lock:
        _llm_cache.clear()
    yield
    with _llm_lock:
        _llm_cache.clear()


class TestProfileVisionCeiling:
    def test_deepseek_reasoner_forces_supports_vision_false_even_if_user_enables(self) -> None:
        """Profile.supports_vision=False overrides user LLMConfig.supports_vision=True."""
        cfg = LLMConfig(
            base_url="https://api.deepseek.com/v1",
            api_key="k",
            model_name="deepseek-reasoner",
            provider="deepseek_reasoner",
            supports_vision=True,  # user sets True, profile says False
        )
        llm = _build_llm(cfg)
        # Adapter's supports_vision must be False (profile ceiling).
        # ActusFallbackChatModel wraps primary (Chat) + fallback (Responses).
        primary = getattr(llm, "primary", llm)
        assert primary.supports_vision is False

    def test_openai_profile_preserves_user_supports_vision(self) -> None:
        """openai_official has supports_vision=True -> respect user config."""
        cfg = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_name="gpt-4o",
            provider="openai_official",
            supports_vision=True,
        )
        llm = _build_llm(cfg)
        primary = getattr(llm, "primary", llm)
        assert primary.supports_vision is True

    def test_openai_profile_respects_user_disabling_vision(self) -> None:
        """User can disable vision even on a profile that supports it."""
        cfg = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_name="gpt-4o",
            provider="openai_official",
            supports_vision=False,
        )
        llm = _build_llm(cfg)
        primary = getattr(llm, "primary", llm)
        assert primary.supports_vision is False


class TestProfilePdfCeiling:
    """P2 — ProviderProfile.supports_pdf_input acts as a hard ceiling.

    A profile that declares no native-PDF support (e.g. ``openai_official``
    keeps the base default ``supports_pdf_input=False``) must force the
    adapter's ``supports_pdf_input=False`` even when the user's LLMConfig
    turns it on. Otherwise the message sanitizer and wire serialization
    still admit native PDF payloads into requests the provider cannot accept.
    """

    def test_profile_pdf_false_forces_adapter_pdf_false(self) -> None:
        cfg = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_name="gpt-4o",
            provider="openai_official",  # profile.supports_pdf_input=False
            supports_vision=True,
            supports_pdf_input=True,  # user turns ON, profile says OFF
        )
        llm = _build_llm(cfg, supports_pdf_input=True)
        primary = getattr(llm, "primary", llm)
        assert primary.supports_pdf_input is False, (
            "profile ceiling should force supports_pdf_input=False on the "
            "adapter regardless of user config"
        )

    def test_profile_pdf_true_preserves_user_pdf_setting(self) -> None:
        """Regression sibling: a profile that permits PDF does NOT strip
        the user setting — the ceiling doesn't over-fire."""
        from dataclasses import replace
        from app.domain.services.provider_profiles import get_profile
        from app.domain.services.provider_profiles._registry import (
            _REGISTRY,
            register_profile,
        )

        base = get_profile("openai_official")
        synthetic_id = "openai_official_pdf_test_synthetic"
        synthetic = replace(
            base,
            provider_id=synthetic_id,
            supports_pdf_input=True,
        )
        register_profile(synthetic)
        try:
            cfg = LLMConfig(
                base_url="https://api.openai.com/v1",
                api_key="k",
                model_name="gpt-4o",
                provider=synthetic_id,
                supports_vision=True,
                supports_pdf_input=True,
            )
            llm = _build_llm(cfg, supports_pdf_input=True)
            primary = getattr(llm, "primary", llm)
            assert primary.supports_pdf_input is True
        finally:
            _REGISTRY.pop(synthetic_id, None)
