"""Tests for _build_llm — strict explicit-provider fail-fast (Task 1.8 P2 fix).

These tests lock in the Task 1.8 P2 fix: _build_llm must distinguish between
an explicitly-configured provider (should fail-fast on unknown id to surface
typos) and an inferred provider (transitional silent fallback to
generic_openai when the inferred id isn't registered yet in P0.1).

Spec anchors:
- A7 spec section 8 rollout — generic_openai fallback for un-registered profiles
- Task 1.8 plan — explicit provider typo must surface as ConfigError
"""

from __future__ import annotations

import logging

import pytest

from app.application.errors.exceptions import ConfigError
from app.domain.models.app_config import LLMConfig
from app.interfaces.service_dependencies import _build_llm, _llm_cache, _llm_lock


@pytest.fixture(autouse=True)
def _clear_llm_cache():
    """_build_llm caches; clear between tests so fingerprint collisions don't mask behavior."""
    with _llm_lock:
        _llm_cache.clear()
    yield
    with _llm_lock:
        _llm_cache.clear()


class TestExplicitProviderFailsFast:
    """When provider is explicitly configured, an unknown id must raise ConfigError."""

    def test_typo_provider_raises_config_error(self) -> None:
        cfg = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_name="m",
            provider="typo_provider",
        )
        with pytest.raises(ConfigError) as excinfo:
            _build_llm(cfg)
        assert "unknown provider" in str(excinfo.value)

    def test_openai_official_explicit_succeeds(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        cfg = LLMConfig(
            base_url="https://api.openai.com/v1",
            api_key="k",
            model_name="gpt-4o",
            provider="openai_official",
        )
        with caplog.at_level(
            logging.WARNING,
            logger="app.interfaces.service_dependencies",
        ):
            llm = _build_llm(cfg)

        # The wrapper (ActusFallbackChatModel when api_type="auto") or the bare
        # chat adapter both expose .profile.
        assert getattr(llm.profile, "provider_id", None) == "openai_official"

        # No "falling back" WARN should fire on an explicit, valid provider.
        warn_msgs = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert not any("falling back to generic_openai" in m for m in warn_msgs)


class TestInferredProviderSilentFallback:
    """When provider is NOT explicitly configured, inference unknowns silent-fall."""

    def test_inferred_unknown_provider_falls_back_with_warn(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """bigmodel.cn base_url -> heuristic returns 'glm' (not yet registered)
        -> silent fallback to generic_openai + WARN.

        Originally used moonshot.ai -> kimi_k2 when kimi_k2 was still un-registered
        in P0.1. Task 2.1 registered kimi_k2, so this test now exercises a still-
        unregistered inferred id (glm, scheduled for a later PR) to keep the
        inferred-silent-fallback contract covered.
        """
        cfg = LLMConfig(
            base_url="https://open.bigmodel.cn/api/paas/v4",
            api_key="k",
            model_name="glm-4",
            # provider omitted -> inferred path
        )
        with caplog.at_level(
            logging.WARNING,
            logger="app.interfaces.service_dependencies",
        ):
            llm = _build_llm(cfg)

        assert getattr(llm.profile, "provider_id", None) == "generic_openai"

        warn_msgs = [
            r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
        ]
        # Must WARN about the inferred id not being registered.
        assert any("falling back to generic_openai" in m for m in warn_msgs)
