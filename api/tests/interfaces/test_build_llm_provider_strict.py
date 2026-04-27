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
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive contract: when ``infer_provider_from_base_url`` returns an
        id that isn't in the registry, ``_build_llm`` must:
          (a) silently fall back to ``generic_openai`` (no ConfigError raised),
          (b) emit a WARN at the ``app.interfaces.service_dependencies`` logger
              naming the unregistered inferred id.

        History: this test originally used ``moonshot.ai → kimi_k2`` (P0.1 era,
        before kimi_k2 was registered), then ``bigmodel.cn → glm`` (pre-A7 P1).
        Both targets have since been registered, so the heuristic no longer
        returns an unregistered id from any real base_url. The defensive
        try/except in ``service_dependencies.py:270-278`` is now unreachable
        from production paths but is kept as a fail-safe for the case where a
        future heuristic returns a new id before its profile lands. We
        monkeypatch the heuristic to return a synthetic unregistered id so the
        contract is verified regardless of registry state — this also stops
        the test from drifting every time a new profile is registered.
        """
        # `service_dependencies._build_llm` does a LOCAL import
        # (`from app.domain.services.provider_profiles import ...,
        # infer_provider_from_base_url, ...`) on each call, so we have to
        # patch the source-package binding rather than service_dependencies'.
        from app.domain.services import provider_profiles as profiles_pkg

        monkeypatch.setattr(
            profiles_pkg,
            "infer_provider_from_base_url",
            lambda *_args, **_kwargs: "future_unregistered_provider",
        )

        cfg = LLMConfig(
            base_url="https://api.example.com/v1",
            api_key="k",
            model_name="some-model",
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
        assert any("falling back to generic_openai" in m for m in warn_msgs), (
            f"expected 'falling back to generic_openai' WARN at "
            f"app.interfaces.service_dependencies; got {warn_msgs!r}"
        )
        assert any("future_unregistered_provider" in m for m in warn_msgs), (
            f"WARN must name the unregistered inferred id; got {warn_msgs!r}"
        )
