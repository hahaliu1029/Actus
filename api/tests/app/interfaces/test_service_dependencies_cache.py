"""Tests for _load_app_config() TTL + mtime/size cache with generation counter."""
import os
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.domain.models.app_config import (
    A2AConfig,
    AgentConfig,
    AppConfig,
    LLMConfig,
    MCPConfig,
)


def _make_config(**overrides) -> AppConfig:
    defaults = {
        "llm_config": LLMConfig(),
        "agent_config": AgentConfig(),
        "mcp_config": MCPConfig(),
        "a2a_config": A2AConfig(),
    }
    defaults.update(overrides)
    return AppConfig(**defaults)


def _write_config(path: Path, config: AppConfig) -> None:
    data = config.model_dump(mode="json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, allow_unicode=True)


def _reset_config_cache():
    import app.interfaces.service_dependencies as mod
    mod._config_cache = None
    mod._config_mtime = 0.0
    mod._config_size = 0
    mod._config_expiry = 0.0
    mod._config_generation = 0


@pytest.fixture(autouse=True)
def reset_cache():
    _reset_config_cache()
    yield
    _reset_config_cache()


@pytest.fixture
def config_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, _make_config())
    with patch("app.interfaces.service_dependencies.settings") as mock_settings:
        mock_settings.app_config_filepath = str(config_path)
        mock_settings.config_cache_ttl = 5
        yield config_path


class TestConfigCacheHit:
    def test_consecutive_calls_return_same_object(self, config_file):
        from app.interfaces.service_dependencies import _load_app_config
        result1 = _load_app_config()
        result2 = _load_app_config()
        assert result1 is result2

    def test_generation_unchanged_on_cache_hit(self, config_file):
        import app.interfaces.service_dependencies as mod
        from app.interfaces.service_dependencies import _load_app_config
        _load_app_config()
        gen1 = mod._config_generation
        _load_app_config()
        gen2 = mod._config_generation
        assert gen1 == gen2


class TestConfigCacheMiss:
    def test_mtime_change_triggers_reload(self, config_file):
        from app.interfaces.service_dependencies import _load_app_config
        result1 = _load_app_config()
        new_config = _make_config(llm_config=LLMConfig(model_name="gpt-changed"))
        _write_config(config_file, new_config)
        new_mtime = config_file.stat().st_mtime + 2
        os.utime(config_file, (new_mtime, new_mtime))
        result2 = _load_app_config()
        assert result2 is not result1
        assert result2.llm_config.model_name == "gpt-changed"

    def test_ttl_expiry_triggers_reload(self, config_file):
        from app.interfaces.service_dependencies import _load_app_config
        import app.interfaces.service_dependencies as mod
        result1 = _load_app_config()
        mod._config_expiry = 0.0
        result2 = _load_app_config()
        assert result2 is not result1

    def test_generation_increments_on_reload(self, config_file):
        import app.interfaces.service_dependencies as mod
        from app.interfaces.service_dependencies import _load_app_config
        _load_app_config()
        gen1 = mod._config_generation
        mod._config_expiry = 0.0
        _load_app_config()
        gen2 = mod._config_generation
        assert gen2 == gen1 + 1


class TestConfigEdgeCases:
    def test_cold_start_creates_default_config(self, tmp_path):
        config_path = tmp_path / "nonexistent" / "config.yaml"
        assert not config_path.exists()
        with patch("app.interfaces.service_dependencies.settings") as mock_settings:
            mock_settings.app_config_filepath = str(config_path)
            mock_settings.config_cache_ttl = 60
            from app.interfaces.service_dependencies import _load_app_config
            _reset_config_cache()
            result = _load_app_config()
            assert result is not None
            assert config_path.exists()

    def test_stat_failure_returns_stale_cache(self, config_file):
        from app.interfaces.service_dependencies import _load_app_config
        import app.interfaces.service_dependencies as mod
        result1 = _load_app_config()
        mod._config_expiry = 0.0
        with patch.object(Path, "stat", side_effect=OSError("gone")):
            result2 = _load_app_config()
        assert result2 is result1


class TestLLMCacheByFingerprint:
    def test_same_config_cache_hit_skips_constructor(self):
        """Second call with identical config must NOT call ActusChatModel again."""
        from app.interfaces.service_dependencies import _build_llm, _llm_cache
        _llm_cache.clear()
        config = LLMConfig(model_name="gpt-4o")
        with patch("app.interfaces.service_dependencies.ActusChatModel") as mock_chat:
            with patch("app.interfaces.service_dependencies.ActusResponsesModel"):
                llm1 = _build_llm(config)
                llm2 = _build_llm(config)
        assert llm1 is llm2
        assert mock_chat.call_count == 1

    def test_different_model_cache_miss_calls_constructor_twice(self):
        """Different LLMConfig fingerprint must produce distinct instances."""
        from app.interfaces.service_dependencies import _build_llm, _llm_cache
        _llm_cache.clear()
        config1 = LLMConfig(model_name="gpt-4o")
        config2 = LLMConfig(model_name="gpt-4o-mini")
        with patch("app.interfaces.service_dependencies.ActusChatModel",
                    side_effect=lambda **kw: MagicMock(name=f"chat-{kw['model_name']}")):
            with patch("app.interfaces.service_dependencies.ActusResponsesModel",
                        side_effect=lambda **kw: MagicMock(name=f"resp-{kw['model_name']}")):
                llm1 = _build_llm(config1)
                llm2 = _build_llm(config2)
        assert llm1 is not llm2
        assert len(_llm_cache) == 2

    def test_response_format_flag_changes_fingerprint(self):
        from app.interfaces.service_dependencies import _llm_fingerprint
        config1 = LLMConfig(supports_response_format=True)
        config2 = LLMConfig(supports_response_format=False)
        assert _llm_fingerprint(config1) != _llm_fingerprint(config2)

    def test_cache_eviction_at_max_size(self):
        from app.interfaces.service_dependencies import _build_llm, _llm_cache
        _llm_cache.clear()
        configs = [LLMConfig(model_name=f"model-{i}") for i in range(5)]
        with patch("app.interfaces.service_dependencies.ActusChatModel",
                    side_effect=lambda **kw: MagicMock(name=f"chat-{kw['model_name']}")):
            with patch("app.interfaces.service_dependencies.ActusResponsesModel",
                        side_effect=lambda **kw: MagicMock(name=f"resp-{kw['model_name']}")):
                for c in configs:
                    _build_llm(c)
        assert len(_llm_cache) == 4  # maxsize=4, oldest evicted
