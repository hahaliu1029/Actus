"""Tests for AgentService singleton, _ConfigSnapshot, and generation-based refresh."""
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

import pytest


def _make_snapshot(**overrides):
    from app.application.services.agent_service import _ConfigSnapshot
    defaults = dict(
        llm=MagicMock(),
        agent_config=MagicMock(),
        mcp_config=MagicMock(),
        a2a_config=MagicMock(),
        skill_risk_policy=MagicMock(),
        overflow_config=MagicMock(),
        summary_llm=None,
        vision_fallback_model=None,
        skill_creator_service=MagicMock(),
        supports_vision=True,
        supports_pdf_input=False,
        file_understanding_config=None,
    )
    defaults.update(overrides)
    return _ConfigSnapshot(**defaults)


class TestConfigSnapshot:
    def test_frozen_immutable(self):
        snap = _make_snapshot()
        with pytest.raises(FrozenInstanceError):
            snap.llm = MagicMock()

    def test_atomic_refresh(self):
        """_refresh_config replaces the whole snapshot atomically."""
        from app.application.services.agent_service import AgentService
        old_snap = _make_snapshot()
        new_snap = _make_snapshot(supports_vision=False)

        svc = AgentService.__new__(AgentService)
        svc._config_snapshot = old_snap
        svc._refresh_config(new_snap)

        assert svc._config_snapshot is new_snap
        assert svc._config_snapshot.supports_vision is False


class TestGenerationBasedRefresh:
    def test_fast_path_no_refresh(self, monkeypatch):
        import app.interfaces.service_dependencies as mod

        mock_request = MagicMock()
        mock_agent_svc = MagicMock()
        mock_request.app.state.agent_service = mock_agent_svc

        monkeypatch.setattr(mod, "_config_generation", 1)
        monkeypatch.setattr(mod, "_last_refresh_generation", 1)

        with patch.object(mod, "_load_app_config"):
            result = mod.get_agent_service(mock_request)

        assert result is mock_agent_svc
        mock_agent_svc._refresh_config.assert_not_called()

    def test_slow_path_triggers_refresh(self, monkeypatch):
        import app.interfaces.service_dependencies as mod

        mock_request = MagicMock()
        mock_agent_svc = MagicMock()
        mock_request.app.state.agent_service = mock_agent_svc

        mock_config = MagicMock()
        monkeypatch.setattr(mod, "_config_cache", mock_config)
        monkeypatch.setattr(mod, "_config_generation", 2)
        monkeypatch.setattr(mod, "_last_refresh_generation", 1)

        mock_snapshot = MagicMock()
        with patch.object(mod, "_load_app_config", return_value=mock_config):
            with patch.object(mod, "_build_config_snapshot", return_value=mock_snapshot):
                result = mod.get_agent_service(mock_request)

        assert result is mock_agent_svc
        mock_agent_svc._refresh_config.assert_called_once_with(mock_snapshot)
        assert mod._last_refresh_generation == 2
