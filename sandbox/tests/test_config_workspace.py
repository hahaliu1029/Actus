"""Sandbox Workspace Isolation — config fields + absolute-path validator (PR-1)."""
import pytest


def test_defaults_are_absolute_roots():
    from app.core.config import Settings

    s = Settings()
    assert s.workspace_root == "/home/ubuntu"
    assert s.service_install_dir == "/sandbox"


def test_relative_workspace_root_rejected():
    from pydantic import ValidationError
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(workspace_root="home/ubuntu")  # not absolute


def test_relative_service_install_dir_rejected():
    from pydantic import ValidationError
    from app.core.config import Settings

    with pytest.raises(ValidationError):
        Settings(service_install_dir="sandbox")  # not absolute


def test_get_settings_picks_up_env_after_cache_clear(monkeypatch):
    """[D1] get_settings() is lru_cached; a fresh root takes effect after
    cache_clear() — proves we do NOT snapshot the root at import time."""
    from app.core import config as cfg

    monkeypatch.setenv("WORKSPACE_ROOT", "/home/custom")
    cfg.get_settings.cache_clear()
    try:
        assert cfg.get_settings().workspace_root == "/home/custom"
    finally:
        monkeypatch.delenv("WORKSPACE_ROOT", raising=False)
        cfg.get_settings.cache_clear()
