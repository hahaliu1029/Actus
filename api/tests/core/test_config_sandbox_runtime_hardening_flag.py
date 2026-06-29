from __future__ import annotations

from core.config import Settings


def test_runtime_hardening_flags_default_false():
    s = Settings(env="test")
    assert s.sandbox_runtime_hardening_enabled is False
    assert s.sandbox_no_new_privileges_enabled is False


def test_runtime_hardening_flag_kwarg_true():
    s = Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert s.sandbox_runtime_hardening_enabled is True


def test_no_new_privileges_flag_kwarg_true():
    s = Settings(env="test", sandbox_no_new_privileges_enabled=True)
    assert s.sandbox_no_new_privileges_enabled is True


def test_runtime_hardening_flag_env_alias(monkeypatch):
    monkeypatch.setenv("ACTUS_C5_SANDBOX_RUNTIME_HARDENING_ENABLED", "true")
    assert Settings(env="test").sandbox_runtime_hardening_enabled is True


def test_no_new_privileges_flag_env_alias(monkeypatch):
    monkeypatch.setenv("ACTUS_C5_SANDBOX_NO_NEW_PRIVILEGES_ENABLED", "true")
    assert Settings(env="test").sandbox_no_new_privileges_enabled is True
