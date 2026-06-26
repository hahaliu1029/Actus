from __future__ import annotations

from core.config import Settings


def test_flag_defaults_false():
    assert Settings(env="test").sandbox_policy_compiler_enabled is False


def test_flag_kwarg_true():
    s = Settings(env="test", sandbox_policy_compiler_enabled=True)
    assert s.sandbox_policy_compiler_enabled is True


def test_flag_env_alias(monkeypatch):
    monkeypatch.setenv("ACTUS_C5_SANDBOX_POLICY_COMPILER_ENABLED", "true")
    assert Settings(env="test").sandbox_policy_compiler_enabled is True
