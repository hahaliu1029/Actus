from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_run_as_user_flag_defaults_false():
    # Default-OFF dark-launch: a bare test Settings has run_as_user OFF.
    assert Settings(env="test").sandbox_run_as_user_enabled is False


def test_run_as_user_flag_parses_via_field_name_when_hardening_on():
    # The field name itself is an alias; run_as_user needs hardening on to pass the guard.
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_run_as_user_enabled=True,
    )
    assert s.sandbox_run_as_user_enabled is True


def test_run_as_user_flag_parses_via_screaming_alias_when_hardening_on():
    s = Settings(
        env="test",
        SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        SANDBOX_RUN_AS_USER_ENABLED=True,
    )
    assert s.sandbox_run_as_user_enabled is True


def test_run_as_user_without_hardening_raises():
    # INV-7 fail-closed: run_as_user alone would emit no --user AND apply NO hardening.
    with pytest.raises(ValidationError):
        Settings(env="test", sandbox_run_as_user_enabled=True)


def test_run_as_user_without_hardening_raises_even_in_external_mode():
    # codex R3 P2: the raise is mode-INDEPENDENT — a contradictory security config must
    # never be silently accepted, even when sandbox_address is set (external mode).
    with pytest.raises(ValidationError):
        Settings(
            env="test",
            sandbox_run_as_user_enabled=True,
            sandbox_address="http://remote-sandbox:8080",
        )


def test_run_as_user_with_hardening_constructs():
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_run_as_user_enabled=True,
    )
    assert s.sandbox_runtime_hardening_enabled is True


def test_hardening_alone_constructs_run_as_user_off():
    # The guard fires ONLY on run_as_user-without-hardening; hardening alone is valid.
    s = Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert s.sandbox_run_as_user_enabled is False


def test_run_as_user_on_with_default_cwd_root_warns(caplog):
    # Non-fatal pre-flip nudge: default sandbox_default_cwd is "/root" (config.py:103),
    # inaccessible to uid 1000 → a WARNING (construction still succeeds).
    with caplog.at_level(logging.WARNING, logger="core.config"):
        s = Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_run_as_user_enabled=True,
        )
    assert s.sandbox_run_as_user_enabled is True
    assert any("sandbox_default_cwd='/root'" in r.message for r in caplog.records)


def test_run_as_user_on_with_nonroot_cwd_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="core.config"):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_run_as_user_enabled=True,
            sandbox_default_cwd="/home/ubuntu",
        )
    assert not any("pre-flip checklist" in r.message for r in caplog.records)


def test_run_as_user_on_external_mode_does_not_warn(caplog):
    # codex R2 P3: in external mode the flag is inert → no spurious /root-cwd warning.
    with caplog.at_level(logging.WARNING, logger="core.config"):
        Settings(
            env="test",
            sandbox_runtime_hardening_enabled=True,
            sandbox_run_as_user_enabled=True,
            sandbox_address="http://remote-sandbox:8080",
        )
    assert not any("pre-flip checklist" in r.message for r in caplog.records)


def test_run_as_user_off_does_not_warn_even_with_default_cwd_root(caplog):
    with caplog.at_level(logging.WARNING, logger="core.config"):
        Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert not any("pre-flip checklist" in r.message for r in caplog.records)
