from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.config import Settings


def test_strict_flag_defaults_false():
    # Default-OFF dark-launch: a bare test Settings has strict OFF.
    assert Settings(env="test").sandbox_strict_caps_enabled is False


def test_strict_flag_parses_via_field_name_when_hardening_on():
    # The field name itself is an alias (validation_alias replaces it as a source);
    # strict needs hardening on to pass the cross-flag guard.
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_strict_caps_enabled=True,
    )
    assert s.sandbox_strict_caps_enabled is True


def test_strict_flag_parses_via_screaming_alias_when_hardening_on():
    s = Settings(
        env="test",
        SANDBOX_RUNTIME_HARDENING_ENABLED=True,
        SANDBOX_STRICT_CAPS_ENABLED=True,
    )
    assert s.sandbox_strict_caps_enabled is True


def test_strict_without_hardening_raises():
    # INV-7 fail-closed: strict alone would silently apply NO hardening → reject at build.
    with pytest.raises(ValidationError):
        Settings(env="test", sandbox_strict_caps_enabled=True)


def test_strict_with_hardening_constructs():
    s = Settings(
        env="test",
        sandbox_runtime_hardening_enabled=True,
        sandbox_strict_caps_enabled=True,
    )
    assert s.sandbox_runtime_hardening_enabled is True


def test_hardening_alone_constructs_strict_off():
    # The guard fires ONLY on strict-without-hardening; hardening alone is valid.
    s = Settings(env="test", sandbox_runtime_hardening_enabled=True)
    assert s.sandbox_strict_caps_enabled is False
