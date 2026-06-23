# api/tests/domain/services/test_coordinator_shell_mode_flag.py
import pytest

from app.domain.services.coordinator_shell_mode_flag import (
    assert_coordinator_shell_mode_enabled,
    is_coordinator_shell_mode_enabled,
)

_ENV = "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED"


def test_default_off(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    assert is_coordinator_shell_mode_enabled() is False


def test_empty_string_off(monkeypatch):
    monkeypatch.setenv(_ENV, "")
    assert is_coordinator_shell_mode_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "yes", "on", "TRUE", "  On  "])
def test_truthy_values_on(monkeypatch, val):
    monkeypatch.setenv(_ENV, val)
    assert is_coordinator_shell_mode_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "off", "garbage"])
def test_falsy_values_off(monkeypatch, val):
    monkeypatch.setenv(_ENV, val)
    assert is_coordinator_shell_mode_enabled() is False


def test_assert_raises_when_off(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(RuntimeError, match="ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED"):
        assert_coordinator_shell_mode_enabled()


def test_assert_noop_when_on(monkeypatch):
    monkeypatch.setenv(_ENV, "true")
    assert_coordinator_shell_mode_enabled()  # no raise
