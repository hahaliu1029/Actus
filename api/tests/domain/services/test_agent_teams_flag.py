import pytest

from app.domain.services.agent_teams_flag import (
    assert_agent_teams_enabled,
    is_agent_teams_enabled,
)

_VAR = "ACTUS_C2_AGENT_TEAMS_ENABLED"


def test_default_off(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)
    assert is_agent_teams_enabled() is False


@pytest.mark.parametrize("val", ["true", "1", "YES", "On", " true "])
def test_truthy_values(monkeypatch, val):
    monkeypatch.setenv(_VAR, val)
    assert is_agent_teams_enabled() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "", "maybe"])
def test_falsy_values(monkeypatch, val):
    monkeypatch.setenv(_VAR, val)
    assert is_agent_teams_enabled() is False


def test_assert_raises_when_off(monkeypatch):
    monkeypatch.delenv(_VAR, raising=False)
    with pytest.raises(RuntimeError):
        assert_agent_teams_enabled()


def test_assert_passes_when_on(monkeypatch):
    monkeypatch.setenv(_VAR, "true")
    assert_agent_teams_enabled()  # no raise
