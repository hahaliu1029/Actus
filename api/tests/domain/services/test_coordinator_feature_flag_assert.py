"""PR-9b-D — assert_coordinator_enabled() raises when flag is off (unit only).

Complements the dark-launch integration test by isolating the raw assert
behavior from the SSE / app stack.
"""
from __future__ import annotations

import pytest


def test_assert_raises_when_flag_off(monkeypatch):
    from app.domain.services.coordinator_feature_flag import (
        assert_coordinator_enabled,
    )

    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "false")
    with pytest.raises(RuntimeError):
        assert_coordinator_enabled()


def test_assert_no_raise_when_flag_on(monkeypatch):
    from app.domain.services.coordinator_feature_flag import (
        assert_coordinator_enabled,
    )

    monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")
    # Should not raise.
    assert_coordinator_enabled()
