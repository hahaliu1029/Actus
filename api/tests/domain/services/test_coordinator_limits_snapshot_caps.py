# api/tests/domain/services/test_coordinator_limits_snapshot_caps.py
import pytest

from app.domain.services.coordinator_limits import (
    CoordinatorLimits,
    load_coordinator_limits_from_env,
)


def test_snapshot_caps_have_positive_defaults() -> None:
    limits = CoordinatorLimits()
    assert limits.max_snapshot_paths > 0
    assert limits.max_snapshot_files > 0
    assert limits.max_snapshot_total_bytes > 0
    assert limits.max_snapshot_seconds > 0


def test_snapshot_caps_loaded_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_PATHS", "9000")
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_FILES", "4000")
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_TOTAL_BYTES", "1048576")
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "12.5")
    limits = load_coordinator_limits_from_env()
    assert limits.max_snapshot_paths == 9000
    assert limits.max_snapshot_files == 4000
    assert limits.max_snapshot_total_bytes == 1048576
    assert limits.max_snapshot_seconds == 12.5


def test_snapshot_caps_non_positive_rejected_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_PATHS", "0")
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "-1")
    default = CoordinatorLimits()
    limits = load_coordinator_limits_from_env()
    assert limits.max_snapshot_paths == default.max_snapshot_paths
    assert limits.max_snapshot_seconds == default.max_snapshot_seconds


def test_snapshot_seconds_is_independent_from_child_wallclock_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD", "10800"
    )
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "250")
    limits = load_coordinator_limits_from_env()
    assert limits.max_wallclock_seconds_per_child == 10800
    assert limits.max_snapshot_seconds == 250.0
