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


def test_snapshot_seconds_too_large_vs_parent_waiter_rejected_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [codex PR-4 R5+R6 P1] COMBINED finalizer-budget invariant: the parent
    # result-ready waiter (600s) must fit the inner-invoke wallclock cap PLUS the
    # 3 serial snapshot-budget phases (PRE scan + capture + manifest build/upload).
    # The loader must reject a max_snapshot_seconds where
    # `wallclock + 3*snapshot >= 600`, else a stalled shell finalize outlives the
    # parent → a late, contradictory terminal.
    default = CoordinatorLimits()  # default wallclock = 300

    # wallclock(300) + 3×250 = 1050 ≥ 600 → rejected to default.
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "250")
    assert (
        load_coordinator_limits_from_env().max_snapshot_seconds
        == default.max_snapshot_seconds
    )

    # [R6] A value the BARE 3× check (R5) would have ACCEPTED (3×199=597<600) but
    # the COMBINED bound rejects (300 + 597 = 897 ≥ 600) → must be rejected.
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "199")
    assert (
        load_coordinator_limits_from_env().max_snapshot_seconds
        == default.max_snapshot_seconds
    )

    # A value safely under the COMBINED backstop is accepted (300 + 3×90 = 570).
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "90")
    assert load_coordinator_limits_from_env().max_snapshot_seconds == 90.0


def test_snapshot_seconds_bound_uses_effective_wallclock_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # [codex PR-4 R6 P1] The combined bound uses the EFFECTIVE (overridden)
    # wallclock cap, not just the default — a higher wallclock leaves less
    # finalize room, so the same snapshot value can flip from accepted to rejected.
    default = CoordinatorLimits()

    # wallclock=400 + 3×70 = 610 ≥ 600 → snapshot rejected to default.
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD", "400")
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "70")
    limits = load_coordinator_limits_from_env()
    assert limits.max_wallclock_seconds_per_child == 400
    assert limits.max_snapshot_seconds == default.max_snapshot_seconds

    # wallclock=400 + 3×60 = 580 < 600 → accepted.
    monkeypatch.setenv("ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS", "60")
    limits = load_coordinator_limits_from_env()
    assert limits.max_wallclock_seconds_per_child == 400
    assert limits.max_snapshot_seconds == 60.0
