import os
from dataclasses import FrozenInstanceError
from unittest.mock import patch
import pytest
from app.domain.services.coordinator_limits import (
    CoordinatorLimits, load_coordinator_limits_from_env,
)

class TestCoordinatorLimitsDefaults:
    def test_default_values(self):
        limits = CoordinatorLimits()
        assert limits.max_work_units_per_run == 5
        assert limits.max_tool_calls_per_child == 25
        assert limits.max_token_cost_usd_per_child == 0.50
        assert limits.max_wallclock_seconds_per_child == 0
        assert limits.max_total_token_cost_usd_per_run == 2.00
        assert limits.max_total_wallclock_seconds_per_run == 0
        assert limits.max_concurrent_coordinator_runs_per_user == 2
        assert limits.max_coordinator_token_cost_usd_per_user_per_day == 50.00

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            CoordinatorLimits().max_work_units_per_run = 99  # type: ignore[misc]

class TestEnvOverride:
    def test_max_work_units(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": "10"}):
            assert load_coordinator_limits_from_env().max_work_units_per_run == 10

    def test_max_token_cost(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD": "1.25"}):
            assert load_coordinator_limits_from_env().max_token_cost_usd_per_child == 1.25

    def test_invalid_falls_back(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": "not_a_number"}):
            assert load_coordinator_limits_from_env().max_work_units_per_run == 5

    def test_other_cap_zero_override_warns_and_falls_back(self, caplog):
        import logging

        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": "0"}):
            with caplog.at_level(logging.WARNING):
                limits = load_coordinator_limits_from_env()
        assert limits.max_work_units_per_run == 5
        assert "ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN" in caplog.text

    def test_other_cap_negative_override_warns_and_falls_back(self, caplog):
        import logging

        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD": "-0.5"}):
            with caplog.at_level(logging.WARNING):
                limits = load_coordinator_limits_from_env()
        assert limits.max_token_cost_usd_per_child == 0.50
        assert "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD" in caplog.text

    @pytest.mark.parametrize(
        ("env_name", "field_name"),
        [
            (
                "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD",
                "max_token_cost_usd_per_child",
            ),
            (
                "ACTUS_COORDINATOR_MAX_TOTAL_TOKEN_COST_USD_PER_RUN",
                "max_total_token_cost_usd_per_run",
            ),
            (
                "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_USER_PER_DAY",
                "max_coordinator_token_cost_usd_per_user_per_day",
            ),
            (
                "ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS",
                "max_snapshot_seconds",
            ),
        ],
    )
    @pytest.mark.parametrize("raw_value", ["nan", "inf", "-inf"])
    def test_non_finite_float_cap_warns_and_falls_back(
        self, env_name, field_name, raw_value, caplog
    ):
        import logging

        default = CoordinatorLimits()
        with patch.dict(os.environ, {env_name: raw_value}, clear=True):
            with caplog.at_level(logging.WARNING):
                limits = load_coordinator_limits_from_env()

        assert getattr(limits, field_name) == getattr(default, field_name)
        assert env_name in caplog.text

    @pytest.mark.parametrize(
        ("env_name", "field_name"),
        [
            (
                "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD",
                "max_wallclock_seconds_per_child",
            ),
            (
                "ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN",
                "max_total_wallclock_seconds_per_run",
            ),
        ],
    )
    def test_wallclock_deadline_unset_or_empty_defaults_to_unlimited(
        self, env_name, field_name
    ):
        with patch.dict(os.environ, {}, clear=True):
            assert getattr(load_coordinator_limits_from_env(), field_name) == 0
        with patch.dict(os.environ, {env_name: ""}, clear=True):
            assert getattr(load_coordinator_limits_from_env(), field_name) == 0

    @pytest.mark.parametrize(
        ("env_name", "field_name"),
        [
            (
                "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD",
                "max_wallclock_seconds_per_child",
            ),
            (
                "ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN",
                "max_total_wallclock_seconds_per_run",
            ),
        ],
    )
    @pytest.mark.parametrize("raw_value", ["0", "10800"])
    def test_wallclock_deadline_accepts_zero_or_positive(
        self, env_name, field_name, raw_value
    ):
        with patch.dict(os.environ, {env_name: raw_value}, clear=True):
            assert getattr(load_coordinator_limits_from_env(), field_name) == int(
                raw_value
            )

    @pytest.mark.parametrize(
        "env_name",
        [
            "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD",
            "ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN",
        ],
    )
    @pytest.mark.parametrize("raw_value", ["-1", "not-a-number"])
    def test_wallclock_deadline_invalid_value_fails_fast(
        self, env_name, raw_value
    ):
        with patch.dict(os.environ, {env_name: raw_value}, clear=True):
            with pytest.raises(ValueError, match=env_name):
                load_coordinator_limits_from_env()

    def test_empty_env_string_is_silent_default(self, caplog):
        """[C2b budget §3-10] docker-compose pass-through yields "" for unset
        vars — must behave exactly like unset (code default, NO warning;
        pre-C2b an empty string hit float("")/int("") → spurious WARNING)."""
        import logging

        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": ""}):
            with caplog.at_level(logging.WARNING):
                limits = load_coordinator_limits_from_env()
        assert limits.max_work_units_per_run == CoordinatorLimits().max_work_units_per_run
        assert not [
            r for r in caplog.records if "coordinator_limits" in r.getMessage()
        ], f"empty string must be silent; got {[r.getMessage() for r in caplog.records]}"
