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
        assert limits.max_wallclock_seconds_per_child == 300
        assert limits.max_total_token_cost_usd_per_run == 2.00
        assert limits.max_total_wallclock_seconds_per_run == 900
        assert limits.max_concurrent_coordinator_runs_per_user == 2
        assert limits.max_coordinator_token_cost_usd_per_user_per_day == 50.00

    def test_frozen(self):
        with pytest.raises(FrozenInstanceError):
            CoordinatorLimits().max_work_units_per_run = 99  # type: ignore[misc]

    def test_wallclock_below_supervisor_backstop(self):
        from app.domain.models.mailbox_envelope import SUBAGENT_RESULT_READY_TIMEOUT_SECONDS
        assert CoordinatorLimits().max_wallclock_seconds_per_child < SUBAGENT_RESULT_READY_TIMEOUT_SECONDS

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

    # [C2 PR-1 codex R4 P2] env overrides must respect invariants:
    # (1) every cap is strictly positive; (2) child wallclock cap must trip
    # BEFORE the 600s supervisor backstop.
    def test_zero_override_falls_back(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": "0"}):
            assert load_coordinator_limits_from_env().max_work_units_per_run == 5

    def test_negative_override_falls_back(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD": "-0.5"}):
            assert load_coordinator_limits_from_env().max_token_cost_usd_per_child == 0.50

    def test_wallclock_at_supervisor_backstop_falls_back(self):
        # Equality with the 600s backstop violates the strict-less-than invariant.
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD": "600"}):
            assert load_coordinator_limits_from_env().max_wallclock_seconds_per_child == 300

    def test_wallclock_above_supervisor_backstop_falls_back(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD": "601"}):
            assert load_coordinator_limits_from_env().max_wallclock_seconds_per_child == 300

    def test_wallclock_below_supervisor_backstop_accepted(self):
        with patch.dict(os.environ, {"ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD": "599"}):
            assert load_coordinator_limits_from_env().max_wallclock_seconds_per_child == 599
