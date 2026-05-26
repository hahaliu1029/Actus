"""C2 v1 hard caps (spec §14.2). All caps in code, not in prompts.
Invariant: max_wallclock_seconds_per_child < SUBAGENT_RESULT_READY_TIMEOUT_SECONDS
(internal cap trips first → NEEDS_AUTHORIZATION; supervisor 600s backstop → TIMED_OUT).
"""
from __future__ import annotations
import logging, os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CoordinatorLimits:
    max_work_units_per_run: int = 5
    max_tool_calls_per_child: int = 25
    max_token_cost_usd_per_child: float = 0.50
    max_wallclock_seconds_per_child: int = 300
    max_total_token_cost_usd_per_run: float = 2.00
    max_total_wallclock_seconds_per_run: int = 900
    max_concurrent_coordinator_runs_per_user: int = 2
    max_coordinator_token_cost_usd_per_user_per_day: float = 50.00


_ENV_MAP: dict[str, tuple[str, type]] = {
    "ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": ("max_work_units_per_run", int),
    "ACTUS_COORDINATOR_MAX_TOOL_CALLS_PER_CHILD": ("max_tool_calls_per_child", int),
    "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD": ("max_token_cost_usd_per_child", float),
    "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD": ("max_wallclock_seconds_per_child", int),
    "ACTUS_COORDINATOR_MAX_TOTAL_TOKEN_COST_USD_PER_RUN": ("max_total_token_cost_usd_per_run", float),
    "ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN": ("max_total_wallclock_seconds_per_run", int),
    "ACTUS_COORDINATOR_MAX_CONCURRENT_RUNS_PER_USER": ("max_concurrent_coordinator_runs_per_user", int),
    "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_USER_PER_DAY": ("max_coordinator_token_cost_usd_per_user_per_day", float),
}

def load_coordinator_limits_from_env() -> CoordinatorLimits:
    # Imported lazily to avoid a domain-level circular import at module load.
    from app.domain.models.mailbox_envelope import SUBAGENT_RESULT_READY_TIMEOUT_SECONDS

    overrides: dict[str, int | float] = {}
    for key, (field, type_) in _ENV_MAP.items():
        raw = os.environ.get(key)
        if raw is None:
            continue
        try:
            parsed = type_(raw)
        except (ValueError, TypeError):
            logger.warning("coordinator_limits: invalid %s=%r, using default", key, raw)
            continue
        # Every cap is a strictly-positive budget; 0 or negative would disable
        # the gate the cap is meant to enforce.
        if parsed <= 0:
            logger.warning(
                "coordinator_limits: %s=%r non-positive, using default", key, raw
            )
            continue
        # Hard invariant from this module's docstring: child wallclock cap
        # must trip BEFORE the supervisor backstop so the failure surfaces as
        # NEEDS_AUTHORIZATION (internal) instead of TIMED_OUT (external).
        if (
            field == "max_wallclock_seconds_per_child"
            and parsed >= SUBAGENT_RESULT_READY_TIMEOUT_SECONDS
        ):
            logger.warning(
                "coordinator_limits: %s=%d >= supervisor backstop %d, using default",
                key,
                parsed,
                SUBAGENT_RESULT_READY_TIMEOUT_SECONDS,
            )
            continue
        overrides[field] = parsed
    return CoordinatorLimits(**overrides)
