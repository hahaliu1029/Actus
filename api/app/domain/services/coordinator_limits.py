"""C2 coordinator limits (spec §14.2). All limits live in code, not prompts.

The child and per-run wallclock fields are optional deadlines: ``0`` means
unlimited and a positive value enables the corresponding deadline. Invalid
operator input for these two fields fails fast. Other caps retain the legacy
warning-and-default behavior for invalid or non-positive overrides.
"""
from __future__ import annotations
import logging
import math
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class CoordinatorLimits:
    max_work_units_per_run: int = 5
    max_tool_calls_per_child: int = 25
    max_token_cost_usd_per_child: float = 0.50
    max_wallclock_seconds_per_child: int = 0
    max_total_token_cost_usd_per_run: float = 2.00
    max_total_wallclock_seconds_per_run: int = 0
    max_concurrent_coordinator_runs_per_user: int = 2
    max_coordinator_token_cost_usd_per_user_per_day: float = 50.00
    # [C2-full S2 §3.6] snapshot-diff caps — bound the per-child workspace walk
    # (PR-1 walk + PR-4 bounded finalizer). Each remains a positive cap.
    max_snapshot_paths: int = 20000
    max_snapshot_files: int = 8000
    max_snapshot_total_bytes: int = 100 * 1024 * 1024
    max_snapshot_seconds: float = 30.0


_ENV_MAP: dict[str, tuple[str, type]] = {
    "ACTUS_COORDINATOR_MAX_WORK_UNITS_PER_RUN": ("max_work_units_per_run", int),
    "ACTUS_COORDINATOR_MAX_TOOL_CALLS_PER_CHILD": ("max_tool_calls_per_child", int),
    "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_CHILD": ("max_token_cost_usd_per_child", float),
    "ACTUS_COORDINATOR_MAX_WALLCLOCK_SECONDS_PER_CHILD": ("max_wallclock_seconds_per_child", int),
    "ACTUS_COORDINATOR_MAX_TOTAL_TOKEN_COST_USD_PER_RUN": ("max_total_token_cost_usd_per_run", float),
    "ACTUS_COORDINATOR_MAX_TOTAL_WALLCLOCK_SECONDS_PER_RUN": ("max_total_wallclock_seconds_per_run", int),
    "ACTUS_COORDINATOR_MAX_CONCURRENT_RUNS_PER_USER": ("max_concurrent_coordinator_runs_per_user", int),
    "ACTUS_COORDINATOR_MAX_TOKEN_COST_USD_PER_USER_PER_DAY": ("max_coordinator_token_cost_usd_per_user_per_day", float),
    # [C2-full S2 §3.6] snapshot-diff caps
    "ACTUS_COORDINATOR_MAX_SNAPSHOT_PATHS": ("max_snapshot_paths", int),
    "ACTUS_COORDINATOR_MAX_SNAPSHOT_FILES": ("max_snapshot_files", int),
    "ACTUS_COORDINATOR_MAX_SNAPSHOT_TOTAL_BYTES": ("max_snapshot_total_bytes", int),
    "ACTUS_COORDINATOR_MAX_SNAPSHOT_SECONDS": ("max_snapshot_seconds", float),
}

_OPTIONAL_DEADLINE_FIELDS = {
    "max_wallclock_seconds_per_child",
    "max_total_wallclock_seconds_per_run",
}

def load_coordinator_limits_from_env() -> CoordinatorLimits:
    overrides: dict[str, int | float] = {}
    for key, (field, type_) in _ENV_MAP.items():
        raw = os.environ.get(key)
        if raw is None or raw == "":
            # Empty string = unset (docker-compose pass-through with no
            # value yields "") — silent default, not a WARNING.
            continue
        if field in _OPTIONAL_DEADLINE_FIELDS:
            try:
                parsed = type_(raw)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{key} must be a non-negative integer, got {raw!r}"
                ) from exc
            if parsed < 0:
                raise ValueError(
                    f"{key} must be a non-negative integer, got {raw!r}"
                )
            overrides[field] = parsed
            continue
        try:
            parsed = type_(raw)
        except (ValueError, TypeError):
            logger.warning("coordinator_limits: invalid %s=%r, using default", key, raw)
            continue
        if type_ is float and not math.isfinite(parsed):
            logger.warning(
                "coordinator_limits: %s=%r non-finite, using default", key, raw
            )
            continue
        # Every cap is a strictly-positive budget; 0 or negative would disable
        # the gate the cap is meant to enforce.
        if parsed <= 0:
            logger.warning(
                "coordinator_limits: %s=%r non-positive, using default", key, raw
            )
            continue
        overrides[field] = parsed
    return CoordinatorLimits(**overrides)
