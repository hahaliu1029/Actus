"""C2 CoordinatorTaskRunner v1 feature flag (spec §6.4).

Default-off until PR-9 acceptance. Acts as kill-switch for the entire
coordinator code path: false → planner does NOT emit parallel_work_units,
executor does NOT enter _run_parallel_backend.

Env-var (not Settings field): coordinator deployment requires supervisor
majority on new envelope schema BEFORE producer flips this flag —
operator-controlled rollout, not application config.
"""
from __future__ import annotations
import os

_TRUTHY = frozenset({"true", "1", "yes", "on"})

def is_coordinator_enabled() -> bool:
    return os.environ.get("ACTUS_C2_COORDINATOR_ENABLED", "").strip().lower() in _TRUTHY

def assert_coordinator_enabled() -> None:
    """Defensive hard-gate; raises in dispatch_node and executor branch."""
    if not is_coordinator_enabled():
        raise RuntimeError(
            "ACTUS_C2_COORDINATOR_ENABLED is false; coordinator dispatch path "
            "should not be reached. Check supervisor majority on new envelope schema first."
        )
