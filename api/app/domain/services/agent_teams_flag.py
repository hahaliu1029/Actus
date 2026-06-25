"""C2-full S4 agent-team feature flag (spec §6).

Env-var (not a Settings field) for the same operator-controlled-rollout reason
as ACTUS_C2_COORDINATOR_ENABLED / ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED:
agent teams widen the child's persona + tool surface, so the flip is a
deliberate ops action. Composes with the coordinator flag (must be ON for any
coordinator path); a team is consumed only when both flags AND a team_slug are
set (§6).
"""
from __future__ import annotations

import os

_TRUTHY = frozenset({"true", "1", "yes", "on"})


def is_agent_teams_enabled() -> bool:
    return (
        os.environ.get("ACTUS_C2_AGENT_TEAMS_ENABLED", "").strip().lower() in _TRUTHY
    )


def assert_agent_teams_enabled() -> None:
    if not is_agent_teams_enabled():
        raise RuntimeError(
            "ACTUS_C2_AGENT_TEAMS_ENABLED is false; agent-team specialization "
            "should not be admitted. Check S4 rollout readiness before flipping."
        )
