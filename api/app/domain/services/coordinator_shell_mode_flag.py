"""C2-full S2 shell-capable task mode feature flag (spec §3.6).

Default-off, INERT in PR-3: it is a master kill-switch for the entire S2
shell-mode code path. While OFF (the PR-3 state), dispatch hard-coerces any
unit carrying shell_mode=True / a non-empty write_tree_lease back to typed-only
(F27 active fail-closed) — so flipping individual WorkUnit.shell_mode without
this flag changes NOTHING.

Env-var (not Settings field) for the same operator-controlled rollout reason as
ACTUS_C2_COORDINATOR_ENABLED: shell-capable children widen the child's runtime
surface, so the flip is a deliberate ops action, not application config.
"""
from __future__ import annotations
import os

_TRUTHY = frozenset({"true", "1", "yes", "on"})


def is_coordinator_shell_mode_enabled() -> bool:
    return (
        os.environ.get("ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED", "")
        .strip()
        .lower()
        in _TRUTHY
    )


def assert_coordinator_shell_mode_enabled() -> None:
    """Defensive hard-gate for the shell-mode admission path."""
    if not is_coordinator_shell_mode_enabled():
        raise RuntimeError(
            "ACTUS_C2_COORDINATOR_SHELL_MODE_ENABLED is false; shell-capable "
            "coordinator children should not be admitted. Check S2 rollout "
            "readiness before flipping this flag."
        )
