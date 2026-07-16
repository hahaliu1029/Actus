"""Run the compressed long-running coordinator acceptance matrix.

Run from ``api/`` with::

    uv run python -m scripts.verify_coordinator_long_running_leases

The selected tests use production services with deterministic clocks/fakes, so
they cross the former 600-second, two-hour, three-hour, and six-hour boundaries
without sleeping in real time or touching the live database/Redis/Docker state.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
API_ROOT = REPO_ROOT / "api"
SANDBOX_ROOT = REPO_ROOT / "sandbox"

API_ACCEPTANCE = (
    "tests/structure/test_coordinator_long_running_timeout_contract.py",
    (
        "tests/application/services/test_coordinator_terminal_envelope_waiter.py"
        "::test_default_none_awaits_delayed_terminal_without_wait_for"
    ),
    (
        "tests/application/services/test_coordinator_run_orchestrator.py"
        "::TestObserverExitConditions::test_default_none_waits_without_fixed_deadline"
    ),
    (
        "tests/application/services/test_coordinator_run_orchestrator.py"
        "::TestRunLevelBudgetCaps::test_explicit_positive_deadline_cancels_pending"
    ),
    (
        "tests/application/services/test_coordinator_run_orchestrator.py"
        "::TestPublishDedup::test_observer_then_watcher_no_double_publish"
    ),
    (
        "tests/application/services/test_coordinator_child_runner_terminal_ordering.py"
        "::test_outer_heartbeat_covers_blocked_finalizer_and_terminal_is_last"
    ),
    (
        "tests/application/services/test_coordinator_liveness_lease_service.py"
        "::test_stale_boundary_missing_semantics_and_clear_are_explicit"
    ),
    (
        "tests/application/services/test_coordinator_liveness_lease_service.py"
        "::test_ordinary_research_heartbeat_renews_its_sandbox_owners"
    ),
    (
        "tests/application/services/test_coordinator_parent_execution_lease.py"
        "::test_backend_phases_renew_past_three_hours_until_owner_done"
    ),
    (
        "tests/application/services/test_coordinator_parent_execution_lease.py"
        "::test_quota_renew_context_keeps_user_and_run_identity_past_six_hours"
    ),
    (
        "tests/app/application/services/test_subagent_research_service.py"
        "::test_probe_quota_lease_renews_until_stopped"
    ),
    (
        "tests/infrastructure/cache/test_probe_quota.py"
        "::test_continuous_ordinary_probe_renew_crosses_original_ttl"
    ),
    (
        "tests/infrastructure/cache/test_probe_quota_coordinator_lease.py"
        "::test_continuous_renew_past_six_hours_keeps_the_same_slot"
    ),
    (
        "tests/application/services/test_patch_applier.py"
        "::test_apply_lock_renews_past_600_seconds_of_fake_time"
    ),
    (
        "tests/application/services/test_patch_applier.py"
        "::test_recovery_rollback_lease_renews_past_600_seconds"
    ),
    (
        "tests/app/domain/services/test_execution_supervisor_background_quota.py"
        "::test_auto_degrade_renew_keeps_extending_for_more_than_two_hours"
    ),
)

SANDBOX_ACCEPTANCE = (
    "tests/test_supervisor_timeout_lease.py",
    "tests/test_timeout_middleware.py",
)


def _run_pytest(root: Path, targets: tuple[str, ...], env: dict[str, str]) -> int:
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *targets, "-q", "--tb=short"],
        cwd=root,
        env=env,
        check=False,
    )
    return completed.returncode


def main() -> None:
    env = os.environ.copy()
    env.setdefault("ENV", "test")
    env.setdefault("JWT_SECRET_KEY", "unit-test-secret")

    api_status = _run_pytest(API_ROOT, API_ACCEPTANCE, env)
    sandbox_status = _run_pytest(SANDBOX_ROOT, SANDBOX_ACCEPTANCE, env)
    result = {
        "api": "passed" if api_status == 0 else f"failed({api_status})",
        "sandbox": (
            "passed" if sandbox_status == 0 else f"failed({sandbox_status})"
        ),
        "live_dependencies_touched": False,
    }
    sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    if api_status != 0 or sandbox_status != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
