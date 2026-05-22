"""C3 PR-3c T6 — pod-restart reconcile recovery (plan §step 7 line 4101-4122).

Verifies that ``SandboxLifecycleService.reconcile_orphans`` re-spawns a
``MailboxSupervisor`` for every RUNNING mailbox-plane root after a pod
restart wipes the in-memory ``SupervisorRegistry``.

Scenario:
  * A subagent session row exists with ``worker_type='subagent'``,
    ``subagent_control_plane='mailbox'``, ``status='running'`` and a
    real parent root row.
  * Fresh pod boot: ``SupervisorRegistry.health_check()`` is empty (no
    in-memory slots yet — supervisor tasks died with the previous pod).
  * ``reconcile_orphans()`` runs (lifespan §step 5 of main.py).
  * Post-condition: ``health_check()`` shows the root id as ``alive`` so
    the new pod's supervisor can resume draining the PEL via the
    PR-3b startup XAUTOCLAIM.

Status: AUTO-SKIPPED on PR-3c — needs the integration harness fixture
``mailbox_plane_running_session`` (and a working ``supervisor_registry``
+ ``sandbox_lifecycle_service``). PR-4 adds the fixture; this test
activates automatically when it appears.

Same auto-deactivating skipif pattern as
``test_mailbox_supervisor_lifecycle.py`` /
``test_mailbox_crash_recovery.py`` so no stale skip lingers.
"""

from __future__ import annotations

import importlib
import pytest


def _has_pr3c_reconcile_harness() -> bool:
    try:
        mod = importlib.import_module("tests.integration.conftest")
    except Exception:
        return False
    return all(
        hasattr(mod, name)
        for name in (
            "supervisor_registry",
            "sandbox_lifecycle_service",
            "mailbox_plane_running_session",
        )
    )


pytestmark = pytest.mark.skipif(
    not _has_pr3c_reconcile_harness(),
    reason=(
        "C3 PR-3c T6: requires 'supervisor_registry' + "
        "'sandbox_lifecycle_service' + 'mailbox_plane_running_session' "
        "fixtures in api/tests/integration/conftest.py. PR-3c stages "
        "the test body; PR-4 adds the fixtures and the test "
        "auto-activates (no skip-removal edit needed)."
    ),
)


@pytest.mark.anyio
async def test_pod_restart_supervisor_recovery(
    sandbox_lifecycle_service,
    supervisor_registry,
    mailbox_plane_running_session,
):
    """Pod restart wipes in-memory supervisors; reconcile rebuilds them."""
    # Pre-condition: fresh registry, no in-memory slots.
    initial = await supervisor_registry.health_check()
    assert mailbox_plane_running_session.parent_session_id not in initial

    # Run the dual-path reconcile: existing sandbox path + new supervisor
    # ensure path (plan §11.3).
    await sandbox_lifecycle_service.reconcile_orphans()

    # Post-condition: the mailbox-plane subagent's root has a live
    # supervisor slot.
    state = await supervisor_registry.health_check()
    assert mailbox_plane_running_session.parent_session_id in state
    assert state[mailbox_plane_running_session.parent_session_id] in (
        "alive",
        "restarting",
    )
