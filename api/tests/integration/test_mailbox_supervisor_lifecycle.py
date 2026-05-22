"""C3 PR-3c — Supervisor lifecycle integration tests (plan §3892-3946).

Verifies that ``AgentTaskRunner`` wires the per-pod ``MailboxSupervisor``
correctly:

* Root session reaching RUNNING → ``SupervisorRegistry.spawn(root_id)``
  registers an alive slot.
* Root session reaching terminal → ``SupervisorRegistry.stop(root_id)``
  removes the slot.
* Subagent session with ``subagent_control_plane='legacy'`` MUST NOT
  spawn a supervisor on the root (legacy plane bypasses the mailbox).

Status: AUTO-SKIPPED on PR-3c. The plan §3948-3950 calls out the harness
fixtures (``full_runner_stack``, ``supervisor_registry``,
``root_session_pending``, ``root_session_running``,
``legacy_subagent_session``) that need a real AgentTaskRunner + DB-seeded
sessions. PR-3c pre-stages the test bodies; PR-4 (terminal-handler +
harness expansion) adds the fixtures. The skip deactivates automatically
the moment they appear — no edit to this file needed.

Same auto-deactivating skipif pattern as PR-3b's
``test_mailbox_crash_recovery.py`` so neither file becomes a stale skip
that someone has to remember to remove.

**codex r5 [MEDIUM TEST] note on placeholder method names** —
``runner.start_session(...)`` / ``runner.complete_session(...)`` below
are INTENT placeholders, NOT real ``AgentTaskRunner`` methods (the
real surface is ``invoke(task)`` + terminal-status flow; see plan
§step 2 line 3910 "adapt to harness"). PR-4 — when it adds the
``full_runner_stack`` fixture — MUST either:
  (a) provide a tiny shim that exposes ``start_session`` / ``complete_session``
      delegating to the real runner lifecycle, OR
  (b) rewrite these test bodies to use ``invoke`` + a synthetic Task and
      drive ``_set_terminal_status_with_notifications`` directly.
Either way the assertions on ``supervisor_registry.health_check()``
remain valid. The AST CI gates in
``tests/domain/services/test_agent_task_runner_supervisor_hooks.py``
already lock the production call sites, so this file's job is the
end-to-end DB+Redis loop.
"""

from __future__ import annotations

import importlib
import pytest


def _has_pr3c_runner_harness() -> bool:
    """Return True iff the lifecycle-integration harness is in conftest.

    Pulled into a function so the introspection runs once at module
    import time. Using ``request.getfixturevalue`` inside a helper
    fixture would trigger the autouse ``_migrate`` (DB connect) before
    the skip decision, defeating the purpose. ``importlib.import_module``
    is a pure attribute lookup — no DB, no event loop.
    """
    try:
        mod = importlib.import_module("tests.integration.conftest")
    except Exception:
        return False
    return all(
        hasattr(mod, name)
        for name in (
            "full_runner_stack",
            "supervisor_registry",
            "root_session_pending",
            "root_session_running",
            "legacy_subagent_session",
        )
    )


pytestmark = pytest.mark.skipif(
    not _has_pr3c_runner_harness(),
    reason=(
        "C3 PR-3c: requires 'full_runner_stack' + supervisor/session "
        "harness fixtures in api/tests/integration/conftest.py. PR-3c "
        "stages the test bodies; PR-4 adds the fixtures — these tests "
        "auto-activate then (no skip-removal edit needed)."
    ),
)


@pytest.mark.anyio
async def test_supervisor_spawned_on_root_running(
    full_runner_stack,
    supervisor_registry,
    root_session_pending,
):
    """Plan §3903 — root reaching RUNNING triggers ``registry.spawn``.

    Asserts ``health_check()`` exposes the root as ``alive`` after the
    runner's PR-3c spawn hook fires inside ``invoke``.
    """
    runner, _ctx = full_runner_stack
    await runner.start_session(root_session_pending.id)
    health = await supervisor_registry.health_check()
    assert health.get(root_session_pending.id) == "alive"


@pytest.mark.anyio
async def test_supervisor_stopped_on_root_terminal(
    full_runner_stack,
    supervisor_registry,
    root_session_running,
):
    """Plan §3917 — root reaching FINISHING/COMPLETED triggers
    ``registry.stop``.

    Asserts the slot is gone after terminal transition.

    codex r1 [MEDIUM TEST] — pre-condition assert added so the test is
    meaningful when PR-4 wires the fixtures. Without it, an empty
    initial ``health_check()`` would pass even if ``complete_session``
    had no side-effect on the registry (i.e. the stop hook is wired
    wrong but the test is asymptotic-on-empty). The pre-condition pins
    the harness contract: ``root_session_running`` MUST come with an
    alive supervisor slot already; the test then verifies ``stop`` was
    actually invoked, not just "slot is absent for any reason".
    """
    runner, _ctx = full_runner_stack
    # Pre-condition: the root has a supervisor slot before terminal.
    initial_health = await supervisor_registry.health_check()
    assert root_session_running.id in initial_health, (
        "harness contract: root_session_running fixture must come with a "
        "live supervisor slot; otherwise this test is vacuously true"
    )

    await runner.complete_session(root_session_running.id)
    health = await supervisor_registry.health_check()
    assert root_session_running.id not in health


@pytest.mark.anyio
async def test_supervisor_not_spawned_for_legacy_subagent(
    full_runner_stack,
    supervisor_registry,
    legacy_subagent_session,
):
    """Plan §3931 — ``subagent_control_plane='legacy'`` subagents must
    NOT spawn a mailbox supervisor on their root.

    The runner's PR-3c spawn hook is double-gated:
      * ``_is_root_session()`` → False for subagent rows (no-op)
      * ``_mailbox_supervisor_enabled`` flag (defaults False)

    Either guard suffices; this test asserts the joint behaviour.
    """
    state_before = await supervisor_registry.health_check()
    runner, _ctx = full_runner_stack
    await runner.start_session(legacy_subagent_session.id)
    state_after = await supervisor_registry.health_check()
    parent_id = legacy_subagent_session.parent_session_id
    # Either the parent never had a slot, or its state is unchanged.
    assert parent_id not in state_after or (
        state_after.get(parent_id) == state_before.get(parent_id, "absent")
    )
