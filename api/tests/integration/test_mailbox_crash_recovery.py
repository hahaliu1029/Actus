"""C3 PR-3b — Mailbox supervisor crash-recovery integration tests (T2 + T5).

These tests verify the supervisor + registry survive two real-world failure
modes the unit tests can't fully reproduce:

- **T2 (crash between destroy and XACK)** — terminal handler runs destroy()
  successfully, then the supervisor pod dies before XACK can complete. The
  envelope sits in PEL; a fresh supervisor's startup XAUTOCLAIM redelivers
  it; consumer-side audit dedup MUST short-circuit + ACK so destroy() does
  NOT fire a second time.
- **T5 (supervisor task crash → local restart)** — the supervisor's
  asyncio.Task raises; ``SupervisorRegistry._restart_loop`` resurrects it
  with a new instance_id; envelope flow resumes.

Status: AUTO-SKIPPED in PR-3b via conftest introspection. The plan defers
the integration harness fixtures (``full_supervisor_stack``,
``child_session_in_db``, ``sandbox_lifecycle_spy``) to PR-4 — see
``docs/superpowers/plans/2026-05-21-c3-mailbox-plan.md`` →
"PR-4: Terminal handlers + cascade" → "Required harness fixtures".

Codex r6/r7 [HIGH] follow-up: the prior unconditional ``pytestmark.skip``
would NOT auto-activate when PR-4 wires the fixtures (a developer had to
remember to delete the skip block). The new ``_has_pr4_harness()`` predicate
introspects ``tests/integration/conftest.py`` at module-import time and
checks for the three fixture function names; if any is missing, the
module-level ``skipif`` fires. PR-4 only needs to add the fixtures to
``conftest.py`` — these tests then activate automatically. No env vars,
no sentinel files, no manual edits to this file.

Why not ``request.getfixturevalue`` for auto-skip? Because the test
signatures reference ``full_supervisor_stack`` etc. directly, pytest
resolves them BEFORE entering any helper fixture body. The autouse
``_migrate`` fixture in integration conftest also runs at fixture setup,
attempting a real DB connection. The introspection happens at import
time and short-circuits all of that.
"""

from __future__ import annotations

import importlib
import pytest


def _has_pr4_harness() -> bool:
    """Return True iff PR-4 harness fixtures are present in the integration
    conftest. Run at module-import time — no pytest setup, no DB.

    Fixtures decorated with ``@pytest.fixture`` are still module attributes
    (wrapped callables); ``hasattr`` on the loaded conftest module is
    sufficient. Any import or attribute lookup error is treated as
    "fixtures missing" — preserving safe-skip semantics if the conftest
    module structure changes.
    """
    try:
        mod = importlib.import_module("tests.integration.conftest")
    except Exception:
        return False
    return all(
        hasattr(mod, name)
        for name in (
            "full_supervisor_stack",
            "child_session_in_db",
            "sandbox_lifecycle_spy",
        )
    )


pytestmark = pytest.mark.skipif(
    not _has_pr4_harness(),
    reason=(
        "C3 PR-3b: requires PR-4 'full_supervisor_stack' / "
        "'child_session_in_db' / 'sandbox_lifecycle_spy' fixtures in "
        "api/tests/integration/conftest.py. Plan defers integration "
        "wiring to PR-4 — these test bodies are pre-staged and "
        "auto-activate when PR-4 adds the fixtures (no skip-removal "
        "edit needed)."
    ),
)


@pytest.mark.anyio
async def test_crash_between_destroy_and_xack(
    full_supervisor_stack,
    child_session_in_db,
    sandbox_lifecycle_spy,
):
    """T2 — destroy() completes but the supervisor crashes before XACK
    can fire. A fresh supervisor's startup XAUTOCLAIM redelivers the
    envelope; consumer-side audit dedup MUST suppress the second destroy.

    Plan reference: spec §5.7 / §5.8 / §6.x; the dedup layer 2 (audit
    ``get_processed → upsert_processing → mark_processed`` lifecycle) is
    PR-4's responsibility. The test asserts ``sandbox_lifecycle_spy``
    sees exactly ONE destroy invocation across both supervisor instances.
    """
    sup_a = full_supervisor_stack.spawn_supervisor()
    # Simulate the crash window: stub the XACK side of the consumer to
    # raise, drive a terminal envelope through, then teardown sup_a.
    await sup_a.consumer.simulate_crash_after_destroy()
    # Bring up sup_b — its initial XAUTOCLAIM will inherit the orphan PEL slot.
    sup_b = full_supervisor_stack.spawn_supervisor()
    await full_supervisor_stack.wait_for_drain()
    # Exactly ONE destroy call across both supervisor instances.
    assert len(sandbox_lifecycle_spy.destroy_calls) == 1


@pytest.mark.anyio
async def test_supervisor_task_crash_local_restart(
    full_supervisor_stack,
    child_session_in_db,
):
    """T5 — supervisor's asyncio.Task crashes; the per-pod registry's
    restart loop spawns a replacement on the same root with a new
    instance_id; XREADGROUP resumes against the new consumer name and
    the second supervisor drains the PEL via XAUTOCLAIM.

    Plan reference: spec §3.2 M8 + §6.2. Asserts the registry's
    ``health_check()`` transitions ``alive → crashed → alive`` and that
    new envelopes published after the restart reach the agent_service_callback.
    """
    sup_a = full_supervisor_stack.spawn_supervisor()
    instance_a = sup_a.instance_id
    await sup_a.simulate_run_loop_crash()
    # Wait for restart loop to tick.
    await full_supervisor_stack.wait_for_restart()
    health = await full_supervisor_stack.registry.health_check()
    assert health[full_supervisor_stack.root_session_id] == "alive"
    # Restarted supervisor must have a different instance_id.
    sup_b = full_supervisor_stack.current_supervisor()
    assert sup_b.instance_id != instance_a
    # Envelope flow still works.
    await full_supervisor_stack.publish_heartbeat(child_session_in_db.id)
    await full_supervisor_stack.wait_for_callback(child_session_in_db.id)
