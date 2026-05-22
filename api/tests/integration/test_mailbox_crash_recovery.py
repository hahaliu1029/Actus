"""C3 PR-3b/PR-4 — Mailbox supervisor crash-recovery integration tests.

Original PR-3b tests:
- **T2 (crash between destroy and XACK)** — terminal handler runs destroy()
  successfully, then the supervisor pod dies before XACK can complete. The
  envelope sits in PEL; a fresh supervisor's startup XAUTOCLAIM redelivers
  it; consumer-side audit dedup MUST short-circuit + ACK so destroy() does
  NOT fire a second time.
- **T5 (supervisor task crash → local restart)** — the supervisor's
  asyncio.Task raises; ``SupervisorRegistry._restart_loop`` resurrects it
  with a new instance_id; envelope flow resumes.

PR-4 Phase H added:
- **T11 (destroy retryable failure → poison drop → orphan cascade)** —
  ``SandboxLifecycleError`` raised on every destroy; reclaim_count exceeds
  ``MAILBOX_POISON_MAX_RECLAIM``; supervisor drops the envelope as poison
  and triggers ``mailbox.poison_message_dropped`` telemetry. See
  ``test_destroy_retryable_failure_poison_then_orphan_cascade`` below.

Codex F2/F3 (HIGH) — T2 and T5 now run against the PR-4 ``full_supervisor_stack``
tuple fixture. The previous bodies assumed a richer harness API
(``spawn_supervisor`` / ``simulate_crash_after_destroy`` / etc.) so were
``xfail(run=False)``; rewritten here to:

  - **T2** — monkeypatch ``_consumer.ack`` to raise once after destroy
    runs, so the envelope stays in PEL. Spawn a second
    ``MailboxSupervisor`` sharing the same audit_repo + redis client (so
    its initial XAUTOCLAIM picks up the un-ACKed entry). Verify
    consumer-side audit dedup suppresses the second ``destroy()``.
  - **T5** — monkeypatch ``_consumer.read`` to raise once, causing the
    supervisor's outer loop to log + sleep 0.5s + continue (this is the
    existing per-iteration isolation path, NOT a task crash). For a true
    task-level crash, we instead drive a ``SupervisorRegistry`` end-to-end:
    spawn → injected factory simulates the crash → registry's restart loop
    resurrects → fresh supervisor handles a new envelope.
"""

from __future__ import annotations

import asyncio
import importlib
import uuid as _uuid

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
        "C3 PR-3b/PR-4: requires 'full_supervisor_stack' / "
        "'child_session_in_db' / 'sandbox_lifecycle_spy' fixtures in "
        "api/tests/integration/conftest.py. These now ship in PR-4 — "
        "module activates automatically when they are present."
    ),
)


# Codex F2/F3 (HIGH) — T2 / T5 rewritten to run against the PR-4 ``full_
# supervisor_stack`` tuple fixture instead of the previously-assumed richer
# harness API. The xfail(run=False) gating is gone; both tests execute on
# every integration CI run.


@pytest.mark.integration
@pytest.mark.anyio
async def test_crash_between_destroy_and_xack(
    full_supervisor_stack,
    redis_client,
    sandbox_lifecycle_spy,
    child_session_in_db,
    monkeypatch,
):
    """T2 — destroy() completes but ``_consumer.ack`` raises before XACK
    can fire. The envelope sits in PEL; a freshly-spawned second
    supervisor's startup XAUTOCLAIM (``min_idle_ms=0`` per spec §5.6 M1)
    picks it up; consumer-side audit dedup MUST suppress the second
    ``destroy()`` invocation.

    Implementation:
      1. Monkeypatch ``supervisor._consumer.ack`` to raise once, simulating
         the crash window between mark_processed (which the supervisor
         writes BEFORE XACK) and the actual XACK.
      2. Publish RESULT_READY; wait for destroy + audit ``processed_at``.
      3. Stop the first supervisor (its loop never ACKed the entry).
      4. Spawn a fresh supervisor bound to the same root via the same
         audit_repo + redis_client; its ``_initial_xautoclaim`` claims the
         PEL entry. The ``get_processed`` short-circuit in
         ``_handle_envelope`` sees ``processed_at`` is already set → ACK
         without dispatching to the handler.
      5. Assert exactly one ``destroy()`` recorded.
    """
    from app.application.services.mailbox_supervisor import (
        MailboxSupervisor,
        SupervisorContext,
    )
    from app.domain.models.mailbox_envelope import (
        CostAggregate,
        MailboxEnvelope,
        MailboxEnvelopeType,
        ProducerRole,
        ResultReadyOutcome,
        ResultReadyPayload,
    )
    from app.domain.models.session import DestroyReason

    supervisor, ctx, audit_repo, publisher, task = full_supervisor_stack

    # Inject a one-shot raise into the first supervisor's ACK path so the
    # entry stays in PEL even though destroy + mark_processed completed.
    original_ack = supervisor._consumer.ack  # noqa: SLF001
    raised = {"n": 0}

    async def _ack_raise_once(redis_id):
        if raised["n"] == 0:
            raised["n"] += 1
            raise RuntimeError("simulated crash between mark_processed and XACK")
        await original_ack(redis_id)

    monkeypatch.setattr(supervisor._consumer, "ack", _ack_raise_once)  # noqa: SLF001

    rr_env = MailboxEnvelope(
        envelope_id="01HSPYU0F20000T2CRASH0XACK00",
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="cid-f2-t2",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ResultReadyPayload(
            summary="t2",
            outcome=ResultReadyOutcome.SUCCESS,
            cost_summary=CostAggregate(),
        ).model_dump(mode="json"),
    )
    await publisher.publish(rr_env)

    # Wait for destroy AND for processed_at to land (mark_processed runs
    # post-side_effect inside the supervisor before the ACK attempt). The
    # ACK monkeypatch raises right after; entry stays in PEL.
    for _ in range(50):
        await asyncio.sleep(0.1)
        if (
            await audit_repo.get_processed(ctx.root_session_id, rr_env.envelope_id)
            and raised["n"] == 1
        ):
            break

    assert raised["n"] == 1, "ACK monkeypatch must have raised once"
    assert await audit_repo.get_processed(ctx.root_session_id, rr_env.envelope_id), (
        "mark_processed must have completed before the simulated crash"
    )
    first_destroy_count = len(
        [
            c
            for c in sandbox_lifecycle_spy.destroy_calls
            if c["session_id"] == child_session_in_db.id
        ]
    )
    assert first_destroy_count == 1, (
        f"first supervisor must have called destroy exactly once; "
        f"got {first_destroy_count}"
    )

    # Stop the first supervisor; entry remains in PEL because the crash
    # raised before XACK. (sup.stop is graceful — it does NOT replay PEL.)
    await supervisor.stop(drain_timeout_s=2.0)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    # Spawn a fresh supervisor bound to the same root. Its ``_initial_xautoclaim``
    # sweeps the PEL with ``min_idle_ms=0`` and ``get_processed`` short-circuits.
    new_instance_id = f"i{_uuid.uuid4().hex[:6]}"
    new_ctx = SupervisorContext(
        root_session_id=ctx.root_session_id,
        pod_id=ctx.pod_id,
        instance_id=new_instance_id,
        redis=ctx.redis,
        audit_repo=ctx.audit_repo,
        publisher=ctx.publisher,
        sandbox_lifecycle=ctx.sandbox_lifecycle,
        agent_service_callback=ctx.agent_service_callback,
        telemetry=ctx.telemetry,
    )
    sup_b = MailboxSupervisor(new_ctx, block_ms=0, idle_poll_sleep_s=0.05)
    sup_b._XAUTOCLAIM_MIN_IDLE_MS = 0  # noqa: SLF001 — claim immediately
    task_b = asyncio.create_task(sup_b.run())
    try:
        # Wait long enough for the new supervisor's initial XAUTOCLAIM to
        # claim the PEL entry, run _handle_envelope, hit get_processed,
        # and ACK without dispatch.
        await asyncio.sleep(1.5)
        post_destroy_count = len(
            [
                c
                for c in sandbox_lifecycle_spy.destroy_calls
                if c["session_id"] == child_session_in_db.id
                and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
            ]
        )
        assert post_destroy_count == 1, (
            f"audit dedup must suppress the second destroy across crash + "
            f"redelivery; got {post_destroy_count} total destroy calls"
        )
    finally:
        await sup_b.stop(drain_timeout_s=2.0)
        if not task_b.done():
            task_b.cancel()
            try:
                await task_b
            except (asyncio.CancelledError, Exception):
                pass


@pytest.mark.integration
@pytest.mark.anyio
async def test_supervisor_task_crash_local_restart(
    full_supervisor_stack,
    redis_client,
    sandbox_lifecycle_spy,
    child_session_in_db,
):
    """T5 — supervisor's asyncio.Task terminates with an exception; the
    per-pod ``SupervisorRegistry``'s restart loop resurrects it with a
    new instance_id. A fresh envelope published after restart MUST flow
    through the resurrected supervisor's callback chain.

    codex r2 [R2-2, HIGH TEST] + codex r5 [R5-1, HIGH TEST] —
    **T5 validates the registry's RESTART INPUT CONTRACT, NOT the
    supervisor's internal crash isolation.** Those are two separate
    invariants (spec §13.1):

      1. **Registry-side (this test):** the registry observes any tracked
         task that reaches ``done() and not cancelled() and exception()``
         and spawns a fresh task via the factory. Whether that task is a
         real ``MailboxSupervisor.run`` or a synthetic ``asyncio.Task``
         is invariant — the registry never introspects the task's
         payload, only its outcome shape. We use a pre-failed task
         marker because the real ``MailboxSupervisor.run`` outer broad-
         except guard intentionally PREVENTS non-CancelledError
         propagation out of the task (it logs + sleeps + continues). That
         guard is the supervisor's INTERNAL crash isolation, covered by
         the complementary test
         ``test_supervisor_run_loop_iteration_broad_except_swallows_runtime_error``
         in ``tests/domain/services/test_mailbox_supervisor.py``.
      2. **Supervisor-side (complementary test):** the outer broad-except
         in ``MailboxSupervisor.run`` keeps the task alive across
         transient iteration failures (XREADGROUP fault, tick raises,
         etc.) so the registry restart loop is reserved for TRULY fatal
         escapes (OOM, lazy-import faults, finally-block raises).

    Together these two tests cover spec §13.1 (supervisor task crash →
    registry restarts → reclaim PEL → continue): the supervisor never
    crashes from synthesizable raises (its job is to log + continue),
    the registry restarts whatever DOES crash regardless of cause.

    Implementation note: we drive the registry's restart contract
    DIRECTLY — build a real ``asyncio.Task`` whose coroutine raises
    immediately, swap it into the registry's slot (the registry only
    reads ``slot.task`` through public fields — no private hook to
    call), and assert ``_restart_crashed`` spawns a fresh task that
    processes a new envelope.
    """
    from app.application.services.mailbox_supervisor import (
        MailboxSupervisor,
        SupervisorContext,
    )
    from app.application.services.supervisor_registry import SupervisorRegistry
    from app.domain.models.mailbox_envelope import (
        CostAggregate,
        MailboxEnvelope,
        MailboxEnvelopeType,
        ProducerRole,
        ResultReadyOutcome,
        ResultReadyPayload,
    )

    supervisor, ctx, audit_repo, publisher, task = full_supervisor_stack
    # Stop the harness-provided supervisor so the registry-driven supervisor
    # is the only writer on the root_session_id stream.
    await supervisor.stop(drain_timeout_s=2.0)
    if not task.done():
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    factory_calls = {"n": 0}

    def _factory(root: str):
        factory_calls["n"] += 1
        new_instance_id = f"i{_uuid.uuid4().hex[:6]}"
        sub_ctx = SupervisorContext(
            root_session_id=root,
            pod_id=ctx.pod_id,
            instance_id=new_instance_id,
            redis=ctx.redis,
            audit_repo=ctx.audit_repo,
            publisher=ctx.publisher,
            sandbox_lifecycle=ctx.sandbox_lifecycle,
            agent_service_callback=ctx.agent_service_callback,
            telemetry=ctx.telemetry,
        )
        return MailboxSupervisor(
            sub_ctx, block_ms=0, idle_poll_sleep_s=0.05
        )

    registry = SupervisorRegistry(
        supervisor_factory=_factory,
        restart_interval_s=0.3,
        max_restart_count=5,
        ready_timeout_s=2.0,
    )
    try:
        await registry.spawn(ctx.root_session_id)

        first_slot = registry._slots.get(ctx.root_session_id)  # noqa: SLF001
        assert first_slot is not None, "registry.spawn must have created a slot"
        first_task = first_slot.task
        first_instance = first_slot.instance_id

        # Tear down the live run() so we can swap in the crashed-task
        # marker without two consumers competing on the same stream
        # (consumer-name uniqueness is enforced by
        # ``RedisMailboxConsumer.__init__``).
        first_task.cancel()
        try:
            await first_task
        except (asyncio.CancelledError, Exception):
            pass

        # codex r2 [R2-2] — build a REAL ``asyncio.Task`` that ends with
        # ``done() and not cancelled() and exception()`` — the precise
        # input shape the registry's ``_restart_crashed`` reacts to.
        # This is the registry's INPUT contract: any task reaching this
        # shape (whether via OOM in production or a synthetic test task)
        # MUST trigger a fresh spawn. The supervisor's own swallow-except
        # is a separate concern — if it ever lets a non-CancelledError
        # escape, the registry's restart contract catches it.
        async def _failed_marker():
            raise RuntimeError("simulated supervisor crash marker")

        failed_task = asyncio.create_task(_failed_marker())
        try:
            await failed_task
        except RuntimeError:
            pass
        # Sanity: the task carries the right shape for restart detection.
        assert failed_task.done(), "marker task must terminate"
        assert not failed_task.cancelled(), (
            "marker task must NOT be cancelled — cancelled is a clean "
            "shutdown signal the registry skips"
        )
        assert failed_task.exception() is not None, (
            "marker task must carry an exception so the registry's "
            "_restart_crashed detects it (done + not cancelled + exception)"
        )

        # Swap into the registry slot. The registry only reads
        # ``slot.task`` (no setter API exists — the live registry mutates
        # ``slot.task`` directly during restart), so writing it here is
        # the legitimate way to assert the restart input contract.
        first_slot.task = failed_task

        # Wait long enough for ≥ 1 restart tick (interval=0.3s).
        await asyncio.sleep(1.0)

        # Verify the registry restarted the slot with a new instance_id.
        post_slot = registry._slots.get(ctx.root_session_id)  # noqa: SLF001
        assert post_slot is not None
        assert post_slot.instance_id != first_instance, (
            f"registry must spawn a fresh supervisor with a new instance_id; "
            f"got same instance_id={post_slot.instance_id!r}"
        )
        assert factory_calls["n"] >= 2, (
            f"factory must have been called ≥ 2 times (spawn + restart); "
            f"got {factory_calls['n']}"
        )

        # Health check sees the resurrected slot as alive.
        health = await registry.health_check()
        assert health[ctx.root_session_id] == "alive", (
            f"health_check expected 'alive' for resurrected root; "
            f"got {health[ctx.root_session_id]!r}"
        )

        # Publish a fresh envelope; the resurrected supervisor processes it.
        rr_env = MailboxEnvelope(
            envelope_id="01HSPYU0F30000T5RESTART0000",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id=ctx.root_session_id,
            child_session_id=child_session_in_db.id,
            correlation_id="cid-f3-t5",
            emitted_at=ctx.now(),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=ResultReadyPayload(
                summary="t5",
                outcome=ResultReadyOutcome.SUCCESS,
                cost_summary=CostAggregate(),
            ).model_dump(mode="json"),
        )
        await publisher.publish(rr_env)
        # Allow the resurrected supervisor to drain.
        for _ in range(30):
            await asyncio.sleep(0.1)
            if any(
                c["session_id"] == child_session_in_db.id
                for c in sandbox_lifecycle_spy.destroy_calls
            ):
                break
        assert any(
            c["session_id"] == child_session_in_db.id
            for c in sandbox_lifecycle_spy.destroy_calls
        ), (
            "resurrected supervisor must process a fresh envelope; "
            f"destroy_calls={sandbox_lifecycle_spy.destroy_calls!r}"
        )
    finally:
        await registry.stop_all()


# ─── T11 — PR-4 Phase H (plan §"Step 13" lines 5030-5097) ────────────────────


@pytest.mark.integration
@pytest.mark.anyio
async def test_destroy_retryable_failure_poison_then_orphan_cascade(
    full_supervisor_stack,
    sandbox_lifecycle_spy,
    child_session_in_db,
    monkeypatch,
):
    """T11 — destroy raises SandboxLifecycleError on every attempt; the
    envelope's reclaim_count crosses ``MAILBOX_POISON_MAX_RECLAIM``; the
    supervisor MUST emit ``mailbox.poison_message_dropped`` telemetry.

    Per spec §5.7 + §7.5: once the poison gate ACKs + drops, the
    ``_on_poison_drop`` hook fires a synthetic CANCEL_REQUEST(TERMINATE)
    so the orphaned child can still be cleaned up. With destroy still
    raising, the cascade destroy also raises — the load-bearing
    invariant is the poison-drop telemetry signal, NOT eventual destroy
    success.

    Test-only tunables:
      * Drop ``supervisor._XAUTOCLAIM_INTERVAL_S`` so the redelivery
        retries fire rapidly.
      * Drop ``_XAUTOCLAIM_MIN_IDLE_MS`` to 0 so XAUTOCLAIM claims every
        un-ACKed entry on each sweep.
    """
    import asyncio  # local import — test-only

    from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
    from app.domain.models.mailbox_envelope import (
        MAILBOX_POISON_MAX_RECLAIM,
        CostAggregate,
        MailboxEnvelope,
        MailboxEnvelopeType,
        ProducerRole,
        ResultReadyOutcome,
        ResultReadyPayload,
    )
    from app.domain.models.session import DestroyReason

    supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    # Make destroy raise retryable error on every call.
    sandbox_lifecycle_spy.set_destroy_side_effect(
        lambda: SandboxLifecycleError("transient docker daemon error")
    )

    # Accelerate XAUTOCLAIM redelivery so reclaim_count crosses the poison
    # threshold within the test window.
    supervisor._XAUTOCLAIM_INTERVAL_S = 0.1
    supervisor._XAUTOCLAIM_MIN_IDLE_MS = 0

    rr_env = MailboxEnvelope(
        envelope_id="01HSPYU0t11000000000000000",
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="01HSPYU0t11000000000000001",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ResultReadyPayload(
            summary="x",
            outcome=ResultReadyOutcome.SUCCESS,
            cost_summary=CostAggregate(),
        ).model_dump(mode="json"),
    )
    await publisher.publish(rr_env)

    # Wait long enough for ≥ MAILBOX_POISON_MAX_RECLAIM reclaim cycles.
    # Each cycle is one XAUTOCLAIM interval; budget generously to absorb
    # ACK / DB latency.
    await asyncio.sleep((MAILBOX_POISON_MAX_RECLAIM + 1) * 0.3)

    # Assert: supervisor attempted destroy multiple times before giving up.
    destroy_attempts = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_session_in_db.id
        and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
    ]
    assert len(destroy_attempts) >= 1, (
        "supervisor must attempt destroy at least once before poison drop"
    )

    # Assert: poison-message-dropped telemetry recorded. The cascade
    # destroy may also fail (destroy spy still raises) — what matters is
    # the poison drop signal so operators can spot the stuck envelope.
    emitted_names = [name for name, _data in ctx.telemetry.emitted]
    assert "mailbox.poison_message_dropped" in emitted_names, (
        f"expected mailbox.poison_message_dropped telemetry; "
        f"got emitted={emitted_names!r}"
    )

    # codex r2 [R2-3, HIGH TEST] — spec §5.7 step 3 + §7.5 invariant:
    # after a terminal-type envelope reaches poison drop, supervisor's
    # ``_on_poison_drop`` MUST publish a synthetic CANCEL_REQUEST(TERMINATE)
    # so the orphaned child still gets a destroy attempt routed through
    # ``CancelRequestHandler``. Without this assertion T11 could pass
    # even if ``_on_poison_drop`` silently failed; the cascade is the
    # load-bearing invariant for the M2 contract (every running child
    # must reach RESULT_READY or destroy()).
    import json as _json

    stream_key = f"actus:child:{ctx.root_session_id}:mailbox"
    cascade_entries: list[dict] = []
    # Drain a few polling cycles to absorb any redis latency between
    # the poison drop and the synthetic publish landing on the stream.
    for _ in range(10):
        await asyncio.sleep(0.1)
        entries = await ctx.redis.xrange(stream_key)
        cascade_entries = []
        for _id, fields in entries:
            # codex r3 [R3-4, HIGH TEST] — ``redis_client`` fixture uses
            # ``decode_responses=True`` (conftest.py: redis_client). Field
            # keys come back as ``str`` rather than ``bytes``, so the prior
            # ``fields.get(b"envelope")`` always returned ``None`` and the
            # cascade assertion below silently never ran. Read both shapes
            # defensively to match the pattern used in
            # ``test_mailbox_approval.py``'s ``_decode_field`` helper.
            envelope_blob = fields.get(b"envelope") or fields.get("envelope")
            if envelope_blob is None:
                continue
            if isinstance(envelope_blob, (bytes, bytearray)):
                envelope_blob = envelope_blob.decode()
            try:
                env_dict = _json.loads(envelope_blob)
            except Exception:
                continue
            if (
                env_dict.get("type") == "CANCEL_REQUEST"
                and env_dict.get("child_session_id") == child_session_in_db.id
                and env_dict.get("payload", {}).get("reason")
                == "poison_drop_terminal_envelope"
            ):
                cascade_entries.append(env_dict)
        if cascade_entries:
            break

    assert cascade_entries, (
        "_on_poison_drop must publish a synthetic CANCEL_REQUEST(TERMINATE) "
        "for the orphaned child after the terminal envelope is poison-dropped "
        "(spec §5.7 step 3 + §7.5). Without this cascade the M2 invariant "
        "breaks — the child has no remaining destroy signal."
    )
    cascade_env = cascade_entries[0]
    assert cascade_env["payload"]["policy"] == "TERMINATE", (
        "cascade payload policy must be TERMINATE so the orphaned "
        "child gets forced destruction"
    )
    # codex r7 [R7-2, HIGH TEST] — R6-2 removed ``destroy_reason`` from
    # ``CancelRequestPayload`` (wire schema restored to ``{reason, policy}``).
    # The previous assertion ``payload.destroy_reason == "orphan_timeout"``
    # would fail post-R6 since the field no longer rides on the wire. The
    # override now lives in the supervisor-private
    # ``_cascade_destroy_overrides`` side-table keyed by
    # ``synthetic_envelope_id``. Assert end-to-end via the lifecycle spy:
    # the eventual destroy call that lands MUST carry
    # ``DestroyReason.ORPHAN_TIMEOUT`` so ops can distinguish poison-drop
    # cleanup from parent-initiated FORCE_TERMINATE.
    #
    # Note: in this test destroy raises on every attempt (transient
    # SandboxLifecycleError), but the supervisor still *invokes* destroy
    # with the threaded reason; the spy records the reason argument
    # regardless of whether the call raised. We poll briefly because the
    # synthetic cascade lands on a subsequent supervisor tick.
    for _ in range(20):
        await asyncio.sleep(0.1)
        orphan_destroys = [
            c
            for c in sandbox_lifecycle_spy.destroy_calls
            if c["session_id"] == child_session_in_db.id
            and c["reason"] == DestroyReason.ORPHAN_TIMEOUT
        ]
        if orphan_destroys:
            break
    assert orphan_destroys, (
        f"poison-drop cascade must invoke destroy(ORPHAN_TIMEOUT) so the "
        f"orphaned child kill is tagged correctly in binding history; "
        f"destroy_calls={sandbox_lifecycle_spy.destroy_calls!r}"
    )
