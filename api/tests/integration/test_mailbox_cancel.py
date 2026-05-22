"""T4 — REQUEST_CANCEL + ACK timeout → auto-escalate TERMINATE (spec §8.4).

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
§"Step 11: T4 — REQUEST_CANCEL auto-escalate" lines 4882-4935.

Scenario:
  1. Parent publishes REQUEST_CANCEL for an active child.
  2. The supervisor records ``_CancelState`` (REQUEST_CANCEL policy).
  3. Child does NOT emit CANCEL_ACK — so the auto-escalate tick must trigger.
  4. After ``CHILD_CANCEL_ACK_TIMEOUT_MS`` (test-shortened to ~200 ms), the
     supervisor synthesizes a CANCEL_REQUEST(TERMINATE) cascade and drives
     ``destroy(FORCE_TERMINATE)``.

Asserts:
  - ``destroy()`` called with reason ``FORCE_TERMINATE`` at least once for
    this child (the cascade fires on the next periodic tick).
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain.models.mailbox_envelope import (
    CancelPolicy,
    CancelRequestPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import DestroyReason


@pytest.mark.integration
@pytest.mark.anyio
async def test_request_cancel_auto_escalate_terminate(
    full_supervisor_stack,
    sandbox_lifecycle_spy,
    child_session_in_db,
    monkeypatch,
):
    """Spec §8.4 — REQUEST_CANCEL + ACK timeout → cascade FORCE_TERMINATE.

    Test-only tunables:
      * Shorten ``CHILD_CANCEL_ACK_TIMEOUT_MS`` (200 ms) via monkeypatch on
        the supervisor module so the auto-escalate tick fires within the
        test window.
      * Drop ``supervisor._CANCEL_CHECK_INTERVAL_S`` to 0.05 s so the
        periodic cascade tick runs frequently.
    """
    supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    # Accelerate the auto-escalate clock. The supervisor reads the constant
    # through its own module namespace, so monkeypatching on the supervisor
    # module is the canonical way to override it for one test.
    from app.application.services import mailbox_supervisor as _ms

    monkeypatch.setattr(_ms, "CHILD_CANCEL_ACK_TIMEOUT_MS", 200, raising=False)
    supervisor._CANCEL_CHECK_INTERVAL_S = 0.05

    cancel_env = MailboxEnvelope(
        envelope_id="01HSPYU0t4r00000000000000",
        type=MailboxEnvelopeType.CANCEL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="01HSPYU0t4r00000000000001",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.PARENT_AGENT,
        payload=CancelRequestPayload(
            reason="user_request",
            policy=CancelPolicy.REQUEST_CANCEL,
        ).model_dump(),
    )
    await publisher.publish(cancel_env)

    # Child never emits CANCEL_ACK; wait long enough that:
    #  * supervisor drains + records the REQUEST_CANCEL state (~200 ms)
    #  * the auto-escalate timeout (200 ms) elapses
    #  * the next ``_maybe_tick_cancel_check`` runs and publishes cascade
    #  * the supervisor reads the cascade and runs the TERMINATE branch
    await asyncio.sleep(1.5)

    # Assert: destroy called with FORCE_TERMINATE for this child at least once.
    force_term_calls = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_session_in_db.id
        and c["reason"] == DestroyReason.FORCE_TERMINATE
    ]
    assert force_term_calls, (
        f"expected ≥1 FORCE_TERMINATE destroy after auto-escalate, "
        f"got destroy_calls={sandbox_lifecycle_spy.destroy_calls!r}"
    )
