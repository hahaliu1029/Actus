"""T1 — duplicate RESULT_READY → destroy once, both XACK (spec §5.8 + §3.2 M2).

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
§"Step 10: T1 — duplicate RESULT_READY idempotency" lines 4813-4880.

Scenario:
  1. Publisher emits a RESULT_READY envelope via the normal publisher
     (XADD goes through SET NX dedup).
  2. After the supervisor has drained / destroyed once, we simulate
     XAUTOCLAIM redelivery by XADDing the same envelope payload directly
     onto the stream (bypassing the publisher's SET NX so we exercise
     consumer-side audit-dedup specifically).
  3. The supervisor must short-circuit at ``get_processed`` and ACK without
     re-firing ``sandbox_lifecycle.destroy``.

Asserts:
  - ``destroy()`` is called exactly ONCE with reason
    ``SUBAGENT_TERMINAL_RESULT``.
  - Audit row's ``processed_at`` is set (single durable success marker).
  - Stream PEL drains to zero pending entries.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain.models.mailbox_envelope import (
    CostAggregate,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.session import DestroyReason


@pytest.mark.integration
@pytest.mark.anyio
async def test_duplicate_result_ready_destroys_child_exactly_once(
    full_supervisor_stack,
    redis_client,
    sandbox_lifecycle_spy,
    child_session_in_db,
):
    """Spec §5.8 + §3.2 M2 — duplicate RESULT_READY drives destroy once.

    Uses the same envelope id for the redelivery; the supervisor's
    audit dedup (``get_processed`` short-circuit) MUST suppress the
    second handler invocation so destroy() runs at most once.
    """
    supervisor, ctx, audit_repo, publisher, _task = full_supervisor_stack
    env = MailboxEnvelope(
        envelope_id="01HSPYU0t1d00000000000000",
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="01HSPYU0t1d00000000000001",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ResultReadyPayload(
            summary="done",
            outcome=ResultReadyOutcome.SUCCESS,
            cost_summary=CostAggregate(),
        ).model_dump(mode="json"),
    )

    # First publish — drives destroy via the real handler chain.
    await publisher.publish(env)
    await asyncio.sleep(1.0)  # let supervisor drain + destroy

    # Second publish via raw XADD bypasses publisher SET NX dedup so we
    # exercise the consumer-side audit dedup explicitly. Fields shape
    # mirrors RedisMailboxPublisher.publish() so the consumer parses it.
    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    await redis_client.xadd(
        stream_key,
        fields={
            "envelope": env.model_dump_json(),
            "envelope_id": env.envelope_id,
            "type": env.type.value,
            "producer_role": env.producer_role.value,
        },
    )
    await asyncio.sleep(1.0)

    # Assert: destroy called EXACTLY ONCE with the expected reason.
    matching = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_session_in_db.id
        and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
    ]
    assert len(matching) == 1, (
        f"expected 1 destroy call, got {len(matching)}; "
        f"all destroy_calls={sandbox_lifecycle_spy.destroy_calls!r}"
    )

    # Assert: audit row marked processed_at.
    assert await audit_repo.get_processed(
        ctx.root_session_id, env.envelope_id
    ), "audit row missing processed_at after destroy"

    # Assert: stream PEL drained (both deliveries ACKed).
    pending = await redis_client.xpending(
        stream_key, "actus:mailbox-supervisor:v1"
    )
    # ``xpending`` summary form returns a dict ``{"pending": N, ...}``.
    # ``decode_responses=True`` (see conftest ``redis_client``) yields str
    # keys; we read either str or bytes defensively.
    pending_count = pending.get("pending", pending.get(b"pending", 0))
    assert pending_count == 0, f"PEL not drained; pending={pending!r}"
