"""T7 — APPROVAL_REQUEST stub deny + correlation_id integrity + dedup.

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
§"Step 12: T7 — approval correlation_id mismatch + duplicate response" lines
4937-5028.

Scope (post-PR-4 stub handler):
  * ``ApprovalRequestHandler`` emits an immediate deny APPROVAL_RESPONSE
    (decided_by = auto_policy) because PE-2 integration is not yet wired.
  * Paired correlation_id MUST round-trip: response's payload.correlation_id
    matches the request's payload.correlation_id.
  * Duplicate APPROVAL_REQUEST with the same envelope_id is short-circuited
    on TWO layers:
      - Publisher layer (SET NX) — second ``publisher.publish`` blocks the
        XADD before it hits the stream.
      - Supervisor consumer layer (audit dedup) — if the same envelope is
        XADDed bypassing the publisher (e.g., from a different pod / a
        manual XAUTOCLAIM redelivery scenario), the ``get_processed``
        short-circuit in ``_handle_envelope`` MUST detect the existing
        ``processed_at`` row and ACK without re-dispatching to the handler.

Codex F4 (HIGH) — added ``test_supervisor_side_audit_dedup_short_circuits``
because the original "duplicate" test only covered the publisher layer.
The supervisor-side dedup is the load-bearing redelivery guard that
guarantees at-most-once handler invocation across pod restarts.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.domain.models.mailbox_envelope import (
    APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
    ApprovalRequestPayload,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)


def _decode_field(value):
    """Normalize redis_client field values that may be bytes or str."""
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


@pytest.mark.integration
@pytest.mark.anyio
async def test_approval_request_emits_paired_deny_response(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """Spec §10.2 — APPROVAL_REQUEST → paired APPROVAL_RESPONSE (deny)."""
    supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    req_env = MailboxEnvelope(
        envelope_id="01HSPYU0t7a00000000000000",
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="01HSPYU0t7a00000000000001",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "ls"},
            risk_tier="medium",
            rationale="example",
            correlation_id="01HSPYU0t7a00000000000001",
            tool_call_id="tc-abc",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )
    await publisher.publish(req_env)
    await asyncio.sleep(0.5)

    # Find APPROVAL_RESPONSE in the stream — supervisor publishes one
    # via its stub handler.
    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    entries = await redis_client.xrange(stream_key)
    responses = [
        e
        for e in entries
        if _decode_field(e[1].get(b"type") or e[1].get("type"))
        == MailboxEnvelopeType.APPROVAL_RESPONSE.value
    ]
    assert len(responses) == 1, (
        f"stub handler must emit exactly one paired response — got "
        f"{len(responses)}; all entries={entries!r}"
    )
    raw = responses[0][1].get(b"envelope") or responses[0][1].get("envelope")
    payload_doc = json.loads(_decode_field(raw))
    assert payload_doc["payload"]["correlation_id"] == "01HSPYU0t7a00000000000001"
    assert payload_doc["payload"]["approved"] is False
    assert payload_doc["payload"]["decided_by"] == "auto_policy"


@pytest.mark.integration
@pytest.mark.anyio
async def test_duplicate_approval_request_dedups_via_audit(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """Repeated APPROVAL_REQUEST with same envelope_id → 1 audit row, 1 response.

    The publisher's SET NX dedup layer blocks the second XADD before it
    reaches the stream, so the supervisor sees only one APPROVAL_REQUEST
    and emits only one APPROVAL_RESPONSE.
    """
    supervisor, ctx, audit_repo, publisher, _task = full_supervisor_stack

    eid = "01HSPYU0t7b00000000000000"
    req = MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="cid-dup",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "ls"},
            risk_tier="medium",
            rationale="ex",
            correlation_id="cid-dup",
            tool_call_id="tc-dup",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )

    await publisher.publish(req)
    await publisher.publish(req)  # SET NX blocks the 2nd XADD client-side
    await asyncio.sleep(0.5)

    raw = await audit_repo.fetch_raw(ctx.root_session_id, eid)
    assert raw, "expected exactly one audit row for the deduped envelope"
    assert raw.get("processed_at") is not None, (
        "stub handler ACKs via outcome.ack=True → mark_processed should run"
    )

    # Stream contains exactly one APPROVAL_REQUEST entry (publisher SET NX
    # blocked the 2nd) AND exactly one APPROVAL_RESPONSE (one handler run).
    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    entries = await redis_client.xrange(stream_key)
    requests = [
        e
        for e in entries
        if _decode_field(e[1].get(b"type") or e[1].get("type"))
        == MailboxEnvelopeType.APPROVAL_REQUEST.value
        and (
            _decode_field(
                e[1].get(b"envelope_id") or e[1].get("envelope_id")
            )
            == eid
        )
    ]
    assert len(requests) == 1, (
        f"publisher SET NX must dedup the second XADD; "
        f"requests for {eid}={len(requests)}"
    )
    responses = [
        e
        for e in entries
        if _decode_field(e[1].get(b"type") or e[1].get("type"))
        == MailboxEnvelopeType.APPROVAL_RESPONSE.value
    ]
    assert len(responses) == 1, (
        f"stub handler must emit exactly one paired response; "
        f"got {len(responses)}"
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_supervisor_side_audit_dedup_short_circuits(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """Codex F4 (HIGH) — supervisor consumer-side audit dedup short-circuits
    a redelivered APPROVAL_REQUEST without re-dispatching to the handler.

    The previous "duplicate" test only exercised the publisher's SET NX
    dedup (second ``publisher.publish`` blocks XADD client-side). This
    test exercises the supervisor-side dedup by bypassing the publisher
    entirely on the second message: we use raw XADD to push the same
    envelope onto the stream after the first dispatch has completed +
    written ``processed_at``. The ``get_processed`` check at the top of
    ``_handle_envelope`` must detect the existing row and ACK without
    publishing a second APPROVAL_RESPONSE.

    The raw-XADD pattern mirrors ``test_mailbox_idempotency.py:81-89``
    (T1's RESULT_READY redelivery path).
    """
    supervisor, ctx, audit_repo, publisher, _task = full_supervisor_stack

    eid = "01HSPYU0F40000APPROVAL0DEDUP"
    req = MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="cid-f4",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "ls"},
            risk_tier="medium",
            rationale="codex-f4 supervisor-side dedup",
            correlation_id="cid-f4",
            tool_call_id="tc-f4",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )

    # First publish via the normal publisher — supervisor processes once,
    # writes processed_at, emits one APPROVAL_RESPONSE.
    await publisher.publish(req)
    # Wait for processed_at to land before redelivery.
    for _ in range(50):
        await asyncio.sleep(0.1)
        if await audit_repo.get_processed(ctx.root_session_id, eid):
            break
    assert await audit_repo.get_processed(ctx.root_session_id, eid), (
        "first publish must complete before redelivery"
    )

    # Raw XADD bypasses publisher SET NX so the supervisor sees a fresh
    # delivery for the same envelope_id. Fields shape mirrors
    # RedisMailboxPublisher.publish (test_mailbox_idempotency.py pattern).
    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    await redis_client.xadd(
        stream_key,
        fields={
            "envelope": req.model_dump_json(),
            "envelope_id": req.envelope_id,
            "type": req.type.value,
            "producer_role": req.producer_role.value,
        },
    )
    # Give the supervisor a window to dispatch the redelivery.
    await asyncio.sleep(0.6)

    # Assert: still exactly ONE APPROVAL_RESPONSE in the stream —
    # supervisor's audit dedup short-circuited the second delivery.
    entries = await redis_client.xrange(stream_key)
    responses = [
        e
        for e in entries
        if _decode_field(e[1].get(b"type") or e[1].get("type"))
        == MailboxEnvelopeType.APPROVAL_RESPONSE.value
    ]
    assert len(responses) == 1, (
        f"supervisor-side audit dedup must suppress the second handler "
        f"invocation; got {len(responses)} APPROVAL_RESPONSEs"
    )

    # Assert: stream PEL drained (both deliveries ACKed).
    pending = await redis_client.xpending(
        stream_key, "actus:mailbox-supervisor:v1"
    )
    pending_count = pending.get("pending", pending.get(b"pending", 0))
    assert pending_count == 0, (
        f"PEL not drained after dedup short-circuit; pending={pending!r}"
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_approval_request_with_mismatched_correlation_id_is_rejected(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """codex r4 [R4-2] + codex r6 [R6-4, HIGH CONTRACT] — T7 invariant:
    envelope.correlation_id MUST equal payload.correlation_id for
    APPROVAL_REQUEST. R4-2's first iteration ACK+dropped without
    publishing; R6-4 changes that to ACK + publish an immediate deny
    keyed to envelope.correlation_id (the supervisor's trusted source)
    so the child unblocks immediately instead of waiting the 300s
    timeout.
    """
    supervisor, ctx, audit_repo, publisher, _task = full_supervisor_stack

    eid = "01HSPYU0R42INTEG0APPROVAL0CID"
    # Build the envelope through the public model, then bypass the
    # publisher to mutate payload.correlation_id away from
    # envelope.correlation_id. The publisher refuses to send a malformed
    # envelope, so the test directly XADDs the raw bytes the way a
    # buggy / hostile producer would.
    req = MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id="envelope-cid-good",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "ls"},
            risk_tier="medium",
            rationale="r4-2 mismatch integration test",
            correlation_id="payload-cid-bad",  # ← deliberately diverges
            tool_call_id="tc-r42",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )
    assert (
        req.correlation_id != req.payload["correlation_id"]
    ), "test precondition: envelope and payload correlation_ids must differ"

    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    await redis_client.xadd(
        stream_key,
        fields={
            "envelope": req.model_dump_json(),
            "envelope_id": req.envelope_id,
            "type": req.type.value,
            "producer_role": req.producer_role.value,
        },
    )
    await asyncio.sleep(0.6)

    # Audit row exists + processed_at written (ACK+drop path).
    raw = await audit_repo.fetch_raw(ctx.root_session_id, eid)
    assert raw is not None
    assert raw.get("processed_at") is not None

    # codex r6 [R6-4] — APPROVAL_RESPONSE IS published, keyed to
    # envelope.correlation_id (NOT the bogus payload one).
    import json as _json

    entries = await redis_client.xrange(stream_key)
    responses = [
        e
        for e in entries
        if _decode_field(e[1].get(b"type") or e[1].get("type"))
        == MailboxEnvelopeType.APPROVAL_RESPONSE.value
    ]
    assert len(responses) == 1, (
        f"R6-4 — mismatched correlation_id must yield exactly one deny "
        f"response keyed to envelope.correlation_id; got "
        f"{len(responses)} responses"
    )
    resp_raw = responses[0][1].get(b"envelope") or responses[0][1].get(
        "envelope"
    )
    resp_blob = resp_raw.decode() if isinstance(resp_raw, bytes) else resp_raw
    resp_env = _json.loads(resp_blob)
    assert resp_env["correlation_id"] == "envelope-cid-good", (
        f"R6-4 — deny envelope must be keyed to envelope.correlation_id "
        f"(trusted source), got {resp_env['correlation_id']!r}"
    )
    assert resp_env["payload"]["correlation_id"] == "envelope-cid-good"
    assert resp_env["payload"]["approved"] is False
    assert resp_env["payload"]["decided_by"] == "auto_policy"
    assert "correlation_id_mismatch" in resp_env["payload"]["reason"]

    # PEL drained.
    pending = await redis_client.xpending(
        stream_key, "actus:mailbox-supervisor:v1"
    )
    pending_count = pending.get("pending", pending.get(b"pending", 0))
    assert pending_count == 0, (
        f"PEL not drained after mismatch publish-deny+drop; pending={pending!r}"
    )
