"""T9 — APPROVAL_REQUEST → stub handler → paired APPROVAL_RESPONSE (correlation match).

**Scope decision (codex r4 [R4-1, HIGH TEST] — Option C)**: this file
ships the envelope-bridge half of T9 plus the in-process callback
dispatch assertion. The full LangGraph half (real `interrupt()` →
`Command(resume=...)` round-trip + LangGraph issue #6792 idempotency
invariant) is deferred to PE-2 (Permission Engine Phase 2 — not yet
integrated). PE-2 ships the LangGraph node that consumes the mailbox
APPROVAL_RESPONSE via ``interrupt()`` / ``Command(resume=...)``; PR-4
cannot meaningfully exercise that round-trip because the consuming
node does not yet exist in tree. The PE-2 plan carries forward the
LangGraph round-trip + #6792 side-effect-after-interrupt idempotency
assertions (see PE-2 milestone).

What PR-4 verifies here (envelope-bridge contract):

1. APPROVAL_REQUEST envelope → supervisor stub handler emits a paired
   APPROVAL_RESPONSE.
2. ``payload.correlation_id`` round-trips from request to response.
3. ``decided_by=auto_policy`` (deny-by-default policy honored).
4. The supervisor's in-process dispatch surface (the path PE-2 will
   replace with a LangGraph ``Command(resume=...)`` wrapper) is
   actually invoked: the supervisor re-reads its own APPROVAL_RESPONSE
   via XREADGROUP and routes it through ``_StubNonTerminalHandler`` →
   ``agent_service_callback``. The callback assertion is the regression
   guard that survives PE-2: the LangGraph layer that PE-2 introduces
   plugs in as a new callback target, but the contract that
   APPROVAL_RESPONSE envelopes reach the callback at all is the
   bridge invariant being locked here.

What PR-4 does NOT verify (deferred to PE-2):
- Real LangGraph node running ``interrupt()`` inside a child agent
- ``await`` inside the node consuming the mailbox response
- ``Command(resume=...)`` re-entry into the graph
- LangGraph #6792 side-effect-after-interrupt idempotency

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
§"R3 P1 fix" line 5115 — T9 must be staged in PR-4 even though the
implementation is a stub.

Spec reference: docs/superpowers/specs/2026-05-21-c3-mailbox-control-protocol-design.md
§13.5 — T9 acceptance criteria (full version covers the LangGraph
round-trip; this file ships the bridge half).

This test deliberately overlaps with ``test_mailbox_approval.py`` (T7)
but exists as a separate file so the C3 PR-4 test inventory matches
the plan's stage-list one-for-one. Once PE-2 lands, this file is where
the real bridge round-trip test goes; T7 retains the correlation_id +
deny-stub assertions.
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
    if isinstance(value, (bytes, bytearray)):
        return value.decode()
    return value


@pytest.mark.integration
@pytest.mark.anyio
async def test_hitl_bridge_stub_handler_emits_paired_response(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """APPROVAL_REQUEST envelope → stub handler → APPROVAL_RESPONSE with
    matching correlation_id, ``approved=False``, ``decided_by='auto_policy'``.

    This is the C3-shippable bridge contract. PE-2 will swap the stub for
    a real interrupt + resume round-trip; this assertion shape stays valid
    (the request/response correlation invariant is unchanged).
    """
    _supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    correlation_id = "01HSPYU0t9b00000000000001"
    req_env = MailboxEnvelope(
        envelope_id="01HSPYU0t9b00000000000000",
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id=correlation_id,
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "echo hitl"},
            risk_tier="high",
            rationale="hitl bridge envelope round-trip",
            correlation_id=correlation_id,
            tool_call_id="tc-hitl-1",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )
    await publisher.publish(req_env)
    await asyncio.sleep(0.5)

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
        f"bridge must emit exactly one paired APPROVAL_RESPONSE; "
        f"got {len(responses)}"
    )
    raw = responses[0][1].get(b"envelope") or responses[0][1].get("envelope")
    response_doc = json.loads(_decode_field(raw))
    payload = response_doc["payload"]
    assert payload["correlation_id"] == correlation_id, (
        "correlation_id MUST round-trip from request to response"
    )
    assert payload["approved"] is False
    assert payload["decided_by"] == "auto_policy"


@pytest.mark.integration
@pytest.mark.anyio
async def test_hitl_bridge_in_process_callback_receives_approval_response(
    full_supervisor_stack,
    redis_client,
    child_session_in_db,
):
    """Codex r4 [R4-1, HIGH TEST] — verify the supervisor's in-process
    dispatch surface (the path PE-2 will replace with a LangGraph
    ``Command(resume=...)`` wrapper) is actually invoked.

    Flow under test:
        child publishes APPROVAL_REQUEST
            → supervisor stub handler emits paired APPROVAL_RESPONSE
            → supervisor re-reads APPROVAL_RESPONSE via XREADGROUP
            → ``_StubNonTerminalHandler`` routes it to
              ``ctx.agent_service_callback`` with the response envelope
            → fixture's ``_StubAgentCallback.received`` captures it

    The callback dispatch is the integration hook PE-2 will swap for
    a real LangGraph resume orchestrator. Locking it here means the
    PE-2 PR can verify the supervisor-side bridge is unchanged without
    re-running the envelope round-trip.

    Without this assertion, the supervisor could (in principle) publish
    a correctly-shaped APPROVAL_RESPONSE but fail to dispatch it
    in-process — and T9's response-shape assertion alone wouldn't
    catch the gap.

    Companion deferred-test doc:
    ``docs/superpowers/follow-ups/c3-pr4-deferred-tests.md`` — describes
    what the LangGraph half of the bridge round-trip covers and when
    PE-2 picks it up.
    """
    _supervisor, ctx, _audit, publisher, _task = full_supervisor_stack
    # The integration fixture exposes the callback via ``ctx.agent_callback``
    # alias (see conftest line ~1155). Use that to spy on the response
    # dispatch.
    callback = ctx.agent_callback  # type: ignore[attr-defined]
    # Drain any pre-test envelopes the callback may have captured
    # (each test gets a fresh ``root_session_id`` so this is normally
    # empty, but the fixture's tick=0.1s window means a stale tick
    # could in theory leak something — be defensive).
    callback.received.clear()

    correlation_id = "01HSPYU0R410CALLBACK0RESPONSE"
    req_env = MailboxEnvelope(
        envelope_id="01HSPYU0R410CALLBACK0REQUEST",
        type=MailboxEnvelopeType.APPROVAL_REQUEST,
        parent_session_id=ctx.root_session_id,
        child_session_id=child_session_in_db.id,
        correlation_id=correlation_id,
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ApprovalRequestPayload(
            tool_name="shell_execute",
            tool_args_snapshot={"cmd": "echo r4-1 bridge"},
            risk_tier="high",
            rationale="r4-1 in-process callback dispatch assertion",
            correlation_id=correlation_id,
            tool_call_id="tc-r41-callback",
            requested_at=ctx.now(),
            timeout_seconds=APPROVAL_REQUEST_TIMEOUT_DEFAULT_SECONDS,
        ).model_dump(mode="json"),
    )
    await publisher.publish(req_env)

    # The supervisor's loop runs at ``idle_poll_sleep_s=0.05`` (see
    # conftest line ~1162). Two cycles minimum: one to dispatch the
    # APPROVAL_REQUEST (which publishes the response), one to dispatch
    # the supervisor's own APPROVAL_RESPONSE back through XREADGROUP
    # to the callback. Poll up to 5s before giving up.
    response_envelopes: list = []
    for _ in range(100):  # 100 * 0.05s = 5s ceiling
        await asyncio.sleep(0.05)
        response_envelopes = [
            e
            for e in callback.received
            if getattr(e, "type", None)
            == MailboxEnvelopeType.APPROVAL_RESPONSE
            and e.correlation_id == correlation_id
        ]
        if response_envelopes:
            break

    assert response_envelopes, (
        "supervisor must dispatch its own APPROVAL_RESPONSE to "
        "``agent_service_callback`` via the in-process bridge — this is "
        "the surface PE-2 will swap for LangGraph resume. callback."
        f"received={[getattr(e, 'envelope_id', None) for e in callback.received]!r}"
    )
    assert len(response_envelopes) == 1, (
        f"exactly one paired APPROVAL_RESPONSE must reach the callback; "
        f"got {len(response_envelopes)}"
    )

    received = response_envelopes[0]
    # Verify the dispatched envelope's payload carries the same
    # correlation_id the request emitted (the same invariant the
    # envelope-bridge test above checks, but now from the callback
    # side of the dispatch boundary).
    assert received.payload["correlation_id"] == correlation_id
    assert received.payload["approved"] is False
    assert received.payload["decided_by"] == "auto_policy"
    # The supervisor produced this envelope (not the child), so
    # ``producer_role=SUPERVISOR`` per ApprovalRequestHandler.handle.
    assert received.producer_role == ProducerRole.SUPERVISOR
