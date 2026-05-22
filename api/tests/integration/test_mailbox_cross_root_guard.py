"""R3-1 — cross-root child_session_id publisher contract (negative coverage).

codex r3 [R3-1, CRITICAL ARCH] decision: keep the cross-root verification as
a *publisher contract* rather than supervisor-side defensive ancestor
traversal (see ``mailbox_supervisor._handle_envelope`` cross-root guard
docstring for the rationale). This test makes the trust assumption
explicit:

    *Given* a publisher that correctly routes ``parent_session_id`` to
    the supervisor's stream key but mistakenly sets ``child_session_id``
    to a session belonging to a different root, *then* the supervisor
    proceeds with dispatch — the cross-root invariant is publisher
    responsibility, not supervisor responsibility.

If a future PR adds defensive cross-root verification (TODO ticket noted
in ``_handle_envelope``), this test must flip to assert that the
supervisor refuses dispatch. Today it documents the load-bearing publisher
contract via a green test.

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
codex r3 finding R3-1 in this PR's review notes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)


@pytest.mark.integration
@pytest.mark.anyio
async def test_supervisor_trusts_publisher_for_child_root_ownership(
    full_supervisor_stack,
    sample_user,
    db_session,
):
    """Negative coverage: publisher-contract trust assumption.

    Scenario: a publisher correctly routes the envelope to the supervisor's
    stream (``parent_session_id == ctx.root_session_id``) but mistakenly
    sets ``child_session_id`` to a child whose actual ancestor is a
    DIFFERENT root row (R-OTHER). The supervisor today does NOT verify
    the child's ancestor chain, so the envelope WILL be dispatched
    (forwarded to ``agent_service_callback``).

    Assertions:
      - ``parent_session_id`` matches the supervisor's root (so the
        existing R8 [P1] cross-root guard on ``parent_session_id`` does
        NOT bite).
      - The agent_service_callback IS invoked even though the child
        belongs to a different root — proving the supervisor proceeds.

    If/when defensive verification lands (PR-5/PR-6 acceptance gate),
    invert the assertion: callback NOT invoked, envelope ACK-drained,
    warning logged.
    """
    _supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    # Create a SECOND root row (R-OTHER) and a child belonging to it.
    # The supervisor does NOT own this child's lineage, but we are
    # going to publish an envelope claiming to be from this child onto
    # the supervisor's stream.
    from app.infrastructure.models.session import SessionModel
    import uuid as _uuid

    other_root_id = f"root-other-{_uuid.uuid4().hex[:12]}"
    other_child_id = f"child-other-{_uuid.uuid4().hex[:12]}"
    other_root = SessionModel(
        id=other_root_id,
        user_id=sample_user.id,
        status="running",
        title="r3-1 negative test other root",
        task_id=other_root_id,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="root",
    )
    other_child = SessionModel(
        id=other_child_id,
        user_id=sample_user.id,
        parent_session_id=other_root_id,
        status="running",
        title="r3-1 negative test child belonging to other root",
        task_id=other_child_id,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="subagent",
        subagent_control_plane="mailbox",
    )
    db_session.add(other_root)
    await db_session.flush()
    db_session.add(other_child)
    await db_session.flush()

    # Publish PROGRESS_UPDATE (non-terminal) so the test stays cheap —
    # we just need to observe whether the supervisor's callback is
    # invoked. parent_session_id matches the supervisor's root, so the
    # routing guard passes. child_session_id belongs to OTHER root.
    forged = MailboxEnvelope(
        envelope_id=f"env-r3-1-{_uuid.uuid4().hex[:12]}",
        type=MailboxEnvelopeType.PROGRESS_UPDATE,
        # parent_session_id MATCHES the supervisor — the existing R8 [P1]
        # guard does NOT bite.
        parent_session_id=ctx.root_session_id,
        # child_session_id belongs to a DIFFERENT root — the supervisor
        # does NOT verify this; it is publisher responsibility.
        child_session_id=other_child_id,
        correlation_id=f"corr-r3-1-{_uuid.uuid4().hex[:12]}",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload={"kind": "heartbeat", "visibility": "hidden"},
    )
    await publisher.publish(forged)

    # Drain a few ticks so the supervisor reads + dispatches.
    for _ in range(20):
        await asyncio.sleep(0.05)
        if any(
            e.envelope_id == forged.envelope_id
            for e in ctx.agent_service_callback.received
        ):
            break

    # Assertion: supervisor TRUSTED the publisher and forwarded the
    # envelope to the callback even though child_session_id belongs to a
    # different root. This documents the load-bearing publisher contract.
    received_ids = [
        e.envelope_id for e in ctx.agent_service_callback.received
    ]
    assert forged.envelope_id in received_ids, (
        f"R3-1 negative test — supervisor must trust the publisher for "
        f"child_session_id ownership today. If this assertion now fails, "
        f"the PR-5/PR-6 defensive verification TODO has likely been "
        f"implemented; flip this assertion to check the callback was "
        f"NOT invoked + warning logged. callback received={received_ids!r}"
    )
