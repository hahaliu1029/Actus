"""§13.4 — Mailbox supervisor concurrency invariants.

Plan reference: docs/superpowers/plans/2026-05-21-c3-mailbox-control-protocol.md
"R3 P1 fix" line 5115 — concurrency tests must be staged in PR-4.

Three scenarios cover the per-root single-writer invariant + dedup layers:

  T-conc-1 — two distinct children publish RESULT_READY concurrently to the
             same root. Each must be destroyed exactly once.
  T-conc-2 — same child + same envelope_id published twice concurrently.
             The publisher's SET NX layer (or the DB unique on
             (parent_session_id, envelope_id) audit row) must block the
             duplicate so destroy() runs exactly once and only one audit
             row has ``processed_at`` set.
  T-conc-3 — cascade orphan TERMINATE fires for N children whose
             last_seen has gone stale. All N must be destroyed; no leak.

All scenarios use the same ``full_supervisor_stack`` (one supervisor task
per root) — the per-root single-writer invariant carries the safety.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from app.domain.models.mailbox_envelope import (
    CostAggregate,
    MAILBOX_STREAM_KEY_TEMPLATE,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
    SUBAGENT_PROGRESS_STALE_AFTER_SECONDS,
)
from app.domain.models.session import DestroyReason
from app.infrastructure.models.session import SessionModel


def _make_result_ready(*, envelope_id, parent_id, child_id, ctx) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id=envelope_id,
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=parent_id,
        child_session_id=child_id,
        correlation_id=f"{envelope_id}-corr",
        emitted_at=ctx.now(),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=ResultReadyPayload(
            summary="done",
            outcome=ResultReadyOutcome.SUCCESS,
            cost_summary=CostAggregate(),
        ).model_dump(mode="json"),
    )


async def _seed_child(db_session, *, sample_user, parent_id, child_id):
    """Flush a child SessionModel under the given parent root."""
    row = SessionModel(
        id=child_id,
        user_id=sample_user.id,
        parent_session_id=parent_id,
        status="running",
        title="c3 ph conc child",
        task_id=child_id,
        execution_mode="foreground",
        execution_phase="running",
        retry_budget_remaining=3,
        was_background=False,
        worker_type="subagent",
        tool_filter_preset="subagent_research",
        subagent_control_plane="mailbox",
    )
    db_session.add(row)
    await db_session.flush()
    return row


@pytest.mark.integration
@pytest.mark.anyio
async def test_concurrent_result_ready_two_children_each_destroyed_once(
    full_supervisor_stack,
    sandbox_lifecycle_spy,
    db_session,
    sample_user,
):
    """Spec §13.4 — two distinct children RESULT_READY in parallel.

    Each child must be destroyed exactly once; no cross-child interference.
    """
    _supervisor, ctx, _audit, publisher, _task = full_supervisor_stack

    child_a_id = f"child-conc-A-{uuid.uuid4().hex[:8]}"
    child_b_id = f"child-conc-B-{uuid.uuid4().hex[:8]}"
    await _seed_child(
        db_session, sample_user=sample_user, parent_id=ctx.root_session_id,
        child_id=child_a_id,
    )
    await _seed_child(
        db_session, sample_user=sample_user, parent_id=ctx.root_session_id,
        child_id=child_b_id,
    )

    env_a = _make_result_ready(
        envelope_id="01HSPYU0CONC0A0000000000",
        parent_id=ctx.root_session_id,
        child_id=child_a_id,
        ctx=ctx,
    )
    env_b = _make_result_ready(
        envelope_id="01HSPYU0CONC0B0000000000",
        parent_id=ctx.root_session_id,
        child_id=child_b_id,
        ctx=ctx,
    )

    # asyncio.gather drives both publishes concurrently; the supervisor's
    # per-root single-writer serializes destruction so both eventually
    # destroy without interleaving.
    await asyncio.gather(publisher.publish(env_a), publisher.publish(env_b))
    await asyncio.sleep(1.5)

    a_calls = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_a_id
        and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
    ]
    b_calls = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_b_id
        and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
    ]
    assert len(a_calls) == 1, f"child A destroyed {len(a_calls)} times, expected 1"
    assert len(b_calls) == 1, f"child B destroyed {len(b_calls)} times, expected 1"


@pytest.mark.integration
@pytest.mark.anyio
async def test_concurrent_duplicate_envelope_id_destroys_child_once(
    full_supervisor_stack,
    sandbox_lifecycle_spy,
    child_session_in_db,
    redis_client,
):
    """Spec §13.4 + §5.8 — same envelope_id published twice in parallel.

    The publisher's SET NX dedup blocks the second XADD client-side; even
    if it raced through, the consumer-side ``get_processed`` check would
    short-circuit. Only ONE destroy + ONE audit ``processed_at``.
    """
    _supervisor, ctx, audit_repo, publisher, _task = full_supervisor_stack

    eid = "01HSPYU0CONCDUP00000000000"
    env = _make_result_ready(
        envelope_id=eid,
        parent_id=ctx.root_session_id,
        child_id=child_session_in_db.id,
        ctx=ctx,
    )

    # Two concurrent publishes with the same envelope_id.
    await asyncio.gather(publisher.publish(env), publisher.publish(env))
    await asyncio.sleep(1.5)

    matching = [
        c
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["session_id"] == child_session_in_db.id
        and c["reason"] == DestroyReason.SUBAGENT_TERMINAL_RESULT
    ]
    assert len(matching) == 1, (
        f"expected exactly 1 destroy for the deduped envelope; "
        f"got {len(matching)}"
    )

    # Audit row has exactly one ``processed_at``.
    raw = await audit_repo.fetch_raw(ctx.root_session_id, eid)
    assert raw, "audit row missing for the only delivered envelope"
    assert raw.get("processed_at") is not None

    # Stream contains exactly one entry for this envelope (publisher SET NX
    # blocked the 2nd XADD).
    stream_key = MAILBOX_STREAM_KEY_TEMPLATE.format(
        root_session_id=ctx.root_session_id
    )
    entries = await redis_client.xrange(stream_key)

    def _decode(v):
        return v.decode() if isinstance(v, (bytes, bytearray)) else v

    rr_entries = [
        e
        for e in entries
        if _decode(e[1].get(b"envelope_id") or e[1].get("envelope_id")) == eid
    ]
    assert len(rr_entries) == 1, (
        f"publisher dedup must block the 2nd XADD; "
        f"stream entries for {eid}={len(rr_entries)}"
    )


@pytest.mark.integration
@pytest.mark.anyio
async def test_cascade_orphan_terminate_destroys_all_n_children(
    full_supervisor_stack,
    sandbox_lifecycle_spy,
    db_session,
    sample_user,
    monkeypatch,
):
    """Spec §7.5 — when N children all go stale, cascade TERMINATE drives
    an ORPHAN_TIMEOUT destroy for each. No child leaks.

    We pre-seed ``supervisor._last_seen_mono`` with stale timestamps for N
    synthetic children (well before SUBAGENT_PROGRESS_STALE_AFTER_SECONDS),
    then drop the orphan tick interval to 0.05s. The supervisor's next
    iteration must publish N synthetic CANCEL_REQUEST(TERMINATE) cascades.

    Codex r3 [R3-3, HIGH TEST] — the orphan tick (R2-6 fix) threads
    ``DestroyReason.ORPHAN_TIMEOUT`` through the cascade payload so ops
    can distinguish orphan-triggered destroys from parent-initiated
    ``FORCE_TERMINATE`` cascades in the binding history. The assertion
    below MUST match the new reason.
    """
    supervisor, ctx, _audit, _publisher, _task = full_supervisor_stack

    # Drop the orphan check interval so the cascade tick runs quickly.
    supervisor._ORPHAN_CHECK_INTERVAL_S = 0.05

    n = 3
    child_ids = []
    for _ in range(n):
        cid = f"child-cascade-{uuid.uuid4().hex[:8]}"
        await _seed_child(
            db_session,
            sample_user=sample_user,
            parent_id=ctx.root_session_id,
            child_id=cid,
        )
        child_ids.append(cid)

    # Make every child appear stale: last_seen well before
    # SUBAGENT_PROGRESS_STALE_AFTER_SECONDS (default 90s).
    now = supervisor._ctx.clock()
    stale_offset = SUBAGENT_PROGRESS_STALE_AFTER_SECONDS + 30.0
    for cid in child_ids:
        supervisor._last_seen_mono[cid] = now - stale_offset

    # Wait for: orphan tick → cascade publish → supervisor reads cascade
    # → TERMINATE branch → destroy(ORPHAN_TIMEOUT) for each child. The R3-3
    # fix threads ``DestroyReason.ORPHAN_TIMEOUT`` through the cascade
    # payload (R2-6 producer-role guard); ops distinguishes orphan-driven
    # destroys from parent-initiated ``FORCE_TERMINATE`` cascades. The
    # assertion below uses the same enum.
    await asyncio.sleep(2.5)

    terminated = {
        c["session_id"]
        for c in sandbox_lifecycle_spy.destroy_calls
        if c["reason"] == DestroyReason.ORPHAN_TIMEOUT
        and c["session_id"] in child_ids
    }
    missing = set(child_ids) - terminated
    assert not missing, (
        f"orphan cascade must drive ORPHAN_TIMEOUT for all stale children — "
        f"missing: {missing}; destroy_calls={sandbox_lifecycle_spy.destroy_calls!r}"
    )
