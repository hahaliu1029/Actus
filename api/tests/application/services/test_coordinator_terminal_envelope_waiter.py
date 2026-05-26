"""C2 PR-3 §7.5 — CoordinatorTerminalEnvelopeWaiter direct unit tests.

Covers (r6 P2-2 fix): worker_node tests mock the waiter; this file pins the
waiter's own semantics — stream key derivation, waiter consumer group, terminal
predicate, asyncio.wait_for timeout, envelope round-trip via MailboxEnvelope.model_validate.

Plan ref: This file is a R6 follow-up addition beyond the original Task 3.5.5
file list in ``docs/superpowers/plans/2026-05-25-c2-coordinator-task-runner-plan.md``.
The waiter is shipped by Task 3.5.5; codex R6 flagged missing direct tests as
P2 and this file closes that gap. The plan should be amended in a follow-up to
include this test file in Task 3.5.5's deliverable list.
"""
from __future__ import annotations

import asyncio
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from app.application.services.coordinator_terminal_envelope_waiter import (
    CoordinatorTerminalEnvelopeWaiter,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)


def _env_dict(env_type: MailboxEnvelopeType, child_sid: str, payload: dict) -> dict:
    """Construct a wire-shape envelope dict (what subscriber yields)."""
    role = (
        ProducerRole.CHILD_AGENT.value
        if env_type in (MailboxEnvelopeType.RESULT_READY, MailboxEnvelopeType.CANCEL_ACK)
        else ProducerRole.PARENT_AGENT.value
    )
    return {
        "envelope_id": "e-1",
        "type": env_type.value,
        "parent_session_id": "p1",
        "child_session_id": child_sid,
        "correlation_id": "corr-r1",
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "producer_role": role,
        "payload": payload,
        "reclaim_count": 0,
    }


def _mk_subscriber(envelopes: list[dict]) -> AsyncMock:
    sub = AsyncMock()

    async def fake_consume(*, predicate, **_kwargs):
        for env in envelopes:
            if await predicate(env):
                yield env

    sub.consume = fake_consume
    sub.subscribe = AsyncMock()
    return sub


@pytest.mark.anyio
async def test_subscribe_uses_root_scoped_stream_and_waiter_group() -> None:
    """[r6 P2-2] Stream key is actus:child:{root}; group is coordinator:waiter:{cid}."""
    payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    sub = _mk_subscriber([_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", payload)])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), timeout=1.0,
    )
    sub.subscribe.assert_awaited_once()
    kwargs = sub.subscribe.await_args.kwargs
    assert kwargs["stream_key"] == "actus:child:root1:mailbox"
    assert kwargs["consumer_group"] == "coordinator:waiter:c1"
    assert kwargs["consumer_name"] == "waiter-c1"


@pytest.mark.anyio
async def test_filters_terminal_types_only() -> None:
    """[r6 P2-2] Non-terminal envelope types (e.g. PROGRESS_UPDATE) MUST be skipped."""
    progress_env = {
        "envelope_id": "e-progress", "type": "PROGRESS_UPDATE",
        "parent_session_id": "p1", "child_session_id": "c1",
        "correlation_id": "corr",
        "emitted_at": datetime.now(timezone.utc).isoformat(),
        "producer_role": "child_agent",
        "payload": {"kind": "heartbeat", "visibility": "hidden"},
    }
    ok_payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    terminal_env = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = _mk_subscriber([progress_env, terminal_env])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), timeout=1.0,
    )
    # Only the terminal envelope was yielded — progress was filtered.
    assert env.type == MailboxEnvelopeType.RESULT_READY


@pytest.mark.anyio
async def test_filters_other_child_sessions() -> None:
    """[r6 P2-2] Terminal envelope addressed to a DIFFERENT child MUST be skipped."""
    ok_payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    other_child = _env_dict(MailboxEnvelopeType.RESULT_READY, "OTHER-CHILD", ok_payload)
    my_child = _env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)
    sub = _mk_subscriber([other_child, my_child])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), timeout=1.0,
    )
    assert env.child_session_id == "c1"


@pytest.mark.anyio
async def test_timeout_raises_when_no_terminal_arrives() -> None:
    """[r6 P2-2] asyncio.wait_for fires after timeout when subscriber yields nothing."""
    sub = AsyncMock()

    async def never_yield(*, predicate, **_kwargs):
        # Sleep forever — emulate stream with no terminal envelope.
        await asyncio.sleep(60)
        if False:  # pragma: no cover
            yield  # type: ignore[unreachable]

    sub.consume = never_yield
    sub.subscribe = AsyncMock()
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    with pytest.raises(asyncio.TimeoutError) as ei:
        await waiter.await_terminal(
            child_session_id="c1", root_session_id="root1",
            cancel_event=asyncio.Event(), timeout=0.05,
        )
    assert "c1" in str(ei.value)


@pytest.mark.anyio
async def test_envelope_validates_to_mailbox_envelope_model() -> None:
    """[r6 P2-2] Returned envelope is a validated MailboxEnvelope (Pydantic model),
    not a raw dict — caller can read .type, .payload, .child_session_id, etc."""
    ok_payload = ResultReadyPayload(
        summary="done", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    sub = _mk_subscriber([_env_dict(MailboxEnvelopeType.RESULT_READY, "c1", ok_payload)])
    waiter = CoordinatorTerminalEnvelopeWaiter(subscriber=sub)
    env = await waiter.await_terminal(
        child_session_id="c1", root_session_id="root1",
        cancel_event=asyncio.Event(), timeout=1.0,
    )
    # Validated as MailboxEnvelope — outcome field reachable via payload dict.
    assert env.payload["outcome"] == "success"
    assert env.producer_role == ProducerRole.CHILD_AGENT
