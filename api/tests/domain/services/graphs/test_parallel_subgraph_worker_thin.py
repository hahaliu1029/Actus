"""C2 PR-3 §7.5 — parallel_execution_subgraph.worker_node unit tests.

Mocks CoordinatorTerminalEnvelopeWaiter; verifies outcome normalization
from RESULT_READY vs CANCEL_ACK final_state.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.services.graphs.parallel_execution_subgraph import worker_node


def _mk_envelope(env_type: MailboxEnvelopeType, payload_dict: dict) -> MailboxEnvelope:
    """Build a real MailboxEnvelope (round-trips through _validate_payload_matches_type)."""
    from datetime import datetime, timezone
    import uuid
    role = (
        ProducerRole.CHILD_AGENT
        if env_type in (MailboxEnvelopeType.RESULT_READY, MailboxEnvelopeType.CANCEL_ACK)
        else ProducerRole.PARENT_AGENT
    )
    return MailboxEnvelope(
        envelope_id=str(uuid.uuid4()),
        type=env_type,
        parent_session_id="p1",
        child_session_id="c1",
        correlation_id="corr",
        emitted_at=datetime.now(timezone.utc),
        producer_role=role,
        payload=payload_dict,
    )


def _state_send() -> dict:
    return {
        "work_unit_id": "wu1",
        "child_session_id": "c1",
        "coordinator_run_id": "r1",
        "root_session_id": "root1",
    }


def _config(*, envelope: MailboxEnvelope) -> dict:
    waiter = AsyncMock()
    waiter.await_terminal = AsyncMock(return_value=envelope)
    return {
        "configurable": {
            "terminal_waiter": waiter,
            "cancel_event": asyncio.Event(),
        }
    }


@pytest.mark.anyio
async def test_worker_node_result_ready_success() -> None:
    payload = ResultReadyPayload(
        summary="done", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    config = _config(envelope=env)
    result = await worker_node(_state_send(), config)
    assert len(result["worker_results"]) == 1
    assert result["worker_results"][0].outcome == ResultReadyOutcome.SUCCESS


@pytest.mark.anyio
async def test_worker_node_result_ready_failed() -> None:
    payload = ResultReadyPayload(
        summary="oops", outcome=ResultReadyOutcome.FAILED,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    assert result["worker_results"][0].outcome == ResultReadyOutcome.FAILED


@pytest.mark.anyio
async def test_worker_node_result_ready_timed_out() -> None:
    payload = ResultReadyPayload(
        summary="budget exceeded", outcome=ResultReadyOutcome.TIMED_OUT,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    assert result["worker_results"][0].outcome == ResultReadyOutcome.TIMED_OUT


@pytest.mark.anyio
async def test_worker_node_cancel_ack_cancelled_normalizes_to_cancelled() -> None:
    payload = CancelAckPayload(final_state="cancelled").model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.CANCEL_ACK, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    assert result["worker_results"][0].outcome == ResultReadyOutcome.CANCELLED


# ---------------------------------------------------------------------------
# [r4 P1] PR-4 wire-schema field propagation into WorkerResult
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_worker_node_propagates_patch_manifest() -> None:
    """[r4 P1] RESULT_READY(SUCCESS) with PatchManifest → WorkerResult.patch_manifest
    is set. Without this, PR-5 reducer would see None and never apply patches."""
    from app.domain.models.patch_manifest import FilePatchEntry, PatchManifest
    pm = PatchManifest(
        patch_id="r1:wu1:p", coordinator_run_id="r1", work_unit_id="wu1",
        files=(FilePatchEntry(
            path="a/b.py", op="add",
            new_digest="a" * 64, content_ref="minio://r", content_size=10,
        ),),
    )
    payload = ResultReadyPayload(
        summary="done", outcome=ResultReadyOutcome.SUCCESS, patch_manifest=pm,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    wr = result["worker_results"][0]
    assert wr.patch_manifest is not None
    assert wr.patch_manifest.patch_id == "r1:wu1:p"
    assert wr.summary == "done"


@pytest.mark.anyio
async def test_worker_node_propagates_needs_authorization_details() -> None:
    """[r4 P1] RESULT_READY(NEEDS_AUTH) with details → WorkerResult.needs_authorization_details
    set. Without this, PR-5 reducer can't route NEEDS_AUTH to the human."""
    from app.domain.models.needs_authorization_details import (
        NeedsAuthorizationDetails,
    )
    details = NeedsAuthorizationDetails(reason="hard_blocked")
    payload = ResultReadyPayload(
        summary="blocked", outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
        needs_authorization_details=details,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    wr = result["worker_results"][0]
    assert wr.needs_authorization_details is not None
    assert wr.needs_authorization_details.reason == "hard_blocked"


@pytest.mark.anyio
async def test_worker_node_propagates_cost_summary() -> None:
    """[r4 P1] cost_summary must reach WorkerResult so PR-6 cost aggregator
    can sum per-child totals into the parent coordinator's run budget."""
    payload = ResultReadyPayload(
        summary="done", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    wr = result["worker_results"][0]
    assert wr.cost_summary is not None


@pytest.mark.anyio
async def test_worker_node_cancel_ack_propagates_summary() -> None:
    """[r4 P1 corollary] CANCEL_ACK summary surfaces for SSE/audit (PR-8)."""
    payload = CancelAckPayload(
        final_state="cancelled", summary="parent_cancel",
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.CANCEL_ACK, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    wr = result["worker_results"][0]
    assert wr.outcome == ResultReadyOutcome.CANCELLED
    assert wr.summary == "parent_cancel"


@pytest.mark.anyio
async def test_worker_node_cancel_ack_force_terminated_normalizes_to_timed_out() -> None:
    payload = CancelAckPayload(final_state="force_terminated").model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.CANCEL_ACK, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    assert result["worker_results"][0].outcome == ResultReadyOutcome.TIMED_OUT


@pytest.mark.anyio
async def test_worker_node_cancel_ack_completed_falls_back_to_failed() -> None:
    """final_state == 'completed' race: PR-3 fallback FAILED; PR-7 will lookup RESULT_READY store."""
    payload = CancelAckPayload(final_state="completed").model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.CANCEL_ACK, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    assert result["worker_results"][0].outcome == ResultReadyOutcome.FAILED


@pytest.mark.anyio
async def test_worker_node_passes_correct_child_and_root_to_waiter() -> None:
    payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    config = _config(envelope=env)
    await worker_node(_state_send(), config)
    waiter = config["configurable"]["terminal_waiter"]
    waiter.await_terminal.assert_awaited_once()
    kwargs = waiter.await_terminal.await_args.kwargs
    assert kwargs["child_session_id"] == "c1"
    assert kwargs["root_session_id"] == "root1"
    assert "timeout" not in kwargs


@pytest.mark.anyio
async def test_worker_result_carries_work_unit_and_child_ids() -> None:
    payload = ResultReadyPayload(
        summary="ok", outcome=ResultReadyOutcome.SUCCESS,
    ).model_dump(mode="python")
    env = _mk_envelope(MailboxEnvelopeType.RESULT_READY, payload)
    result = await worker_node(_state_send(), _config(envelope=env))
    wr = result["worker_results"][0]
    assert wr.work_unit_id == "wu1"
    assert wr.child_session_id == "c1"
