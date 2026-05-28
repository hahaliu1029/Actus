"""C2 PR-8 §13 Task 8.4 — CoordinatorRunOrchestrator emit tests.

Asserts the PR-6 placeholder-dict emit is replaced by a typed
CoordinatorSiblingCancelEvent, and that the orchestrator skips emit
when ``outcome is None`` (cancel-ack-only path) because the event
schema requires a non-None ResultReadyOutcome.
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

import pytest
from unittest.mock import AsyncMock

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.application.services.coordinator_run_orchestrator import (
    CoordinatorRunOrchestrator,
)
from app.domain.models.event import CoordinatorSiblingCancelEvent
from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    MailboxEnvelope,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
)


def _result_ready_env(
    outcome: ResultReadyOutcome,
    *,
    child_session_id: str = "c1",
    needs_auth: Optional[NeedsAuthorizationDetails] = None,
) -> MailboxEnvelope:
    payload = ResultReadyPayload(
        summary="x",
        outcome=outcome,
        needs_authorization_details=needs_auth,
    )
    return CoordinatorEnvelopeFactory().make_result_ready(
        parent_session_id="p1",
        child_session_id=child_session_id,
        correlation_id="r1",
        payload=payload,
    )


class _FakeSubscriber:
    def __init__(
        self,
        envelopes: list[MailboxEnvelope],
        *,
        exhaust_then_return: bool = False,
    ) -> None:
        self._dicts = [env.model_dump(mode="json") for env in envelopes]
        self._exhaust_then_return = exhaust_then_return

    async def subscribe(self, **_: Any) -> None:
        return None

    async def consume(self, **_: Any):
        for d in self._dicts:
            yield d
        if self._exhaust_then_return:
            return
        await asyncio.sleep(60)  # block forever

    async def destroy_group(self, **_: Any) -> None:
        return None


@pytest.mark.anyio
async def test_emit_event_is_typed_coordinator_sibling_cancel_event() -> None:
    """When a sibling-cancel cascade fires, emit_event receives a
    CoordinatorSiblingCancelEvent — not the PR-6 placeholder dict."""
    publisher = AsyncMock()
    emitted: list[Any] = []

    async def emit(event: Any) -> None:
        emitted.append(event)

    env = _result_ready_env(
        ResultReadyOutcome.FAILED, child_session_id="c1",
    )
    subscriber = _FakeSubscriber([env], exhaust_then_return=True)
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        mailbox_subscriber=subscriber,
        parent_session_id="p1", coordinator_run_id="r1",
        emit_event=emit,
    )
    cancel_event = asyncio.Event()
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu2"],
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        cancel_event=cancel_event, timeout_seconds=1.0,
    )
    assert len(emitted) == 1
    ev = emitted[0]
    assert isinstance(ev, CoordinatorSiblingCancelEvent)
    assert ev.coordinator_run_id == "r1"
    assert ev.triggered_by_work_unit_id == "wu1"
    assert ev.triggered_by_outcome == ResultReadyOutcome.FAILED
    assert ev.cancelled_work_unit_ids == ["wu2"]
    # ``reason`` mirrors the publisher's ``sibling_terminal_<outcome>``.
    assert ev.reason == "sibling_terminal_failed"


@pytest.mark.anyio
async def test_emit_event_skipped_when_outcome_is_none() -> None:
    """If ``_fan_out_sibling_cancel`` is reached with ``outcome=None``
    (defensive — observer loop normally extracts a real outcome) we
    skip emit because ``CoordinatorSiblingCancelEvent.triggered_by_outcome``
    is required-non-None."""
    publisher = AsyncMock()
    emitted: list[Any] = []

    async def emit(event: Any) -> None:
        emitted.append(event)

    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
        emit_event=emit,
    )
    # Call _fan_out_sibling_cancel directly with outcome=None — siblings
    # exist so the publish fan-out runs; only emit should be suppressed.
    await orch._fan_out_sibling_cancel(
        triggered_by_wu="wu1",
        outcome=None,
        pending={"wu2"},
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        coordinator_run_id="r1",
    )
    assert emitted == []


@pytest.mark.anyio
async def test_emit_callable_failure_does_not_propagate() -> None:
    """If emit_event raises, the orchestrator logs + swallows so the
    sibling-cancel path still completes."""
    publisher = AsyncMock()

    async def emit(_: Any) -> None:
        raise RuntimeError("emit broke")

    env = _result_ready_env(
        ResultReadyOutcome.FAILED, child_session_id="c1",
    )
    subscriber = _FakeSubscriber([env], exhaust_then_return=True)
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        mailbox_subscriber=subscriber,
        parent_session_id="p1", coordinator_run_id="r1",
        emit_event=emit,
    )
    cancel_event = asyncio.Event()
    # Must not raise.
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu2"],
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        cancel_event=cancel_event, timeout_seconds=1.0,
    )
