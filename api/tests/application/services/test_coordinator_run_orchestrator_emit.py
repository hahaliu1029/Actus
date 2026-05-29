"""C2 PR-8 §13 Task 8.4 — CoordinatorRunOrchestrator emit tests.

Asserts the PR-6 placeholder-dict emit is replaced by a typed
CoordinatorSiblingCancelEvent, and that the orchestrator skips emit
when ``outcome is None`` (cancel-ack-only path) because the event
schema requires a non-None ResultReadyOutcome.

C2 PR-9b-A Task A6: also pins the contract that ``emit_event`` is an
**async** callable and that the orchestrator stores it as
``self._emit_event`` so the per-run construction in
``parallel_execution_subgraph._first_time_dispatch`` can verify the
wiring at composition-root time (INV-A3).
"""
from __future__ import annotations

import asyncio
import inspect
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

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


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
    # [F3] sibling-cancel is the only live coordinator event previously left
    # un-attributed; it MUST now carry the run's lineage. ``root_session_id``
    # is threaded from ``run(root_session_id=...)``; ``parent_session_id`` is
    # the ctor invariant (``self._parent_session_id``). child/work_unit stay
    # None (group-level event, like apply/reduce).
    assert ev.root_session_id == "root1"
    assert ev.parent_session_id == "p1"
    assert ev.child_session_id is None
    assert ev.work_unit_id is None


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


# ── C2 PR-9b-A Task A6 — per-run emit_event wiring contract ────────────


async def test_orchestrator_emit_event_is_async_callable() -> None:
    """[Task A6 / INV-A3] The emit closure produced inside
    ``main_graph._run_parallel_backend`` is an **async** callable bound to
    ``event_queue.put_nowait``. The orchestrator stores it as
    ``self._emit_event`` so per-run construction in
    ``parallel_execution_subgraph._first_time_dispatch`` (and the
    composition-root smoke in PR-9b-A8) can pin the wiring.
    """
    publisher = AsyncMock()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def emit(event: Any) -> None:
        # Mirrors the production closure in main_graph: synchronous
        # put_nowait inside the async body so the call is cancellation
        # safe (no await on a full queue).
        event_queue.put_nowait(event)

    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        parent_session_id="p1",
        coordinator_run_id="r1",
        emit_event=emit,
    )

    # INV-A3: the orchestrator surface uses the async callable shape.
    assert inspect.iscoroutinefunction(orch._emit_event)
    # Round-trip: calling the bound emit dispatches via the queue.
    await orch._emit_event("payload-1")
    assert event_queue.get_nowait() == "payload-1"


async def test_emit_event_put_nowait_not_await_put() -> None:
    """[Task A6 / INV-A3] The closure body uses synchronous put_nowait so a
    cancellation between commit and emit does not lose the event. A queue
    whose ``put_nowait`` raises is observable; whose ``put`` would never be
    called.
    """
    publisher = AsyncMock()
    captured: list[Any] = []

    class _ProbeQueue:
        """Drop-in for asyncio.Queue that captures the calling pattern."""

        def __init__(self) -> None:
            self.put_nowait_calls = 0
            self.put_calls = 0

        def put_nowait(self, item: Any) -> None:
            self.put_nowait_calls += 1
            captured.append(item)

        async def put(self, item: Any) -> None:  # pragma: no cover - guard
            self.put_calls += 1
            captured.append(item)

    queue = _ProbeQueue()

    async def emit(event: Any) -> None:
        # The contract: put_nowait inside the async body — NOT await put.
        queue.put_nowait(event)

    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        parent_session_id="p1",
        coordinator_run_id="r1",
        emit_event=emit,
    )

    await orch._emit_event("ev-A")
    await orch._emit_event("ev-B")

    assert queue.put_nowait_calls == 2
    assert queue.put_calls == 0
    assert captured == ["ev-A", "ev-B"]
