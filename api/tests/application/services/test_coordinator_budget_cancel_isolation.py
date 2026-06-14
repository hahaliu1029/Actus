"""[C2b budget §5-12/§5-13] D9 event-split end-to-end semantics — two-child
assemblies, no Redis.

INV-B7: a child's budget trip sets ONLY that child's event; the run-level
event and every sibling event stay unset; sibling cancellation happens ONLY
through the orchestrator's RESULT_READY policy → CANCEL_REQUEST envelopes
(correct PARENT_CANCEL label), never raw-event contagion.

INV-B8: after the split, user/parent cancel still reaches every child:
run event → orchestrator _cancel_watcher → CANCEL_REQUEST × pending → each
child's listener → request_stop(PARENT_CANCEL) → child-local event.

The in-memory _GroupBacklogSubscriber models the ONE Redis-stream property
the R4#1 group pre-creation relies on: a consumer group created at index T
receives every message appended AFTER T even if consume starts later, and
subscribe is BUSYGROUP-idempotent (keeps the EARLIEST creation index).
"""
from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_child_cancel_listener import (
    CoordinatorChildCancelListener,
)
from app.application.services.coordinator_child_runner import (
    CoordinatorChildRunner,
    StopReason,
)
from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.application.services.coordinator_run_orchestrator import (
    CoordinatorRunOrchestrator,
    should_trigger_sibling_cancel,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import (
    NeedsAuthorizationDetails,
)


pytestmark = pytest.mark.anyio


# ── In-memory group-semantics subscriber ─────────────────────────────────────


class _GroupBacklogSubscriber:
    """Minimal mailbox-subscriber fake with Redis consumer-group retention
    semantics (see module docstring)."""

    def __init__(self) -> None:
        self._messages: list[dict] = []
        self._groups: dict[str, int] = {}

    async def subscribe(self, *, stream_key: str, consumer_group: str,
                        consumer_name: str, start_id: str = "$") -> None:
        # BUSYGROUP-idempotent: keep the EARLIEST creation index.
        self._groups.setdefault(consumer_group, len(self._messages))

    async def publish(self, envelope_dict: dict) -> None:
        self._messages.append(envelope_dict)

    async def consume(self, *, stream_key: str, consumer_group: str,
                      consumer_name: str, predicate: Any,
                      max_iterations: int | None = None):
        start = self._groups[consumer_group]
        for env in self._messages[start:]:
            if await predicate(env):
                yield env

    async def destroy_group(self, *, stream_key: str, consumer_group: str) -> None:
        self._groups.pop(consumer_group, None)


def _budget_needs_auth_envelope(child_session_id: str = "cA"):
    """A budget-exhausted RESULT_READY — the §11.8 case-4 shape."""
    return CoordinatorEnvelopeFactory().make_result_ready(
        parent_session_id="p1",
        child_session_id=child_session_id,
        correlation_id="r1",
        payload=ResultReadyPayload(
            summary="budget exhausted",
            outcome=ResultReadyOutcome.NEEDS_AUTHORIZATION,
            needs_authorization_details=NeedsAuthorizationDetails(
                reason="budget_exhausted",
                observed_evidence="stop_reason=token_budget",
            ),
        ),
    )


# ── spec §5-12 — INV-B7: no raw-event contagion ─────────────────────────────


async def test_budget_trip_no_raw_event_contagion() -> None:
    """Child A trips its token budget:
      (i)   A's event set; B's event AND the run-level event stay unset —
            the orchestrator's _cancel_watcher (which awaits the run event)
            cannot be woken by a budget trip;
      (ii)  A's RESULT_READY(NEEDS_AUTHORIZATION, budget_exhausted) IS
            sibling-cancel policy case-4 (fan-out is the ORCHESTRATOR's
            decision on the envelope channel — by-design behavior, the
            point is the correct PARENT_CANCEL label, not raw contagion);
      (iii) B is then cancelled VIA its listener with PARENT_CANCEL —
            never with stop_reason=None raw contagion."""
    run_event = asyncio.Event()
    event_a, event_b = asyncio.Event(), asyncio.Event()
    runner_a = CoordinatorChildRunner(cancel_event=event_a)
    runner_b = CoordinatorChildRunner(cancel_event=event_b)

    # (i) Budget trip on A only.
    runner_a.request_stop(StopReason.TOKEN_BUDGET)
    assert event_a.is_set()
    assert not event_b.is_set(), "INV-B7: sibling event must stay unset"
    assert not run_event.is_set(), "INV-B7: run-level event must stay unset"

    # (ii) Policy half: the budget envelope IS a fan-out trigger (case-4).
    env = _budget_needs_auth_envelope("cA")
    assert should_trigger_sibling_cancel(env) is True

    # (iii) The fan-out arrives at B as a CANCEL_REQUEST envelope → B's
    # listener → request_stop(PARENT_CANCEL) — the CORRECT label.
    sub = _GroupBacklogSubscriber()
    listener_b = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="cB",
        runner=runner_b,
    )
    await listener_b.start()
    cancel_req = CoordinatorEnvelopeFactory().make_cancel_request(
        parent_session_id="p1", child_session_id="cB",
        correlation_id="r1", reason="parent_cancel",
    )
    await sub.publish(cancel_req.model_dump(mode="json"))
    await listener_b._listen_loop_one_iteration()
    await listener_b.shutdown()

    assert runner_b._stop_reason == StopReason.PARENT_CANCEL, (
        "sibling must be cancelled with the PARENT_CANCEL label via the "
        "envelope channel — not stop_reason=None raw contagion"
    )
    assert event_b.is_set()
    # A's first-wins reason untouched by the whole sequence.
    assert runner_a._stop_reason == StopReason.TOKEN_BUDGET


# ── spec §5-13 — INV-B8: parent cancel still fans out after the split ────────


async def test_parent_cancel_fans_out_after_event_split(monkeypatch) -> None:
    """run event set → REAL orchestrator _cancel_watcher publishes
    CANCEL_REQUEST × pending → each child listener stops its runner with
    PARENT_CANCEL → child-local events set → (for one child) run_work_unit
    short-circuits into CANCEL_ACK."""
    run_event = asyncio.Event()
    sub = _GroupBacklogSubscriber()

    published: list[dict] = []

    class _Publisher:
        async def publish(self, envelope) -> None:
            published.append(envelope.model_dump(mode="json"))
            await sub.publish(published[-1])

    orchestrator = CoordinatorRunOrchestrator(
        publisher=_Publisher(),
        parent_session_id="p1",
        coordinator_run_id="r1",
    )

    event_1, event_2 = asyncio.Event(), asyncio.Event()
    runner_1 = CoordinatorChildRunner(cancel_event=event_1)
    runner_2 = CoordinatorChildRunner(cancel_event=event_2)
    listeners = [
        CoordinatorChildCancelListener(
            subscriber=sub, root_session_id="root1",
            child_session_id=sid, runner=rn,
        )
        for sid, rn in (("c1", runner_1), ("c2", runner_2))
    ]
    for ln in listeners:
        await ln.start()

    run_event.set()  # user/parent cancel
    await orchestrator._cancel_watcher(
        cancel_event=run_event,
        pending={"wu1", "wu2"},
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        coordinator_run_id="r1",
    )
    assert len(published) == 2
    for ln in listeners:
        await ln._listen_loop_one_iteration()
        await ln.shutdown()

    assert runner_1._stop_reason == StopReason.PARENT_CANCEL
    assert runner_2._stop_reason == StopReason.PARENT_CANCEL
    assert event_1.is_set() and event_2.is_set()

    # Tail: a stopped child's run_work_unit short-circuits → CANCEL_ACK.
    envf = MagicMock()
    envf.make_cancel_ack = MagicMock(
        return_value=MagicMock(type=MailboxEnvelopeType.CANCEL_ACK),
    )
    envf.make_result_ready = MagicMock()
    publisher = AsyncMock()
    publisher.publish = AsyncMock()
    runner_1._publisher = publisher
    runner_1._envelope_factory = envf
    runner_1._inner_runner = MagicMock()  # has invoke_until_done auto-attr

    from app.domain.models.work_unit import WorkUnit

    listener_stub = MagicMock()
    listener_stub.ready_event = asyncio.Event()
    listener_stub.start = AsyncMock(
        side_effect=lambda: listener_stub.ready_event.set(),
    )
    listener_stub.shutdown = AsyncMock()
    monkeypatch.setattr(
        "app.application.services.coordinator_child_runner."
        "CoordinatorChildCancelListener",
        lambda **_kw: listener_stub,
    )
    await runner_1.run_work_unit(
        coordinator_run_id="r1",
        work_unit=WorkUnit(
            work_unit_id="wu1", objective="x", phase="exploration",
            allowed_tools=["file_read"], write_lease=[],
        ),
        child_session_id="c1", spawn_manifest=MagicMock(),
        cancel_event=event_1, root_session_id="root1",
    )
    envf.make_cancel_ack.assert_called_once()
    envf.make_result_ready.assert_not_called()
    # Pre-invoke short-circuit: the inner runner was never invoked.
    runner_1._inner_runner.invoke_until_done.assert_not_called()
    publisher.publish.assert_awaited_once()


async def test_pre_subscribe_cancel_request_retained_by_precreated_group() -> None:
    """[spec §5-13 R4#1 子用例] CANCEL_REQUEST published BEFORE the listener
    subscribes, but AFTER dispatch pre-created the listener group → the
    backlog envelope is consumed once the listener starts (race closed)."""
    sub = _GroupBacklogSubscriber()
    # Dispatch hoist: pre-create the listener group BEFORE any publish.
    await sub.subscribe(
        stream_key="actus:child:root1:mailbox",
        consumer_group="coordinator:child:c1",
        consumer_name="c1-listener",
    )
    # CANCEL_REQUEST lands while the child is still booting (no listener yet).
    cancel_req = CoordinatorEnvelopeFactory().make_cancel_request(
        parent_session_id="p1", child_session_id="c1",
        correlation_id="r1", reason="parent_cancel",
    )
    await sub.publish(cancel_req.model_dump(mode="json"))

    event_1 = asyncio.Event()
    runner_1 = CoordinatorChildRunner(cancel_event=event_1)
    listener = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=runner_1,
    )
    await listener.start()  # idempotent re-subscribe keeps the earlier index
    await listener._listen_loop_one_iteration()
    await listener.shutdown()

    assert runner_1._stop_reason == StopReason.PARENT_CANCEL
    assert event_1.is_set()


async def test_targeted_cancel_does_not_stop_other_children() -> None:
    """[spec §5-13 R6#6 负断言] A CANCEL_REQUEST addressed to c1 on the SAME
    root stream must NOT stop a c2 runner — pins the listener's
    child_session_id filter predicate."""
    sub = _GroupBacklogSubscriber()
    event_1, event_2 = asyncio.Event(), asyncio.Event()
    runner_1 = CoordinatorChildRunner(cancel_event=event_1)
    runner_2 = CoordinatorChildRunner(cancel_event=event_2)
    l1 = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c1",
        runner=runner_1,
    )
    l2 = CoordinatorChildCancelListener(
        subscriber=sub, root_session_id="root1", child_session_id="c2",
        runner=runner_2,
    )
    await l1.start()
    await l2.start()

    cancel_req = CoordinatorEnvelopeFactory().make_cancel_request(
        parent_session_id="p1", child_session_id="c1",
        correlation_id="r1", reason="parent_cancel",
    )
    await sub.publish(cancel_req.model_dump(mode="json"))
    await l1._listen_loop_one_iteration()
    await l2._listen_loop_one_iteration()
    await l1.shutdown()
    await l2.shutdown()

    assert runner_1._stop_reason == StopReason.PARENT_CANCEL
    assert runner_2._stop_reason is None, "filter predicate must exclude c2"
    assert not event_2.is_set()
