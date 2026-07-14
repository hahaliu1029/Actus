"""C2 PR-3 §11.1/§11.5 — CoordinatorRunOrchestrator skeleton tests.

Verifies the parent-cancel path:
- positive deadline publishes CANCEL_REQUEST for pending work units
- cancel_event set → CANCEL_REQUEST published × pending work_units
- missing child_session_id skipped
- per-envelope publish failure logged + swallowed (rest still fire)
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock

from app.application.services.coordinator_envelope_factory import (
    CoordinatorEnvelopeFactory,
)
from app.application.services.coordinator_run_orchestrator import (
    CoordinatorRunOrchestrator,
)
from app.domain.models.mailbox_envelope import MailboxEnvelopeType


@pytest.mark.anyio
async def test_positive_deadline_cancels_pending() -> None:
    publisher = AsyncMock()
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()  # never set
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1"], child_session_ids={"wu1": "c1"},
        cancel_event=ce, timeout_seconds=0.05,
    )
    publisher.publish.assert_awaited_once()
    envelope = publisher.publish.await_args.args[0]
    assert envelope.child_session_id == "c1"
    assert envelope.payload["reason"] == "run_total_wallclock_budget_exceeded"


@pytest.mark.anyio
async def test_cancel_event_set_publishes_cancel_for_all_pending() -> None:
    publisher = AsyncMock()
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()

    async def cancel_after_short_delay() -> None:
        await asyncio.sleep(0.02)
        ce.set()

    asyncio.create_task(cancel_after_short_delay())
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu2"],
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        cancel_event=ce, timeout_seconds=1.0,
    )
    assert publisher.publish.await_count == 2
    for call in publisher.publish.await_args_list:
        env = call.args[0]
        assert env.type == MailboxEnvelopeType.CANCEL_REQUEST
        assert env.payload["reason"] == "parent_cancel"


@pytest.mark.anyio
async def test_cancel_skips_missing_child_session_id() -> None:
    publisher = AsyncMock()
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu_missing"],
        child_session_ids={"wu1": "c1"},
        cancel_event=ce, timeout_seconds=1.0,
    )
    assert publisher.publish.await_count == 1


@pytest.mark.anyio
async def test_publish_failure_swallowed_other_envelopes_still_fire() -> None:
    publisher = AsyncMock()

    async def flaky_publish(env):
        if env.child_session_id == "c1":
            raise RuntimeError("redis down")

    publisher.publish.side_effect = flaky_publish
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu2"],
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        cancel_event=ce, timeout_seconds=1.0,
    )
    assert publisher.publish.await_count == 2


def test_ctor_rejects_empty_parent_session_id() -> None:
    """[r3 P1-2 fix] parent_session_id must be non-empty. Default '' would
    cause publisher to derive stream key ``actus:child::mailbox`` — CANCEL
    envelopes lost."""
    with pytest.raises(ValueError) as ei:
        CoordinatorRunOrchestrator(
            publisher=AsyncMock(),
            envelope_factory=CoordinatorEnvelopeFactory(),
            parent_session_id="", coordinator_run_id="r1",
        )
    assert "parent_session_id" in str(ei.value)


def test_ctor_rejects_empty_coordinator_run_id() -> None:
    """[r3 P1-2 fix] coordinator_run_id must be non-empty."""
    with pytest.raises(ValueError) as ei:
        CoordinatorRunOrchestrator(
            publisher=AsyncMock(),
            envelope_factory=CoordinatorEnvelopeFactory(),
            parent_session_id="p1", coordinator_run_id="",
        )
    assert "coordinator_run_id" in str(ei.value)


@pytest.mark.anyio
async def test_published_envelope_carries_real_parent_session_id() -> None:
    """[r3 P1-2 fix] published CANCEL_REQUEST.parent_session_id matches the
    orchestrator's parent_session_id (not '')."""
    publisher = AsyncMock()
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="real-parent-1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1"], child_session_ids={"wu1": "c1"},
        cancel_event=ce, timeout_seconds=1.0,
    )
    env = publisher.publish.await_args.args[0]
    assert env.parent_session_id == "real-parent-1", (
        "RedisMailboxPublisher derives the stream key from "
        "envelope.parent_session_id; if this is '' the CANCEL would be "
        "published to actus:child::mailbox and never reach the child."
    )


@pytest.mark.anyio
async def test_all_publish_failed_raises_runtime_error() -> None:
    """[r5 P1-2 fix] If ALL CANCEL_REQUEST publishes fail, raise so
    orchestrator_task done-callback surfaces the failure. Previously the loop
    silently swallowed every failure and returned with parent_cancel lost."""
    publisher = AsyncMock()
    publisher.publish.side_effect = RuntimeError("redis fully down")
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    with pytest.raises(RuntimeError) as ei:
        await orch.run(
            coordinator_run_id="r1", root_session_id="root1",
            work_units_pending=["wu1", "wu2"],
            child_session_ids={"wu1": "c1", "wu2": "c2"},
            cancel_event=ce, timeout_seconds=1.0,
        )
    assert "all 2 CANCEL_REQUEST publishes failed" in str(ei.value)
    assert "redis fully down" in str(ei.value)


@pytest.mark.anyio
async def test_partial_failure_does_not_raise() -> None:
    """[r5 P1-2] If at least one publish succeeds, no raise (partial success ok)."""
    publisher = AsyncMock()

    async def flaky(env):
        if env.child_session_id == "c1":
            raise RuntimeError("c1 down")
        # c2 succeeds.

    publisher.publish.side_effect = flaky
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    # Must NOT raise — at least c2 succeeded.
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=["wu1", "wu2"],
        child_session_ids={"wu1": "c1", "wu2": "c2"},
        cancel_event=ce, timeout_seconds=1.0,
    )


@pytest.mark.anyio
async def test_zero_pending_no_raise_no_publish() -> None:
    """[r5 P1-2] If attempted==0 (e.g. empty pending list), no raise (nothing to publish)."""
    publisher = AsyncMock()
    orch = CoordinatorRunOrchestrator(
        publisher=publisher,
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p1", coordinator_run_id="r1",
    )
    ce = asyncio.Event()
    ce.set()
    await orch.run(
        coordinator_run_id="r1", root_session_id="root1",
        work_units_pending=[], child_session_ids={},
        cancel_event=ce, timeout_seconds=1.0,
    )
    publisher.publish.assert_not_called()


@pytest.mark.anyio
async def test_shutdown_is_noop() -> None:
    orch = CoordinatorRunOrchestrator(
        publisher=AsyncMock(),
        envelope_factory=CoordinatorEnvelopeFactory(),
        parent_session_id="p", coordinator_run_id="r",
    )
    await orch.shutdown()
