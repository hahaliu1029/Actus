"""C3 PR-4.5 — ChildHeartbeatTask (spec §9.1 + M5 invariant).

Tests the 15s independent heartbeat loop emitted by mailbox-plane child
agents while their main loop is busy. Uses tiny intervals (~50ms) so the
suite runs in well under a second.
"""

from __future__ import annotations

import asyncio

import pytest

from app.domain.models.mailbox_envelope import (
    MailboxEnvelopeType,
    ProducerRole,
    ProgressKind,
    ProgressVisibility,
)
from app.domain.services.child_heartbeat_task import ChildHeartbeatTask


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _CapturingPublisher:
    """Fake MailboxPublisher capturing all published envelopes."""

    def __init__(self) -> None:
        self.published: list = []

    async def publish(self, envelope) -> None:
        self.published.append(envelope)


@pytest.mark.anyio
async def test_heartbeat_emits_progress_update_at_interval() -> None:
    pub = _CapturingPublisher()
    hb = ChildHeartbeatTask(
        publisher=pub,
        parent_session_id="root-1",
        child_session_id="child-1",
        interval_seconds=0.05,
    )
    task = asyncio.create_task(hb.run())
    # Wait for at least 3 heartbeats: t=0 (immediate), t=0.05, t=0.10
    await asyncio.sleep(0.18)
    await hb.stop()
    await task

    assert len(pub.published) >= 3, (
        f"expected ≥3 heartbeats in ~180ms at 50ms interval, got "
        f"{len(pub.published)}"
    )
    for env in pub.published:
        assert env.type == MailboxEnvelopeType.PROGRESS_UPDATE
        assert env.producer_role == ProducerRole.CHILD_AGENT
        assert env.payload["kind"] == ProgressKind.HEARTBEAT.value
        assert env.payload["visibility"] == ProgressVisibility.HIDDEN.value
        assert env.parent_session_id == "root-1"
        assert env.child_session_id == "child-1"
        assert env.correlation_id == "hb:child-1"


@pytest.mark.anyio
async def test_heartbeat_stops_promptly_on_stop() -> None:
    """``stop()`` must take effect well inside ``interval_seconds`` — the
    wait_for(stop_event) pattern is the contract.

    Set interval to 10s, run for 50ms, stop, and assert task completes
    within 500ms — clearly faster than the interval.
    """
    pub = _CapturingPublisher()
    hb = ChildHeartbeatTask(
        pub, "root-1", "child-1", interval_seconds=10.0,
    )
    task = asyncio.create_task(hb.run())
    await asyncio.sleep(0.05)
    await hb.stop()
    # If stop didn't interrupt the wait_for, this would take 10s and raise
    # TimeoutError. 500ms is generous given asyncio scheduling jitter.
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.anyio
async def test_heartbeat_phase_and_tool_call_id_reflected() -> None:
    pub = _CapturingPublisher()
    hb = ChildHeartbeatTask(
        pub, "root-1", "child-1", interval_seconds=0.05,
    )
    task = asyncio.create_task(hb.run())
    # Let the initial t=0 heartbeat fire with default phase, then update.
    await asyncio.sleep(0.02)
    hb.set_phase("in_tool", tool_call_id="tc-abc")
    await asyncio.sleep(0.12)
    await hb.stop()
    await task

    # The last published heartbeat must reflect the updated phase.
    last = pub.published[-1]
    assert last.payload["phase"] == "in_tool"
    assert last.payload["tool_call_id"] == "tc-abc"


@pytest.mark.anyio
async def test_stop_is_idempotent() -> None:
    """Repeated ``stop()`` calls must not raise."""
    pub = _CapturingPublisher()
    hb = ChildHeartbeatTask(pub, "r", "c", interval_seconds=0.05)
    task = asyncio.create_task(hb.run())
    await asyncio.sleep(0.02)
    await hb.stop()
    await hb.stop()
    await hb.stop()
    await asyncio.wait_for(task, timeout=0.5)


@pytest.mark.anyio
async def test_transient_publish_failure_does_not_kill_loop() -> None:
    """codex r1 [R1-5, HIGH PERF] regression — a single publisher exception
    (e.g. Redis blip) MUST NOT terminate the heartbeat loop. Otherwise
    the child looks orphaned to the supervisor and gets force-terminated
    while still healthy.
    """

    class _FlakyPublisher:
        def __init__(self) -> None:
            self.calls = 0
            self.published = []

        async def publish(self, envelope) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient redis blip")
            self.published.append(envelope)

    pub = _FlakyPublisher()
    hb = ChildHeartbeatTask(pub, "r", "c", interval_seconds=0.05)
    task = asyncio.create_task(hb.run())
    await asyncio.sleep(0.18)
    await hb.stop()
    await task

    # First call raised; subsequent calls succeeded — loop must have
    # survived the first failure to produce additional published
    # envelopes.
    assert pub.calls >= 3, (
        f"expected ≥3 publish attempts despite first one raising; got {pub.calls}"
    )
    assert len(pub.published) >= 2, (
        f"expected ≥2 successful heartbeats after the transient failure; "
        f"got {len(pub.published)}"
    )


@pytest.mark.anyio
async def test_cancelled_error_propagates() -> None:
    """``asyncio.CancelledError`` is the ONE exception that must
    propagate so the asyncio task can be cancelled cleanly by the
    parent runner during terminal cleanup.
    """

    class _CancellingPublisher:
        async def publish(self, envelope) -> None:
            raise asyncio.CancelledError()

    pub = _CancellingPublisher()
    hb = ChildHeartbeatTask(pub, "r", "c", interval_seconds=0.05)
    task = asyncio.create_task(hb.run())
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.5)
