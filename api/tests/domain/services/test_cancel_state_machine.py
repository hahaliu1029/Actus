"""C3 PR-4 — CancelPolicy state machine + auto-escalate tick (spec §8).

Covers:
  - TERMINATE policy drives stop_session + destroy(FORCE_TERMINATE) in side_effect
  - REQUEST_CANCEL forwards + records cancel_state without destroy
  - Auto-escalate tick: REQUEST_CANCEL > CHILD_CANCEL_ACK_TIMEOUT_MS →
    synthetic CANCEL_REQUEST(TERMINATE) published

Sibling-fixture file uses tests/domain/services/conftest.py for ``audit_repo``,
``stub_lifecycle``, ``stub_agent_callback`` and tests/conftest.py for ``fake_redis``.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from app.application.services.mailbox_supervisor import (
    CancelRequestHandler,
    MailboxSupervisor,
    SupervisorContext,
    _CancelState,
)
from app.domain.models.mailbox_envelope import (
    CHILD_CANCEL_ACK_TIMEOUT_MS,
    CancelPolicy,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.infrastructure.external.mailbox.redis_mailbox_publisher import (
    RedisMailboxPublisher,
)


class _StubTelemetry:
    def __init__(self) -> None:
        self.emitted: list[tuple[str, dict]] = []

    async def emit(self, name: str, data: dict) -> None:
        self.emitted.append((name, data))


def _cancel_env(
    *,
    policy: CancelPolicy,
    eid: str = "01HSPYU0CANCEL000000000000001",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    reason: str = "user_request",
) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id=eid,
        type=MailboxEnvelopeType.CANCEL_REQUEST,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id=eid + "C",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.PARENT_AGENT,
        payload={"reason": reason, "policy": policy.value},
    )


@pytest.fixture
async def cancel_ctx(fake_redis, audit_repo, stub_lifecycle, stub_agent_callback):
    return SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-a",
        instance_id="i1",
        redis=fake_redis,
        audit_repo=audit_repo,
        publisher=RedisMailboxPublisher(fake_redis),
        sandbox_lifecycle=stub_lifecycle,
        agent_service_callback=stub_agent_callback,
        telemetry=_StubTelemetry(),
    )


@pytest.mark.anyio
async def test_terminate_policy_calls_stop_then_destroy(
    cancel_ctx, stub_lifecycle, stub_agent_callback, fake_redis
):
    """Spec §8.2 / §7.6 — TERMINATE ordering hard rule:
    ``callback (stop_session) → destroy → synthetic CANCEL_ACK publish``.

    codex r7 [R7-3, HIGH TEST] — earlier this test asserted the *presence*
    of destroy + callback but not their *order*. A regression that swapped
    ``destroy → callback`` (violating spec §7.6 invariant: callback failure
    must NOT prevent destroy, which implies callback runs first) would
    have passed. Now we record a shared event_log via wrapped spies and
    assert the exact canonical sequence."""
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    assert sup is not None  # keeps lint happy; supervisor wires ctx hook

    event_log: list[str] = []

    # Wrap callback + lifecycle so each invocation appends to the shared
    # event_log in the order they fire inside _terminate_outcome's
    # side_effect. The publish step is recorded by observing the synthetic
    # CANCEL_ACK landing in fake_redis (a third, after-destroy event).
    original_callback = cancel_ctx.agent_service_callback
    original_destroy = cancel_ctx.sandbox_lifecycle.destroy
    original_publish = cancel_ctx.publisher.publish

    async def _logged_callback(envelope):
        event_log.append("callback")
        await original_callback(envelope)

    async def _logged_destroy(session_id, reason):
        event_log.append("destroy")
        await original_destroy(session_id, reason)

    async def _logged_publish(envelope):
        # Only record the synthetic CANCEL_ACK echo so we don't pollute
        # the log with any incidental publishes (currently none in this
        # path, but defensive).
        if envelope.type == MailboxEnvelopeType.CANCEL_ACK:
            event_log.append("publish_ack")
        await original_publish(envelope)

    cancel_ctx.agent_service_callback = _logged_callback
    cancel_ctx.sandbox_lifecycle.destroy = _logged_destroy
    cancel_ctx.publisher.publish = _logged_publish

    handler = CancelRequestHandler()
    env = _cancel_env(policy=CancelPolicy.TERMINATE)
    outcome = await handler.handle(env, cancel_ctx)
    await outcome.side_effect()

    # destroy called with FORCE_TERMINATE.
    assert stub_lifecycle.destroy_calls
    _, reason = stub_lifecycle.destroy_calls[0]
    assert reason.value == "force_terminate"

    # Callback dispatched first (stop_session route).
    assert any(
        e.envelope_id == env.envelope_id for e in stub_agent_callback.received
    )

    # R7-3 — exact 3-step ordering per spec §7.6.
    assert event_log == ["callback", "destroy", "publish_ack"], (
        f"TERMINATE side_effect must run callback → destroy → publish_ack "
        f"in that exact order (spec §7.6 hard rule); got {event_log!r}"
    )


@pytest.mark.anyio
async def test_request_cancel_policy_records_state_without_destroy(
    cancel_ctx, stub_lifecycle, stub_agent_callback
):
    """Spec §8.3 — REQUEST_CANCEL forwards to child + records cancel_state.
    No destroy yet (child still has CHILD_CANCEL_ACK_TIMEOUT_MS to ACK)."""
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    handler = CancelRequestHandler()
    env = _cancel_env(policy=CancelPolicy.REQUEST_CANCEL)
    outcome = await handler.handle(env, cancel_ctx)
    await outcome.side_effect()

    assert stub_lifecycle.destroy_calls == []
    assert any(
        e.envelope_id == env.envelope_id for e in stub_agent_callback.received
    )

    state = sup._cancel_states.get("child-1")  # noqa: SLF001
    assert state is not None
    assert state.policy == CancelPolicy.REQUEST_CANCEL


@pytest.mark.anyio
async def test_request_cancel_auto_escalates_after_timeout(cancel_ctx, fake_redis):
    """Spec §8.4 — REQUEST_CANCEL > CHILD_CANCEL_ACK_TIMEOUT_MS without ACK →
    supervisor publishes synthetic CANCEL_REQUEST(policy=TERMINATE).
    """
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001 — fire every tick

    # Register a REQUEST_CANCEL state with old requested_at_mono so it's
    # already past the ACK timeout when the tick runs.
    elapsed_seconds = (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0) + 0.5
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.REQUEST_CANCEL,
        requested_at_mono=cancel_ctx.clock() - elapsed_seconds,
    )

    await sup._maybe_tick_cancel_check()  # noqa: SLF001

    emitted = [n for n, _ in cancel_ctx.telemetry.emitted]
    assert "mailbox.cascade_auto_escalate_terminate" in emitted

    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    cancel_entries = [
        e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
    ]
    assert len(cancel_entries) == 1
    cascaded = json.loads(cancel_entries[0][1][b"envelope"])
    assert cascaded["payload"]["policy"] == "TERMINATE"
    assert cascaded["payload"]["reason"] == "cancel_ack_timeout"
    assert cascaded["producer_role"] == "supervisor"
    # State cleared after escalation.
    assert "child-1" not in sup._cancel_states  # noqa: SLF001


@pytest.mark.anyio
async def test_request_cancel_within_timeout_not_escalated(cancel_ctx, fake_redis):
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001

    # Fresh request — still within timeout window.
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.REQUEST_CANCEL,
        requested_at_mono=cancel_ctx.clock(),
    )
    await sup._maybe_tick_cancel_check()  # noqa: SLF001

    emitted = [n for n, _ in cancel_ctx.telemetry.emitted]
    assert "mailbox.cascade_auto_escalate_terminate" not in emitted
    # No published synthetic envelope.
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    cancel_entries = [
        e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
    ]
    assert cancel_entries == []
    # State retained.
    assert "child-1" in sup._cancel_states  # noqa: SLF001


@pytest.mark.anyio
async def test_terminate_state_not_re_escalated(cancel_ctx, fake_redis):
    """TERMINATE states must NOT be picked up by the auto-escalate tick —
    they don't need to be promoted; the destroy already drove the kill path."""
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.TERMINATE,
        requested_at_mono=cancel_ctx.clock()
        - (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0 + 1.0),
    )
    await sup._maybe_tick_cancel_check()  # noqa: SLF001
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    cancel_entries = [
        e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
    ]
    assert cancel_entries == []


@pytest.mark.anyio
async def test_auto_escalate_disabled_via_flag(cancel_ctx, monkeypatch, fake_redis):
    """Spec §8.4 — ``CANCEL_AUTO_ESCALATE_TO_TERMINATE=False`` disables tick."""
    from app.application.services import mailbox_supervisor as ms

    monkeypatch.setattr(ms, "CANCEL_AUTO_ESCALATE_TO_TERMINATE", False)
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.REQUEST_CANCEL,
        requested_at_mono=cancel_ctx.clock()
        - (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0 + 1.0),
    )
    await sup._maybe_tick_cancel_check()  # noqa: SLF001
    # State retained, nothing published, no telemetry emitted.
    assert "child-1" in sup._cancel_states  # noqa: SLF001
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    assert entries == []
    emitted = [n for n, _ in cancel_ctx.telemetry.emitted]
    assert "mailbox.cascade_auto_escalate_terminate" not in emitted


@pytest.mark.anyio
async def test_cancel_check_interval_throttle(cancel_ctx, fake_redis):
    """Spec §8.4 — auto-escalate tick must throttle to ``_CANCEL_CHECK_INTERVAL_S``
    so we don't spin-burn between iterations of the main loop."""
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    # Force a long interval and pre-stamp the throttle anchor to a recent
    # timestamp, then call the tick — it must short-circuit.
    sup._CANCEL_CHECK_INTERVAL_S = 10.0  # noqa: SLF001
    sup._last_cancel_check_mono = cancel_ctx.clock()  # noqa: SLF001
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.REQUEST_CANCEL,
        requested_at_mono=cancel_ctx.clock()
        - (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0 + 1.0),
    )
    await sup._maybe_tick_cancel_check()  # noqa: SLF001
    # Throttled — no escalation despite the stale state.
    entries = await fake_redis.xrange("actus:child:root-1:mailbox")
    cancel_entries = [
        e for e in entries if e[1].get(b"type") == b"CANCEL_REQUEST"
    ]
    assert cancel_entries == []
    assert "child-1" in sup._cancel_states  # noqa: SLF001


@pytest.mark.anyio
async def test_run_loop_drives_cancel_tick(cancel_ctx, fake_redis):
    """Integration-style — once the supervisor loop is running, the cancel
    tick fires automatically on the same cadence as XAUTOCLAIM."""
    sup = MailboxSupervisor(cancel_ctx, block_ms=0, idle_poll_sleep_s=0.01)
    sup._CANCEL_CHECK_INTERVAL_S = 0.0  # noqa: SLF001
    # Pre-populate stale state.
    sup._cancel_states["child-1"] = _CancelState(  # noqa: SLF001
        child_session_id="child-1",
        policy=CancelPolicy.REQUEST_CANCEL,
        requested_at_mono=cancel_ctx.clock()
        - (CHILD_CANCEL_ACK_TIMEOUT_MS / 1000.0 + 1.0),
    )
    task = asyncio.create_task(sup.run())
    await asyncio.sleep(0.3)
    await sup.stop()
    await task
    emitted = [n for n, _ in cancel_ctx.telemetry.emitted]
    assert "mailbox.cascade_auto_escalate_terminate" in emitted
