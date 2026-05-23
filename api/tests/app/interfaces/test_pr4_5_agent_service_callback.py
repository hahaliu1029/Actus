"""C3 PR-4.5 — supervisor callback bridge semantics (codex r2 [R2-1]).

The bridge in ``service_dependencies._pr4_5_agent_service_callback`` is
invoked by MailboxSupervisor handlers on several envelope types. Only
``CANCEL_REQUEST(policy=TERMINATE)`` should trigger
``AgentService.stop_session`` — every other type must be a no-op so a
SPAWN_REQUEST / heartbeat / RESULT_READY / CANCEL_ACK doesn't
accidentally cancel the child mid-flight.

These tests directly exercise the bridge function with synthetic
envelopes against a recording AgentService stub.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    CancelPolicy,
    CancelRequestPayload,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ProgressKind,
    ProgressUpdatePayload,
    ProgressVisibility,
    ResultReadyOutcome,
    ResultReadyPayload,
    SpawnRequestPayload,
)
from app.interfaces import service_dependencies


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _RecordingAgentService:
    def __init__(self) -> None:
        self.stop_calls: list[dict[str, Any]] = []

    async def get_session(self, sid: str):
        # codex r8 [R8-2] — callback now validates the row is a
        # mailbox-plane subagent before issuing stop_session, so the
        # stub must return that shape for TERMINATE tests to fire.
        from app.domain.models.session import Session
        return Session(
            id=sid,
            user_id="u1",
            worker_type="subagent",
            subagent_control_plane="mailbox",
            parent_session_id="root-1",
        )

    async def stop_session(self, session_id: str, user_id: str, is_admin: bool = False) -> None:
        self.stop_calls.append(
            {"session_id": session_id, "user_id": user_id, "is_admin": is_admin}
        )


@pytest.fixture
def stub_agent_service():
    """Install a recording AgentService stub via the production bind
    helper so the bind event is set as it would be in lifespan. Tear
    down via the test-only reset helper so subsequent tests start
    clean.
    """
    svc = _RecordingAgentService()
    service_dependencies._bind_agent_service_for_callback(svc)
    yield svc
    service_dependencies._reset_agent_service_callback_state_for_tests()


def _make_envelope(envelope_type: MailboxEnvelopeType, payload_obj) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id=f"env-{envelope_type.value}",
        type=envelope_type,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id="corr-1",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.SUPERVISOR,
        payload=payload_obj.model_dump(mode="json"),
    )


@pytest.mark.anyio
async def test_cancel_request_terminate_triggers_stop_session(
    stub_agent_service,
) -> None:
    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_REQUEST,
        CancelRequestPayload(reason="cascade", policy=CancelPolicy.TERMINATE),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert len(stub_agent_service.stop_calls) == 1
    call = stub_agent_service.stop_calls[0]
    assert call["session_id"] == "child-1"
    assert call["is_admin"] is True


@pytest.mark.anyio
async def test_cancel_request_request_cancel_does_not_stop(
    stub_agent_service,
) -> None:
    """``REQUEST_CANCEL`` (cooperative request, distinct from TERMINATE)
    must NOT call stop_session — the child driver decides whether/how to
    honor the request."""
    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_REQUEST,
        CancelRequestPayload(reason="user", policy=CancelPolicy.REQUEST_CANCEL),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert stub_agent_service.stop_calls == []


@pytest.mark.anyio
async def test_spawn_request_does_not_stop(stub_agent_service) -> None:
    env = _make_envelope(
        MailboxEnvelopeType.SPAWN_REQUEST,
        SpawnRequestPayload(agent_kind="research", task_prompt=""),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert stub_agent_service.stop_calls == []


@pytest.mark.anyio
async def test_progress_update_heartbeat_does_not_stop(
    stub_agent_service,
) -> None:
    env = _make_envelope(
        MailboxEnvelopeType.PROGRESS_UPDATE,
        ProgressUpdatePayload(
            kind=ProgressKind.HEARTBEAT,
            visibility=ProgressVisibility.HIDDEN,
        ),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert stub_agent_service.stop_calls == []


@pytest.mark.anyio
async def test_result_ready_does_not_stop(stub_agent_service) -> None:
    """RESULT_READY callback fires AFTER destroy; supervisor uses it as
    a wake signal only. Calling stop_session would be redundant and
    would corrupt user_cancel terminal_reason audit."""
    env = _make_envelope(
        MailboxEnvelopeType.RESULT_READY,
        ResultReadyPayload(summary="", outcome=ResultReadyOutcome.SUCCESS),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert stub_agent_service.stop_calls == []


@pytest.mark.anyio
async def test_cancel_ack_does_not_stop(stub_agent_service) -> None:
    """CANCEL_ACK callback fires AFTER destroy on the cooperative path —
    the child has already self-terminated."""
    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_ACK,
        CancelAckPayload(final_state="cancelled"),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)
    assert stub_agent_service.stop_calls == []


@pytest.mark.anyio
async def test_cancel_request_terminate_skipped_when_session_not_mailbox(
    monkeypatch,
) -> None:
    """codex r8 [R8-2, HIGH SEC] — even on a valid
    ``CANCEL_REQUEST(TERMINATE)``, the callback MUST refuse to dispatch
    stop_session when the looked-up row is not a mailbox-plane subagent.
    Defense in depth: ``is_admin=True`` bypasses ownership checks, so a
    misaddressed envelope (e.g. a root_session_id leaked into
    child_session_id) could otherwise end the wrong session.
    """
    class _LegacyAgentService(_RecordingAgentService):
        async def get_session(self, sid: str):
            from app.domain.models.session import Session
            return Session(
                id=sid,
                user_id="u1",
                worker_type="subagent",
                subagent_control_plane="legacy",
                parent_session_id="root-1",
            )

    svc = _LegacyAgentService()
    service_dependencies._bind_agent_service_for_callback(svc)
    try:
        env = _make_envelope(
            MailboxEnvelopeType.CANCEL_REQUEST,
            CancelRequestPayload(reason="x", policy=CancelPolicy.TERMINATE),
        )
        await service_dependencies._pr4_5_agent_service_callback(env)
        assert svc.stop_calls == []
    finally:
        service_dependencies._reset_agent_service_callback_state_for_tests()


@pytest.mark.anyio
async def test_callback_returns_when_bind_event_never_set(monkeypatch) -> None:
    """codex r5 [R5-1] — the module-level bind event is always present
    (created at import). When the AgentService never binds within the
    timeout window, the callback returns cleanly rather than blocking
    forever. ``_reset_agent_service_callback_state_for_tests`` clears
    the holder + event so the unbound path is exercisable.
    """
    import asyncio as _asyncio

    service_dependencies._reset_agent_service_callback_state_for_tests()
    monkeypatch.setattr(
        service_dependencies,
        "_PR4_5_BIND_WAIT_TIMEOUT_SECONDS",
        0.05,
    )

    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_REQUEST,
        CancelRequestPayload(reason="x", policy=CancelPolicy.TERMINATE),
    )
    # Must not raise and must return inside the test timeout.
    await _asyncio.wait_for(
        service_dependencies._pr4_5_agent_service_callback(env),
        timeout=1.0,
    )


@pytest.mark.anyio
async def test_callback_waits_for_bind_then_stops(monkeypatch) -> None:
    """codex r3 [R3-1, HIGH ARCH] — TERMINATE envelope received before
    bind blocks on the module-level bind event briefly. When bind
    completes inside the wait window the callback DOES dispatch
    stop_session. Uses ``_reset_agent_service_callback_state_for_tests``
    to clear the module-level holder + clear the event, then
    ``_bind_agent_service_for_callback`` to set them both at once
    (matches production lifespan wiring).
    """
    import asyncio as _asyncio

    service_dependencies._reset_agent_service_callback_state_for_tests()
    monkeypatch.setattr(
        service_dependencies,
        "_PR4_5_BIND_WAIT_TIMEOUT_SECONDS",
        2.0,
    )

    recording = _RecordingAgentService()

    async def _bind_after_delay() -> None:
        await _asyncio.sleep(0.05)
        service_dependencies._bind_agent_service_for_callback(recording)

    _asyncio.create_task(_bind_after_delay())

    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_REQUEST,
        CancelRequestPayload(reason="cascade", policy=CancelPolicy.TERMINATE),
    )
    await service_dependencies._pr4_5_agent_service_callback(env)

    assert len(recording.stop_calls) == 1
    assert recording.stop_calls[0]["session_id"] == "child-1"


@pytest.mark.anyio
async def test_callback_times_out_when_bind_never_happens(monkeypatch) -> None:
    """codex r3 [R3-1] — when bind never completes, callback gives up
    after the configured timeout and returns. Destroy still proceeds
    (handler swallows callback exceptions) but the error is auditable.
    """
    import asyncio as _asyncio

    service_dependencies._reset_agent_service_callback_state_for_tests()
    monkeypatch.setattr(
        service_dependencies,
        "_PR4_5_BIND_WAIT_TIMEOUT_SECONDS",
        0.05,
    )

    env = _make_envelope(
        MailboxEnvelopeType.CANCEL_REQUEST,
        CancelRequestPayload(reason="x", policy=CancelPolicy.TERMINATE),
    )
    # Must not raise; must not block forever.
    await _asyncio.wait_for(
        service_dependencies._pr4_5_agent_service_callback(env), timeout=1.0
    )
