"""Task 4: mailbox terminal handlers own the child session DB terminal CAS.

The coordinator child runner publishes the terminal envelope but deliberately
does not write the session row when ``external_terminal_owner=True``.  These
tests pin the supervisor's authoritative side-effect boundary: terminalize,
then sandbox cleanup/tracking cleanup, then audit seal/XACK.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.mailbox_supervisor import (
    CancelAckHandler,
    MailboxSupervisor,
    ResultReadyHandler,
    SupervisorContext,
)
from app.application.services.coordinator_terminal_transition import (
    CoordinatorTerminalCommand,
)
from app.domain.errors.sandbox_lifecycle import SandboxLifecycleError
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import SessionStatus


pytestmark = pytest.mark.anyio


class _AuditRepo:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def get_processed(self, parent: str, envelope_id: str) -> bool:
        return bool(
            self.rows.get((parent, envelope_id), {}).get("processed_at")
        )

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None:
        self.rows.setdefault(
            (envelope.parent_session_id, envelope.envelope_id), {}
        )["processing_at"] = processing_at

    async def mark_processed(
        self,
        parent: str,
        envelope_id: str,
        *,
        processed_at: datetime,
    ) -> None:
        self.rows.setdefault((parent, envelope_id), {})[
            "processed_at"
        ] = processed_at


class _Lifecycle:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[str, object]] = []

    async def destroy(self, child_id: str, reason: object) -> None:
        self.calls.append((child_id, reason))
        if self.failures:
            self.failures -= 1
            raise SandboxLifecycleError("docker unavailable")


class _Consumer:
    def __init__(self) -> None:
        self.acked: list[bytes] = []

    async def ack(self, redis_id: bytes) -> None:
        self.acked.append(redis_id)


class _Terminalizer:
    """CAS-faithful fake: repeated calls update the row exactly once."""

    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[CoordinatorTerminalCommand] = []
        self.transition_count = 0
        self.is_terminal = False

    async def __call__(self, command: CoordinatorTerminalCommand) -> bool:
        self.calls.append(command)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("terminal DB write failed")
        if self.is_terminal:
            return False
        self.is_terminal = True
        self.transition_count += 1
        return True


def _result_payload(outcome: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"summary": outcome, "outcome": outcome}
    if outcome == "needs_authorization":
        payload["needs_authorization_details"] = {"reason": "hard_blocked"}
    return payload


def _envelope(
    envelope_type: MailboxEnvelopeType,
    payload: dict[str, Any],
    *,
    envelope_id: str = "terminal-1",
    producer_role: ProducerRole = ProducerRole.CHILD_AGENT,
    correlation_id: str = "run-1",
) -> MailboxEnvelope:
    return MailboxEnvelope(
        envelope_id=envelope_id,
        type=envelope_type,
        parent_session_id="root-1",
        child_session_id="child-1",
        correlation_id=correlation_id,
        emitted_at=datetime.now(timezone.utc),
        producer_role=producer_role,
        payload=payload,
    )


def _context(
    terminalizer: _Terminalizer,
    *,
    lifecycle: _Lifecycle | None = None,
) -> SupervisorContext:
    ctx = SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-1",
        instance_id="instance-1",
        redis=MagicMock(),
        audit_repo=_AuditRepo(),
        publisher=MagicMock(),
        sandbox_lifecycle=lifecycle or _Lifecycle(),
        agent_service_callback=AsyncMock(),
        telemetry=AsyncMock(),
    )
    ctx.terminalize_child = terminalizer
    ctx.clear_child_tracking = MagicMock()
    return ctx


async def test_live_legacy_terminal_contract_skips_db_even_when_ports_fail() -> None:
    """The live research runner uses spawn:<child> correlation for terminals.

    It owned and wrote its DB terminal state before publishing, so Supervisor
    must preserve the pre-Task4 cleanup/ACK path without consulting the new
    coordinator row reader or terminalizer.
    """
    terminalizer = _Terminalizer(failures=1)
    lifecycle = _Lifecycle()
    ctx = _context(
        terminalizer,
        lifecycle=lifecycle,
    )
    supervisor = MailboxSupervisor(ctx)
    consumer = _Consumer()
    supervisor._consumer = consumer
    supervisor._last_seen_mono["child-1"] = 1.0
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY,
        _result_payload("success"),
        correlation_id="spawn:child-1",
    )

    await supervisor._handle_envelope(b"legacy-1", envelope)

    assert terminalizer.calls == []
    assert lifecycle.calls
    ctx.agent_service_callback.assert_awaited_once_with(envelope)
    assert consumer.acked == [b"legacy-1"]
    assert "child-1" not in supervisor._last_seen_mono


async def test_terminal_handler_builds_typed_lineage_command() -> None:
    terminalizer = _Terminalizer()
    ctx = _context(terminalizer)
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("success")
    )

    handler_outcome = await ResultReadyHandler().handle(envelope, ctx)
    assert handler_outcome.side_effect is not None
    await handler_outcome.side_effect()

    assert len(terminalizer.calls) == 1
    command = terminalizer.calls[0]
    assert command.status == SessionStatus.COMPLETED
    assert command.reason == "natural"
    assert command.lineage.child_session_id == "child-1"
    assert command.lineage.parent_session_id == "root-1"
    assert command.lineage.root_session_id == "root-1"
    assert command.lineage.coordinator_run_id == "run-1"


async def test_authority_refusal_keeps_existing_cleanup() -> None:
    terminalizer = _Terminalizer()
    terminalizer.is_terminal = True
    lifecycle = _Lifecycle()
    ctx = _context(terminalizer, lifecycle=lifecycle)
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("failed")
    )

    handler_outcome = await ResultReadyHandler().handle(envelope, ctx)
    assert handler_outcome.side_effect is not None
    await handler_outcome.side_effect()

    assert len(terminalizer.calls) == 1
    assert lifecycle.calls
    ctx.agent_service_callback.assert_awaited_once_with(envelope)
    ctx.clear_child_tracking.assert_called_once_with("child-1")


async def test_atomic_terminal_port_failure_retains_pel_and_tracking() -> None:
    terminalizer = _Terminalizer(failures=1)
    lifecycle = _Lifecycle()
    ctx = _context(terminalizer, lifecycle=lifecycle)
    supervisor = MailboxSupervisor(ctx)
    consumer = _Consumer()
    supervisor._consumer = consumer
    supervisor._last_seen_mono["child-1"] = 1.0
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("success")
    )

    await supervisor._handle_envelope(b"coordinator-lookup-1", envelope)

    assert len(terminalizer.calls) == 1
    assert lifecycle.calls == []
    ctx.agent_service_callback.assert_not_awaited()
    assert consumer.acked == []
    assert "child-1" in supervisor._last_seen_mono
    assert not await ctx.audit_repo.get_processed("root-1", "terminal-1")


@pytest.mark.parametrize(
    ("outcome", "expected_status", "expected_reason"),
    [
        ("success", SessionStatus.COMPLETED, "natural"),
        ("failed", SessionStatus.COMPLETED, "natural"),
        ("cancelled", SessionStatus.COMPLETED, "natural"),
        ("needs_authorization", SessionStatus.COMPLETED, "natural"),
        ("timed_out", SessionStatus.TIMED_OUT, "watchdog_timeout"),
    ],
)
async def test_result_ready_maps_to_existing_child_row_terminal_semantics(
    outcome: str,
    expected_status: SessionStatus,
    expected_reason: str,
) -> None:
    terminalizer = _Terminalizer()
    ctx = _context(terminalizer)
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload(outcome)
    )

    handler_outcome = await ResultReadyHandler().handle(envelope, ctx)
    assert handler_outcome.side_effect is not None
    await handler_outcome.side_effect()

    assert [
        (call.lineage.child_session_id, call.status, call.reason)
        for call in terminalizer.calls
    ] == [(envelope.child_session_id, expected_status, expected_reason)]


@pytest.mark.parametrize(
    ("final_state", "expected_status", "expected_reason"),
    [
        ("cancelled", SessionStatus.COMPLETED, "natural"),
        ("completed", SessionStatus.COMPLETED, "natural"),
        ("force_terminated", SessionStatus.TIMED_OUT, "watchdog_timeout"),
    ],
)
async def test_cancel_ack_maps_to_existing_child_row_terminal_semantics(
    final_state: str,
    expected_status: SessionStatus,
    expected_reason: str,
) -> None:
    terminalizer = _Terminalizer()
    ctx = _context(terminalizer)
    envelope = _envelope(
        MailboxEnvelopeType.CANCEL_ACK, {"final_state": final_state}
    )

    handler_outcome = await CancelAckHandler().handle(envelope, ctx)
    assert handler_outcome.side_effect is not None
    await handler_outcome.side_effect()

    assert [
        (call.lineage.child_session_id, call.status, call.reason)
        for call in terminalizer.calls
    ] == [(envelope.child_session_id, expected_status, expected_reason)]


async def test_supervisor_echo_cancel_ack_terminalizes_without_repeating_cleanup() -> None:
    terminalizer = _Terminalizer()
    ctx = _context(terminalizer)
    envelope = _envelope(
        MailboxEnvelopeType.CANCEL_ACK,
        {"final_state": "force_terminated"},
        producer_role=ProducerRole.SUPERVISOR_ECHO,
    )

    handler_outcome = await CancelAckHandler().handle(envelope, ctx)
    assert handler_outcome.side_effect is not None
    await handler_outcome.side_effect()

    assert [
        (call.lineage.child_session_id, call.status, call.reason)
        for call in terminalizer.calls
    ] == [("child-1", SessionStatus.TIMED_OUT, "watchdog_timeout")]
    assert ctx.sandbox_lifecycle.calls == []
    ctx.agent_service_callback.assert_not_awaited()


async def test_redelivery_dedups_terminal_transition_at_audit_boundary() -> None:
    terminalizer = _Terminalizer()
    ctx = _context(terminalizer)
    supervisor = MailboxSupervisor(ctx)
    consumer = _Consumer()
    supervisor._consumer = consumer
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("success")
    )

    await supervisor._handle_envelope(b"1-0", envelope)
    await supervisor._handle_envelope(b"1-0", envelope)

    assert terminalizer.transition_count == 1
    assert len(terminalizer.calls) == 1
    assert consumer.acked == [b"1-0", b"1-0"]


async def test_terminal_transition_failure_retains_pel_and_tracking_for_retry() -> None:
    terminalizer = _Terminalizer(failures=1)
    lifecycle = _Lifecycle()
    ctx = _context(terminalizer, lifecycle=lifecycle)
    supervisor = MailboxSupervisor(ctx)
    consumer = _Consumer()
    supervisor._consumer = consumer
    supervisor._last_seen_mono["child-1"] = 1.0
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("success")
    )

    await supervisor._handle_envelope(b"2-0", envelope)

    assert consumer.acked == []
    assert "child-1" in supervisor._last_seen_mono
    assert lifecycle.calls == []
    assert not await ctx.audit_repo.get_processed("root-1", "terminal-1")

    await supervisor._handle_envelope(b"2-0", envelope)

    assert terminalizer.transition_count == 1
    assert consumer.acked == [b"2-0"]
    assert "child-1" not in supervisor._last_seen_mono


async def test_cleanup_failure_retries_after_single_terminal_cas() -> None:
    terminalizer = _Terminalizer()
    lifecycle = _Lifecycle(failures=1)
    ctx = _context(terminalizer, lifecycle=lifecycle)
    supervisor = MailboxSupervisor(ctx)
    consumer = _Consumer()
    supervisor._consumer = consumer
    supervisor._last_seen_mono["child-1"] = 1.0
    envelope = _envelope(
        MailboxEnvelopeType.RESULT_READY, _result_payload("failed")
    )

    await supervisor._handle_envelope(b"3-0", envelope)

    assert terminalizer.is_terminal is True
    assert terminalizer.transition_count == 1
    assert consumer.acked == []
    assert "child-1" in supervisor._last_seen_mono

    await supervisor._handle_envelope(b"3-0", envelope)

    assert terminalizer.transition_count == 1
    assert len(terminalizer.calls) == 2  # second CAS is the idempotent loser
    assert consumer.acked == [b"3-0"]
    assert "child-1" not in supervisor._last_seen_mono
