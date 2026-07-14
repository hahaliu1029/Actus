"""Cascade TERMINATE echo must carry authoritative coordinator lineage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.coordinator_terminal_transition import (
    terminalize_authoritative_coordinator_child,
)
from app.application.services.mailbox_supervisor import (
    CancelAckHandler,
    CancelRequestHandler,
    MailboxSupervisor,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    CancelPolicy,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import DestroyReason, Session, SessionStatus
from app.domain.services.session.default_state_machine import (
    DefaultSessionStateMachine,
)


pytestmark = pytest.mark.anyio


class _AuditRepo:
    def __init__(self) -> None:
        self.processed: set[tuple[str, str]] = set()

    async def get_processed(self, parent: str, envelope_id: str) -> bool:
        return (parent, envelope_id) in self.processed

    async def upsert_processing(
        self,
        envelope: MailboxEnvelope,
        *,
        processing_at: datetime,
    ) -> None:
        del envelope, processing_at

    async def mark_processed(
        self,
        parent: str,
        envelope_id: str,
        *,
        processed_at: datetime,
    ) -> None:
        del processed_at
        self.processed.add((parent, envelope_id))


class _Publisher:
    def __init__(self) -> None:
        self.published: list[MailboxEnvelope] = []

    async def publish(self, envelope: MailboxEnvelope) -> None:
        self.published.append(envelope)


class _Lifecycle:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DestroyReason]] = []

    async def destroy(self, child_id: str, reason: DestroyReason) -> None:
        self.calls.append((child_id, reason))


class _Reader:
    def __init__(self, result: Session | None | BaseException) -> None:
        self.result = result
        self.calls: list[str] = []

    async def get_by_id(self, child_id: str) -> Session | None:
        self.calls.append(child_id)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class _LockedRepo:
    def __init__(self, row: Session) -> None:
        self.row = row
        self.update_count = 0
        self.lock_reads: list[str] = []

    async def get_by_id_for_update(self, child_id: str) -> Session:
        self.lock_reads.append(child_id)
        return self.row

    async def update_to_terminal(
        self,
        child_id: str,
        status: SessionStatus,
        reason: str,
    ) -> bool:
        assert child_id == self.row.id
        if self.row.status in (SessionStatus.COMPLETED, SessionStatus.TIMED_OUT):
            return False
        self.row.status = status
        self.row.terminal_reason = reason
        self.update_count += 1
        return True


class _UoW:
    def __init__(self, repo: _LockedRepo) -> None:
        self.session = repo
        self.commit = AsyncMock()

    async def __aenter__(self) -> "_UoW":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None


def _coordinator_row(**overrides: Any) -> Session:
    values: dict[str, Any] = {
        "id": "child-1",
        "parent_session_id": "root-1",
        "root_session_id": "root-1",
        "worker_type": "subagent",
        "subagent_control_plane": "mailbox",
        "tool_filter_preset": "coordinator_step",
        "coordinator_run_id": "root-1:step-hash:a2",
        "work_unit_id": "wu-1",
        "status": SessionStatus.RUNNING,
    }
    values.update(overrides)
    return Session(**values)


def _build_harness(
    row: Session,
    *,
    reader_result: Session | None | BaseException | object = ...,
) -> tuple[SupervisorContext, MailboxSupervisor, _Publisher, _LockedRepo, _UoW]:
    publisher = _Publisher()
    lifecycle = _Lifecycle()
    locked_repo = _LockedRepo(row)
    uow = _UoW(locked_repo)
    state_machine = DefaultSessionStateMachine(uow_factory=lambda: uow)
    reader = _Reader(row if reader_result is ... else reader_result)  # type: ignore[arg-type]
    ctx = SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-1",
        instance_id="instance-1",
        redis=MagicMock(),
        audit_repo=_AuditRepo(),
        publisher=publisher,
        sandbox_lifecycle=lifecycle,
        agent_service_callback=AsyncMock(),
        telemetry=AsyncMock(),
        session_repo=reader,  # type: ignore[arg-type]
    )

    async def _terminalize(command) -> bool:  # noqa: ANN001
        return await terminalize_authoritative_coordinator_child(
            command,
            state_machine=state_machine,
            uow_factory=lambda: uow,
        )

    ctx.terminalize_child = _terminalize
    supervisor = MailboxSupervisor(ctx, block_ms=0, idle_poll_sleep_s=0.01)
    return ctx, supervisor, publisher, locked_repo, uow


async def _emit_and_handle_cascade(
    ctx: SupervisorContext,
    supervisor: MailboxSupervisor,
    publisher: _Publisher,
) -> MailboxEnvelope:
    await supervisor._emit_cascade_terminate(  # noqa: SLF001
        "child-1",
        reason="orphan_timeout",
        destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
    )
    cascade = publisher.published[-1]
    assert cascade.type == MailboxEnvelopeType.CANCEL_REQUEST
    assert cascade.producer_role == ProducerRole.SUPERVISOR
    assert cascade.correlation_id.startswith("cascade:")

    outcome = await CancelRequestHandler().handle(cascade, ctx)
    assert outcome.side_effect is not None
    await outcome.side_effect()
    echo = publisher.published[-1]
    assert echo.type == MailboxEnvelopeType.CANCEL_ACK
    assert echo.producer_role == ProducerRole.SUPERVISOR_ECHO
    return echo


async def test_real_cascade_echo_terminalizes_coordinator_row_once() -> None:
    row = _coordinator_row()
    ctx, supervisor, publisher, locked_repo, uow = _build_harness(row)

    echo = await _emit_and_handle_cascade(ctx, supervisor, publisher)

    assert echo.correlation_id == "root-1:step-hash:a2"
    ack_outcome = await CancelAckHandler().handle(echo, ctx)
    assert ack_outcome.side_effect is not None
    await ack_outcome.side_effect()

    assert row.status == SessionStatus.TIMED_OUT
    assert locked_repo.update_count == 1
    assert locked_repo.lock_reads == ["child-1"]
    uow.commit.assert_awaited_once()


async def test_run_flip_between_echo_read_and_consumption_is_refused() -> None:
    row = _coordinator_row()
    ctx, supervisor, publisher, locked_repo, uow = _build_harness(row)
    echo = await _emit_and_handle_cascade(ctx, supervisor, publisher)
    assert echo.correlation_id == "root-1:step-hash:a2"

    row.coordinator_run_id = "root-1:step-hash:a3"
    ack_outcome = await CancelAckHandler().handle(echo, ctx)
    assert ack_outcome.side_effect is not None
    await ack_outcome.side_effect()

    assert row.status == SessionStatus.RUNNING
    assert locked_repo.update_count == 0
    uow.commit.assert_not_awaited()


@pytest.mark.parametrize(
    "reader_result",
    [
        RuntimeError("authoritative reader unavailable"),
        None,
        _coordinator_row(parent_session_id="other-root"),
    ],
)
async def test_coordinator_echo_lineage_read_failure_retains_cascade_for_retry(
    reader_result: Session | None | BaseException,
) -> None:
    row = _coordinator_row()
    ctx, supervisor, publisher, _locked_repo, _uow = _build_harness(
        row,
        reader_result=reader_result,
    )

    await supervisor._emit_cascade_terminate(  # noqa: SLF001
        "child-1",
        reason="orphan_timeout",
        destroy_reason=DestroyReason.ORPHAN_TIMEOUT,
    )
    cascade = publisher.published[-1]
    outcome = await CancelRequestHandler().handle(cascade, ctx)
    assert outcome.side_effect is not None

    with pytest.raises(RuntimeError, match="coordinator cascade echo"):
        await outcome.side_effect()

    assert len(publisher.published) == 1
    assert not await ctx.audit_repo.get_processed(
        cascade.parent_session_id, cascade.envelope_id
    )
    assert ctx.sandbox_lifecycle.calls == [
        ("child-1", DestroyReason.ORPHAN_TIMEOUT)
    ]


async def test_legacy_cascade_keeps_original_echo_contract() -> None:
    row = _coordinator_row(
        subagent_control_plane="legacy",
        tool_filter_preset="subagent_research",
        coordinator_run_id=None,
        work_unit_id=None,
    )
    ctx, supervisor, publisher, _locked_repo, _uow = _build_harness(row)

    echo = await _emit_and_handle_cascade(ctx, supervisor, publisher)

    assert echo.correlation_id.startswith("cascade:")
    assert len(publisher.published) == 2
