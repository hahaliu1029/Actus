"""C2 PR-3 §6/§7 — CoordinatorEnvelopeFactory.

Factory for constructing ``MailboxEnvelope`` with the correct producer_role +
envelope_id + emitted_at. Live ``MailboxPublisher.publish(envelope)`` takes a
single MailboxEnvelope argument so callers (dispatch_node, runner, orchestrator)
delegate field assembly here and stay focused on payload semantics.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from app.domain.models.mailbox_envelope import (
    CancelAckPayload,
    CancelPolicy,
    CancelRequestPayload,
    CoordinatorBudgetSnapshot,
    CoordinatorChildContext,
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
    ResultReadyPayload,
    SpawnRequestPayload,
)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class CoordinatorEnvelopeFactory:
    """Centralised envelope assembly. Pure (no I/O); stateless instance."""

    def make_spawn_request(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        correlation_id: str,
        coordinator_run_id: str,
        work_unit_id: str,
        spawn_manifest_ref: str,
        spawn_manifest_sha256: str,
        session_mode_revision: int = 0,
        budget: Optional[CoordinatorBudgetSnapshot] = None,
        task_prompt: str = "",
    ) -> MailboxEnvelope:
        ctx = CoordinatorChildContext(
            coordinator_run_id=coordinator_run_id,
            work_unit_id=work_unit_id,
            parent_session_id=parent_session_id,
            spawn_manifest_ref=spawn_manifest_ref,
            spawn_manifest_sha256=spawn_manifest_sha256,
            session_mode_revision=session_mode_revision,
            budget=budget or CoordinatorBudgetSnapshot(
                max_tool_calls=25,
                max_token_cost_usd=0.5,
                max_wallclock_seconds=300,
            ),
        )
        payload = SpawnRequestPayload(
            agent_kind="coordinator_step",
            task_prompt=task_prompt,
            coordinator_context=ctx,
        )
        return MailboxEnvelope(
            envelope_id=str(uuid.uuid4()),
            type=MailboxEnvelopeType.SPAWN_REQUEST,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            correlation_id=correlation_id,
            emitted_at=_now_utc(),
            producer_role=ProducerRole.PARENT_AGENT,
            payload=payload.model_dump(mode="python"),
        )

    def make_cancel_request(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        correlation_id: str,
        reason: str,
        policy: CancelPolicy = CancelPolicy.REQUEST_CANCEL,
    ) -> MailboxEnvelope:
        payload = CancelRequestPayload(reason=reason, policy=policy)
        return MailboxEnvelope(
            envelope_id=str(uuid.uuid4()),
            type=MailboxEnvelopeType.CANCEL_REQUEST,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            correlation_id=correlation_id,
            emitted_at=_now_utc(),
            producer_role=ProducerRole.PARENT_AGENT,
            payload=payload.model_dump(mode="python"),
        )

    def make_result_ready(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        correlation_id: str,
        payload: ResultReadyPayload,
    ) -> MailboxEnvelope:
        return MailboxEnvelope(
            envelope_id=str(uuid.uuid4()),
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            correlation_id=correlation_id,
            emitted_at=_now_utc(),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=payload.model_dump(mode="python"),
        )

    def make_cancel_ack(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        correlation_id: str,
        payload: CancelAckPayload,
    ) -> MailboxEnvelope:
        return MailboxEnvelope(
            envelope_id=str(uuid.uuid4()),
            type=MailboxEnvelopeType.CANCEL_ACK,
            parent_session_id=parent_session_id,
            child_session_id=child_session_id,
            correlation_id=correlation_id,
            emitted_at=_now_utc(),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=payload.model_dump(mode="python"),
        )
