"""[C2 PR-7 Task 7.4 §12.4] Mailbox supervisor persist_terminal hook tests.

Pins the PROLOGUE contract added to BOTH ``ResultReadyHandler._side_effect``
and ``CancelAckHandler._side_effect``:

- ``ResultReadyHandler``: coordinator_step child + work_unit_id set →
  persist_terminal called with envelope_type="RESULT_READY"; otherwise skipped.
- ``CancelAckHandler``: same gate, envelope_type="CANCEL_ACK".
- Persist failure (DB hiccup, IntegrityError replay) MUST NOT abort the
  load-bearing destroy + audit body — best-effort observability path.
- Missing ``coordinator_envelope_store`` OR missing ``session_repo`` →
  silent skip (legacy / pre-PR-7 contexts).
- Missing ``coordinator_run_id`` / ``work_unit_id`` on the child session →
  defensive skip (the orchestrator always sets both on coordinator_step
  children, but the gate guards malformed sessions).

Mock strategy mirrors ``test_mailbox_supervisor_cost_rollup.py``: invoke
``handler.handle`` + ``outcome.side_effect`` directly (no fake_redis / no
main loop) and assert on the AsyncMock call records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.mailbox_supervisor import (
    CancelAckHandler,
    ResultReadyHandler,
    SupervisorContext,
)
from app.domain.models.mailbox_envelope import (
    MailboxEnvelope,
    MailboxEnvelopeType,
    ProducerRole,
)
from app.domain.models.session import (
    DestroyReason,
    SandboxBinding,
    Session,
)


pytestmark = pytest.mark.anyio


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result_ready_envelope(
    *,
    envelope_id: str = "01HSPYU0CR0000000000000001",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    payload: Optional[dict[str, Any]] = None,
) -> MailboxEnvelope:
    """Build a RESULT_READY envelope."""
    if payload is None:
        payload = {
            "summary": "done",
            "outcome": "success",
            "cost_summary": {
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_usd": 0.001,
                "tool_call_count": 2,
            },
        }
    return MailboxEnvelope(
        envelope_id=envelope_id,
        type=MailboxEnvelopeType.RESULT_READY,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id="01HSPYU0CR0000000000000002",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=ProducerRole.CHILD_AGENT,
        payload=payload,
        reclaim_count=0,
    )


def _make_cancel_ack_envelope(
    *,
    envelope_id: str = "01HSPYU0CR0000000000000010",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    payload: Optional[dict[str, Any]] = None,
    producer_role: ProducerRole = ProducerRole.CHILD_AGENT,
) -> MailboxEnvelope:
    """Build a CANCEL_ACK envelope. ``producer_role`` defaults to CHILD_AGENT
    (the supervisor-echo short-circuit only triggers on SUPERVISOR_ECHO and
    bypasses the persist PROLOGUE entirely — see CancelAckHandler doc).
    """
    if payload is None:
        payload = {"final_state": "cancelled"}
    return MailboxEnvelope(
        envelope_id=envelope_id,
        type=MailboxEnvelopeType.CANCEL_ACK,
        parent_session_id=parent_session_id,
        child_session_id=child_session_id,
        correlation_id="01HSPYU0CR0000000000000011",
        emitted_at=datetime.now(tz=timezone.utc),
        producer_role=producer_role,
        payload=payload,
        reclaim_count=0,
    )


def _make_session(
    *,
    session_id: str = "child-1",
    parent_session_id: Optional[str] = "root-1",
    tool_filter_preset: Optional[str] = "coordinator_step",
    coordinator_run_id: Optional[str] = "run-1",
    work_unit_id: Optional[str] = "wu-1",
) -> Session:
    """Construct a child Session shaped the way the PROLOGUE gate reads it."""
    return Session(
        id=session_id,
        parent_session_id=parent_session_id,
        worker_type="subagent" if parent_session_id is not None else "root",
        tool_filter_preset=tool_filter_preset,  # type: ignore[arg-type]
        sandbox_binding=SandboxBinding(),
        coordinator_run_id=coordinator_run_id,
        work_unit_id=work_unit_id,
    )


class _AuditRepoStub:
    """Minimal audit-repo stub matching the surface both handlers touch."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], dict[str, Any]] = {}

    async def get_processed(
        self, parent_session_id: str, envelope_id: str
    ) -> bool:
        row = self.rows.get((parent_session_id, envelope_id))
        return row is not None and row.get("processed_at") is not None

    async def upsert_processing(
        self, envelope: MailboxEnvelope, *, processing_at: datetime
    ) -> None:
        key = (envelope.parent_session_id, envelope.envelope_id)
        row = self.rows.setdefault(key, {})
        row["processing_at"] = processing_at

    async def mark_processed(
        self,
        parent_session_id: str,
        envelope_id: str,
        *,
        processed_at: datetime,
    ) -> None:
        self.rows.setdefault((parent_session_id, envelope_id), {})[
            "processed_at"
        ] = processed_at


def _build_ctx(
    *,
    coordinator_envelope_store: Optional[AsyncMock] = None,
    session_repo: Optional[MagicMock] = None,
    cost_rollup_service: Optional[AsyncMock] = None,
    sandbox_lifecycle: Optional[AsyncMock] = None,
    agent_service_callback: Optional[AsyncMock] = None,
) -> SupervisorContext:
    """Build a minimal SupervisorContext exercising the persist-terminal path."""
    if sandbox_lifecycle is None:
        sandbox_lifecycle = AsyncMock()
        sandbox_lifecycle.destroy = AsyncMock(return_value=None)
    if agent_service_callback is None:
        agent_service_callback = AsyncMock(return_value=None)

    telemetry = AsyncMock()
    telemetry.emit = AsyncMock(return_value=None)

    return SupervisorContext(
        root_session_id="root-1",
        pod_id="pod-a",
        instance_id="i1",
        redis=MagicMock(),  # not exercised on the prologue path
        audit_repo=_AuditRepoStub(),
        publisher=MagicMock(),  # not exercised on the prologue path
        sandbox_lifecycle=sandbox_lifecycle,
        agent_service_callback=agent_service_callback,
        telemetry=telemetry,
        session_repo=session_repo,
        cost_rollup_service=cost_rollup_service,
        coordinator_envelope_store=coordinator_envelope_store,
    )


# ── ResultReadyHandler persist tests ─────────────────────────────────────────


class TestResultReadyPersistTerminal:
    """Pins the §12.4 persist-terminal PROLOGUE on ResultReadyHandler."""

    async def test_coordinator_step_child_persists_envelope(self) -> None:
        """Happy path: coordinator_step child + run_id + work_unit_id set →
        ``persist_terminal`` called with envelope_type="RESULT_READY" and
        the wire-form payload dict.
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id="wu-1",
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        assert outcome.side_effect is not None
        await outcome.side_effect()

        store.persist_terminal.assert_awaited_once_with(
            coordinator_run_id="run-1",
            work_unit_id="wu-1",
            child_session_id=env.child_session_id,
            envelope_type="RESULT_READY",
            payload=env.payload,
        )
        # Destroy still ran (load-bearing safety op).
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.SUBAGENT_TERMINAL_RESULT
        )

    async def test_research_child_does_not_persist(self) -> None:
        """``tool_filter_preset='subagent_research'`` → no persist; destroy
        still runs.
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="subagent_research",
                coordinator_run_id=None,
                work_unit_id=None,
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_persist_failure_does_not_block_destroy(self) -> None:
        """``persist_terminal`` raises → logged + swallowed; destroy still
        fires and side_effect completes normally (no exception leaks).
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(
            side_effect=RuntimeError("envelope store down")
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id="wu-1",
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)

        # Must NOT raise even though persist threw.
        await outcome.side_effect()

        store.persist_terminal.assert_awaited_once()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.SUBAGENT_TERMINAL_RESULT
        )
        # Mark_processed completed too — the persist observability path
        # must not block the dedup write.
        assert await ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    async def test_envelope_store_none_silent_noop(self) -> None:
        """``coordinator_envelope_store=None`` (legacy wiring) → silent skip;
        session repo NOT consulted (cheap guard avoids the DB lookup).
        """
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(tool_filter_preset="coordinator_step")
        )
        ctx = _build_ctx(
            coordinator_envelope_store=None,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.get_by_id.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_session_repo_none_silent_noop(self) -> None:
        """``session_repo=None`` (legacy / pre-PR-7) → gate trips, no persist."""
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=None,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_no_child_session_skipped(self) -> None:
        """``session_repo.get_by_id`` returns None (race with delete) → skip."""
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=None)
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_work_unit_id_skipped(self) -> None:
        """coordinator_step child but ``work_unit_id=None`` → defensive skip."""
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id=None,
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_coordinator_run_id_skipped(self) -> None:
        """coordinator_step child but ``coordinator_run_id=None`` → skip."""
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id=None,
                work_unit_id="wu-1",
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_result_ready_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_non_dict_payload_passes_empty_dict(self) -> None:
        """Defensive: envelope payload that isn't a dict (synthetic-dispatch
        edge case) → persist still called with payload={}.
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(tool_filter_preset="coordinator_step")
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = MailboxEnvelope.model_construct(
            envelope_id="01HSPYU0CR0000000000000099",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id="root-1",
            child_session_id="child-1",
            correlation_id="01HSPYU0CR0000000000000098",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload=None,  # bypass validator → exercise non-dict default
            reclaim_count=0,
        )

        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_awaited_once_with(
            coordinator_run_id="run-1",
            work_unit_id="wu-1",
            child_session_id="child-1",
            envelope_type="RESULT_READY",
            payload={},
        )


# ── CancelAckHandler persist tests ───────────────────────────────────────────


class TestCancelAckPersistTerminal:
    """Pins the §12.4 persist-terminal PROLOGUE on CancelAckHandler."""

    async def test_coordinator_step_child_persists_envelope(self) -> None:
        """Happy path: cooperative CANCEL_ACK on coordinator_step child →
        ``persist_terminal`` called with envelope_type="CANCEL_ACK".
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id="wu-1",
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope()
        outcome = await CancelAckHandler().handle(env, ctx)
        assert outcome.side_effect is not None
        await outcome.side_effect()

        store.persist_terminal.assert_awaited_once_with(
            coordinator_run_id="run-1",
            work_unit_id="wu-1",
            child_session_id=env.child_session_id,
            envelope_type="CANCEL_ACK",
            payload=env.payload,
        )
        # Destroy still ran with CANCEL_ACK_OBSERVED reason.
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.CANCEL_ACK_OBSERVED
        )

    async def test_research_child_does_not_persist(self) -> None:
        """Non-coordinator child CANCEL_ACK → no persist; destroy still runs."""
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="subagent_research",
                coordinator_run_id=None,
                work_unit_id=None,
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope()
        outcome = await CancelAckHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_persist_failure_does_not_block_destroy(self) -> None:
        """``persist_terminal`` raises → logged + swallowed; CANCEL_ACK
        destroy still fires (side_effect completes normally).
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(
            side_effect=RuntimeError("envelope store down")
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id="wu-1",
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope()
        outcome = await CancelAckHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_awaited_once()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.CANCEL_ACK_OBSERVED
        )

    async def test_supervisor_echo_short_circuits_before_persist(self) -> None:
        """SUPERVISOR_ECHO short-circuits the entire side_effect path —
        no persist, no destroy, just ack+mark_processed (existing R2-4
        contract, preserved by PR-7).
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(tool_filter_preset="coordinator_step")
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope(
            producer_role=ProducerRole.SUPERVISOR_ECHO,
        )
        outcome = await CancelAckHandler().handle(env, ctx)

        # supervisor-echo outcome: ack=True, side_effect=None.
        assert outcome.ack is True
        assert outcome.side_effect is None
        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_not_awaited()

    async def test_envelope_store_none_silent_noop(self) -> None:
        """``coordinator_envelope_store=None`` → silent skip; destroy ok."""
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(tool_filter_preset="coordinator_step")
        )
        ctx = _build_ctx(
            coordinator_envelope_store=None,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope()
        outcome = await CancelAckHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.get_by_id.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_work_unit_id_skipped(self) -> None:
        """CANCEL_ACK on coordinator_step child with ``work_unit_id=None`` →
        defensive skip; destroy still runs.
        """
        store = AsyncMock()
        store.persist_terminal = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                coordinator_run_id="run-1",
                work_unit_id=None,
            )
        )
        ctx = _build_ctx(
            coordinator_envelope_store=store,
            session_repo=repo,
        )

        env = _make_cancel_ack_envelope()
        outcome = await CancelAckHandler().handle(env, ctx)
        await outcome.side_effect()

        store.persist_terminal.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()
