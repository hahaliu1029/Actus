"""[C2 PR-6 Task 6.4 §14.4] ResultReadyHandler cost rollup hook tests.

Pins the prologue contract added in ``ResultReadyHandler._side_effect``:

- Coordinator-step child + parent_session_id present → rollup_to_parent
  invoked with the wire-form ``cost_summary`` dict and source attribution.
- Non-coordinator child (preset ``"subagent_research"`` or ``None``) → no
  rollup; destroy + callback + mark_processed body still runs.
- Coordinator-step child without parent_session_id → defensive skip.
- ``ctx.session_repo`` missing → silent skip (legacy wiring).
- ``ctx.cost_rollup_service`` missing → silent skip.
- Rollup raises → logged + swallowed; destroy still fires + completes.
- Envelope payload missing ``cost_summary`` → rollup called with empty dict.

Mock strategy: invoke ``ResultReadyHandler.handle`` + ``outcome.side_effect``
directly (no fake_redis / no main loop) and assert on the AsyncMock call
records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.mailbox_supervisor import (
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


def _make_envelope(
    *,
    envelope_id: str = "01HSPYU0CR0000000000000001",
    parent_session_id: str = "root-1",
    child_session_id: str = "child-1",
    cost_summary: Optional[dict[str, Any]] = None,
    include_cost_summary: bool = True,
) -> MailboxEnvelope:
    """Build a RESULT_READY envelope with optional ``cost_summary`` payload.

    ``include_cost_summary=False`` omits the key entirely (defensive path
    in the prologue's ``payload.get("cost_summary", {})``).
    """
    payload: dict[str, Any] = {
        "summary": "done",
        "outcome": "success",
    }
    if include_cost_summary:
        payload["cost_summary"] = cost_summary if cost_summary is not None else {
            "total_input_tokens": 1234,
            "total_output_tokens": 567,
            "total_usd": 0.0789,
            "tool_call_count": 3,
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


def _make_session(
    *,
    session_id: str = "child-1",
    parent_session_id: Optional[str] = "root-1",
    tool_filter_preset: Optional[str] = "coordinator_step",
) -> Session:
    """Construct a child Session shaped the way the prologue's gate reads it.

    Note: ``Session.parent_session_id``-vs-``worker_type`` invariant is
    enforced by CHECK constraint on the DB layer (not the Pydantic model),
    so we can mint a "subagent" child with parent_session_id set without
    needing to also flip ``worker_type``.
    """
    return Session(
        id=session_id,
        parent_session_id=parent_session_id,
        worker_type="subagent" if parent_session_id is not None else "root",
        tool_filter_preset=tool_filter_preset,  # type: ignore[arg-type]
        sandbox_binding=SandboxBinding(),
    )


class _AuditRepoStub:
    """Minimal audit-repo stub matching the surface ``ResultReadyHandler``
    touches: ``get_processed`` / ``upsert_processing`` / ``mark_processed``.

    Tracks rows in-memory so the handler's idempotency precheck behaves
    like the in-tree ``_InMemoryAuditRepo`` (see
    ``tests/domain/services/conftest.py``).
    """

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
    cost_rollup_service: Optional[AsyncMock] = None,
    session_repo: Optional[MagicMock] = None,
    sandbox_lifecycle: Optional[AsyncMock] = None,
    agent_service_callback: Optional[AsyncMock] = None,
) -> SupervisorContext:
    """Build a minimal SupervisorContext exercising only the prologue branch.

    Fields we don't touch in the rollup tests (publisher / redis /
    register_cancel_state / etc.) are filled with sentinel ``MagicMock``s
    because :class:`SupervisorContext` is a plain ``@dataclass`` — there
    is no runtime type check on those fields.
    """
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
    )


# ── Tests ────────────────────────────────────────────────────────────────────


class TestCostRollupProlog:
    """Pins the §14.4 prologue contract layered on ResultReadyHandler."""

    async def test_coordinator_step_child_rolls_up_cost(self) -> None:
        """Happy path: child has ``tool_filter_preset='coordinator_step'`` AND
        a parent_session_id → ``cost_rollup_service.rollup_to_parent`` is
        called with the dict-form ``cost_summary`` and the canonical source.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                parent_session_id="root-1",
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope(
            cost_summary={
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_usd": 0.001,
                "tool_call_count": 2,
            }
        )

        handler = ResultReadyHandler()
        outcome = await handler.handle(env, ctx)
        assert outcome.side_effect is not None
        await outcome.side_effect()

        repo.get_by_id.assert_awaited_once_with(env.child_session_id)
        rollup.rollup_to_parent.assert_awaited_once_with(
            parent_session_id="root-1",
            cost={
                "total_input_tokens": 100,
                "total_output_tokens": 50,
                "total_usd": 0.001,
                "tool_call_count": 2,
            },
            source="coordinator_subagent",
            idempotency_key=env.envelope_id,
        )
        # [codex R2 P1-5] Idempotency-key contract sanity check: the
        # kwarg flows verbatim from envelope.envelope_id.
        assert (
            rollup.rollup_to_parent.await_args.kwargs.get("idempotency_key")
            == env.envelope_id
        )
        # Destroy still ran (load-bearing safety op).
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.SUBAGENT_TERMINAL_RESULT
        )

    async def test_non_coordinator_child_skips_rollup(self) -> None:
        """``tool_filter_preset='subagent_research'`` → no rollup; destroy
        still runs and the side_effect completes normally.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="subagent_research",
                parent_session_id="root-1",
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_preset_none_skips_rollup(self) -> None:
        """``tool_filter_preset=None`` (legacy / unset) → no rollup."""
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset=None, parent_session_id="root-1"
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()

    async def test_no_parent_session_id_skips_rollup(self) -> None:
        """Defensive: coordinator_step child without parent_session_id → skip.

        This should not happen in production (the orchestrator always sets
        ``parent_session_id`` when minting a coordinator child) but the
        gate guards against malformed sessions explicitly.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                parent_session_id=None,
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_session_lookup_returns_none_skips_rollup(self) -> None:
        """``session_repo.get_by_id`` returns ``None`` (e.g. race with delete)
        → skip rollup; destroy still runs.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(return_value=None)
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_session_repo_skips_rollup(self) -> None:
        """``ctx.session_repo`` is ``None`` (legacy wiring) → silent skip; the
        rollup service is also not invoked even when wired.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=None)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_cost_rollup_service_skips_rollup(self) -> None:
        """``ctx.cost_rollup_service`` is ``None`` → silent skip; the session
        repo is also not consulted (cheap guard avoids the DB lookup).
        """
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                parent_session_id="root-1",
            )
        )
        ctx = _build_ctx(cost_rollup_service=None, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        repo.get_by_id.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_rollup_failure_does_not_block_destroy(self) -> None:
        """``rollup_to_parent`` raises → logged + swallowed; destroy still
        fires and the side_effect completes normally (no exception leaks).
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(
            side_effect=RuntimeError("rollup backend down")
        )
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                parent_session_id="root-1",
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)

        # Must NOT raise even though rollup threw.
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_awaited_once()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once_with(
            env.child_session_id, DestroyReason.SUBAGENT_TERMINAL_RESULT
        )
        # mark_processed completed too (audit row sealed) so the load-
        # bearing dedup write isn't blocked by the rollup observability path.
        assert await ctx.audit_repo.get_processed(
            env.parent_session_id, env.envelope_id
        )

    async def test_session_lookup_failure_does_not_block_destroy(self) -> None:
        """``session_repo.get_by_id`` itself raising (e.g. DB hiccup) is also
        swallowed — the rollup path is best-effort end-to-end, not just the
        ``rollup_to_parent`` call.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(side_effect=RuntimeError("db down"))
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        env = _make_envelope()
        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_not_awaited()
        ctx.sandbox_lifecycle.destroy.assert_awaited_once()

    async def test_missing_cost_summary_in_payload_passes_empty_dict(self) -> None:
        """Defensive: an envelope whose payload dict has no ``cost_summary``
        key → rollup is still called with ``cost={}``.

        The canonical post-validator path (``MailboxEnvelope.
        _validate_payload_matches_type``) materializes a default
        ``CostAggregate()`` so the wire payload always carries a
        ``cost_summary`` dict; this test bypasses that validator via
        ``model_construct`` to exercise the prologue's
        ``payload.get("cost_summary", {})`` defensive default for the
        synthetic-dispatch / future-replay edge case where a non-canonical
        envelope might reach the handler.
        """
        rollup = AsyncMock()
        rollup.rollup_to_parent = AsyncMock(return_value=None)
        repo = MagicMock()
        repo.get_by_id = AsyncMock(
            return_value=_make_session(
                tool_filter_preset="coordinator_step",
                parent_session_id="root-1",
            )
        )
        ctx = _build_ctx(cost_rollup_service=rollup, session_repo=repo)

        # ``model_construct`` skips validators → the payload dict stays
        # exactly as provided, no ``cost_summary`` materialization.
        env = MailboxEnvelope.model_construct(
            envelope_id="01HSPYU0CR0000000000000099",
            type=MailboxEnvelopeType.RESULT_READY,
            parent_session_id="root-1",
            child_session_id="child-1",
            correlation_id="01HSPYU0CR0000000000000098",
            emitted_at=datetime.now(tz=timezone.utc),
            producer_role=ProducerRole.CHILD_AGENT,
            payload={"summary": "done", "outcome": "success"},
            reclaim_count=0,
        )

        outcome = await ResultReadyHandler().handle(env, ctx)
        await outcome.side_effect()

        rollup.rollup_to_parent.assert_awaited_once_with(
            parent_session_id="root-1",
            cost={},
            source="coordinator_subagent",
            idempotency_key=env.envelope_id,
        )
