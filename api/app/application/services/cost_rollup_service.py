"""[C2 PR-6 §14.4] CostRollupService Protocol — supervisor-side cost rollup
hook fired on ``coordinator_step`` child terminal.

The supervisor only requires the shape: ``rollup_to_parent`` accepting
``parent_session_id`` + dict-form cost summary + source attribution. Concrete
implementations land in a separate task (likely PR-6 wiring task or PR-8)
and may write ``CostRecord`` rows, emit telemetry, or update a Redis
aggregate.

This module lives in ``application/`` (not ``domain/``) because the rollup
contract is an orchestration concern owned by the supervisor — cost
accounting persistence belongs alongside the cost aggregation service
(``cost_aggregation_service.py``), not the domain core.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.domain.models.mailbox_envelope import CostAggregate


class CostRollupService(Protocol):
    """[C2 PR-6 §14.4] Application-layer Protocol; the supervisor calls
    ``rollup_to_parent`` from ``ResultReadyHandler`` when a
    ``coordinator_step`` child reaches terminal state.
    """

    async def rollup_to_parent(
        self,
        *,
        parent_session_id: str,
        cost: Mapping[str, Any],
        source: str,
        idempotency_key: str,
    ) -> None:
        """Roll up a child's terminal cost into the parent session.

        ``cost`` is a dict-shape ``CostAggregate``
        (``total_input_tokens`` / ``total_output_tokens`` / ``total_usd`` /
        ``tool_call_count``) — accepted as a ``Mapping[str, Any]`` so the
        supervisor can pass either the raw wire-form dict (after
        ``envelope.payload.get("cost_summary")``) or a Pydantic
        ``model_dump`` interchangeably.

        ``source`` is a free-form string attribution (e.g.
        ``"coordinator_subagent"``) for downstream observability.

        ``idempotency_key`` is a stable identifier (typically the
        :attr:`MailboxEnvelope.envelope_id` of the RESULT_READY event
        that triggered the rollup) — concrete implementations MUST use
        this to dedupe on PEL retry. If the same ``idempotency_key``
        has been seen before for the same ``parent_session_id``, the
        implementation MUST be a no-op (do NOT raise, do NOT
        double-count).

        Implementations MUST be idempotent — supervisor PEL retry may
        re-invoke after a transient failure, so the same envelope's
        ``cost_summary`` must not double-count. ``idempotency_key`` is
        the contract the supervisor passes for that dedup.
        """
        ...

    async def aggregate(
        self,
        *,
        coordinator_run_id: str,
        child_session_ids: list[str],
    ) -> "AggregateResult":
        """[PR-9b-B] Pull cost from durable cost_records ledger.

        Authoritative source for CoordinatorReduceEvent.cost_total. Returns
        AggregateResult(cost, missing_children) where ``missing_children`` is
        the list of child_session_ids that contributed ZERO rows to the SUM
        (either ledger empty or sessions.coordinator_run_id mismatched).

        The reducer caller uses ``missing_children`` to set
        ``diagnostics_summary='cost_unavailable: <ids>'`` per INV-B3 — zero
        cost from non-empty children is observable, not silently swallowed.
        """
        ...


@dataclass(frozen=True)
class AggregateResult:
    """[PR-9b-B] Pull-cost result + diagnostic carrier.

    Single source of truth for whether a child contributed cost AND whether
    a coordinator_run_id mismatch / empty-ledger case occurred. The reducer
    in parallel_execution_subgraph reads BOTH fields to construct the
    CoordinatorReduceEvent.
    """

    cost: CostAggregate
    missing_children: tuple[str, ...] = ()
