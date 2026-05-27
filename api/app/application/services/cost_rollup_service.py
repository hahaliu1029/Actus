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
from typing import Any, Protocol


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
