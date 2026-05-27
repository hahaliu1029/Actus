"""[C2 PR-6 §14.4] SessionCostSnapshot + CostSource state machine.

Determines the attribution of a session's cumulative cost — whether it came
directly from the session's own LLM/tool calls, from coordinator subagent
children, from research subagent children, or a mix. Consumed by the
``GET /cost/tree`` augmentation so the frontend can render attribution
breakdowns.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CostSource(StrEnum):
    """Attribution bucket for a session's rolled-up cost.

    State machine derived from the three non-negative cost dimensions on
    ``SessionCostSnapshot``. A snapshot with all zeros lands at ``NONE``;
    a snapshot with exactly one positive dimension reports that
    dimension; any combination of two or three positives is ``MIXED``.
    """

    NONE = "none"
    DIRECT = "direct"
    COORDINATOR_SUBAGENT = "coordinator_subagent"
    RESEARCH_SUBAGENT = "research_subagent"
    MIXED = "mixed"


@dataclass(frozen=True)
class SessionCostSnapshot:
    """Immutable cost attribution snapshot for a single session.

    Float-typed per spec §14.4 — the snapshot is a UI-facing projection of
    the underlying ``Decimal`` cost rows, so any precision loss is
    cosmetic (the source ``CostTreeAggregate.total_cost.total_usd``
    remains the authoritative ledger number).
    """

    direct_cost_usd: float
    coordinator_child_cost_usd: float
    research_child_cost_usd: float

    @property
    def total_cost_usd(self) -> float:
        return (
            self.direct_cost_usd
            + self.coordinator_child_cost_usd
            + self.research_child_cost_usd
        )

    @property
    def cost_source(self) -> CostSource:
        sources: list[CostSource] = []
        if self.direct_cost_usd > 0:
            sources.append(CostSource.DIRECT)
        if self.coordinator_child_cost_usd > 0:
            sources.append(CostSource.COORDINATOR_SUBAGENT)
        if self.research_child_cost_usd > 0:
            sources.append(CostSource.RESEARCH_SUBAGENT)
        if not sources:
            return CostSource.NONE
        if len(sources) == 1:
            return sources[0]
        return CostSource.MIXED
