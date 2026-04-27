"""B4 M0 Phase C: CostRecord — per-LLM-call cost row in the domain layer.

One row per completed LLM call. The callback handler (``CostCallbackHandler``)
builds these from ``AIMessage.usage_metadata`` + node metadata from LangGraph,
and the aggregation service reads them back when a caller asks
``GET /api/v1/sessions/{id}/cost``.

Design invariants:
- Frozen so the value cannot be mutated between handler and persister.
- ``created_at`` is tz-aware (Issue OV-3) so tuple-sort in the aggregation
  never explodes on mixed naive/aware rows.
- 5-dim token model (input / output / cache_read / cache_write / reasoning)
  covers OpenAI Chat Completions, Responses API, and Anthropic-style cache
  creation without losing precision at the domain boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum


class CostStatus(str, Enum):
    """Row-level or aggregate-level confidence in the cost number.

    - ``actual``: provider returned usage; numbers reflect what was billed.
    - ``unknown``: usage was missing (no ``usage_metadata``) or the model
      wasn't priced. Row-level in M0 whenever the provider didn't give us
      tokens, or when the handler's persist-failure fallback writes a
      degraded marker.
    - ``partial``: aggregate-only — a session that mixes ``actual`` with
      ``unknown`` (or contains a degraded-marker row) reports ``partial``
      instead of silently rolling up as clean ``actual``.
    - ``estimated``: reserved for future use. M0 does NOT compute a
      char-count estimate; ``usage_metadata is None`` currently records as
      ``unknown``. The enum value is kept so a later milestone can
      introduce real estimation without a schema migration.
    """

    ACTUAL = "actual"
    ESTIMATED = "estimated"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class CostRecord:
    """Single-row cost tally for one completed LLM invocation."""

    id: str
    session_id: str
    user_id: str
    run_id: str
    node_name: str
    step_ix: int
    attempt_ix: int
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    total_usd: Decimal
    pricing_version: str
    cost_status: CostStatus
    created_at: datetime

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError(
                "CostRecord.created_at must be timezone-aware (Issue OV-3); "
                "naive datetimes break tuple-sort aggregation."
            )
