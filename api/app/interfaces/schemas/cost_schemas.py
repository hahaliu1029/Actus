"""B4 M0 Phase I: cost endpoint response schemas.

Decimal fields serialize to strings so consumers don't lose precision at
float conversion time (``total_usd`` uses 10 decimal places so small
cache-hit deltas stay visible).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_serializer


def _dec_to_str(v: Decimal) -> str:
    return format(v, "f")


class CostAggregateResponse(BaseModel):
    """Serialized rollup for GET /sessions/{id}/cost."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_usd: Decimal = Field(..., description="Total USD cost across the session.")
    record_count: int
    by_node: dict[str, Decimal]
    by_model: dict[str, Decimal]
    by_provider: dict[str, Decimal]
    pricing_version: str
    cost_status: str = Field(
        ..., description="One of: actual | estimated | partial | unknown"
    )
    first_record_at: Optional[datetime]
    last_record_at: Optional[datetime]
    has_partial_records: bool

    @field_serializer("total_usd")
    def _ser_total(self, v: Decimal) -> str:
        return _dec_to_str(v)

    @field_serializer("by_node", "by_model", "by_provider")
    def _ser_breakdown(self, v: dict[str, Decimal]) -> dict[str, str]:
        return {k: _dec_to_str(amount) for k, amount in v.items()}
