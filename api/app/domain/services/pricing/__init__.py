"""B4 M0 pricing subpackage.

Houses the static pricing table, deterministic pricing_version computation
(Issue 2B), and the 5-dim ``compute_cost`` helper used by
``CostCallbackHandler`` and ``CostAggregationService``.
"""

from .static_pricing import (
    PRICING_TABLE,
    PRICING_VERSION,
    compute_cost,
    compute_pricing_version,
    get_price,
)

__all__ = [
    "PRICING_TABLE",
    "PRICING_VERSION",
    "compute_cost",
    "compute_pricing_version",
    "get_price",
]
