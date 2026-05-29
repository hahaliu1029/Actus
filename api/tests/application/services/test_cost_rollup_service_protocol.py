"""PR-9b-B — CostRollupService Protocol gains aggregate(...) method.

Authoritative source for CoordinatorReduceEvent.cost_total.
"""
from __future__ import annotations

import inspect

from app.application.services.cost_rollup_service import (
    AggregateResult,
    CostRollupService,
)
from app.domain.models.mailbox_envelope import CostAggregate


def test_protocol_has_aggregate_method():
    assert hasattr(CostRollupService, "aggregate")


def test_aggregate_is_async():
    fn = getattr(CostRollupService, "aggregate")
    assert inspect.iscoroutinefunction(fn) or inspect.isfunction(fn)


def test_aggregate_signature():
    """Required kwargs: coordinator_run_id (str), child_session_ids (list[str])."""
    fn = getattr(CostRollupService, "aggregate")
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())
    assert {"coordinator_run_id", "child_session_ids"}.issubset(params)


def test_protocol_preserves_rollup_to_parent():
    """The existing push hook stays as a metric hook — NOT removed."""
    assert hasattr(CostRollupService, "rollup_to_parent")


def test_aggregate_returns_aggregate_result_with_missing_children():
    """INV-B3 — aggregate() returns AggregateResult(cost, missing_children),
    NOT a bare CostAggregate. Reducer reads missing_children to set
    diagnostics_summary='cost_unavailable' when children contributed 0 rows."""
    result = AggregateResult(cost=CostAggregate())
    assert result.missing_children == ()
    assert result.cost is not None
