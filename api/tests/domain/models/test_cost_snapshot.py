"""[C2 PR-6 §14.4] SessionCostSnapshot + CostSource state machine tests.

Locks the eight transitions on the ``cost_source`` derived property and the
``total_cost_usd`` rollup. Frozen dataclass invariant is asserted via
``FrozenInstanceError`` so the snapshot can be safely passed by reference
between application service and the ``GET /cost/tree`` serializer without
shadow-mutation risk.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.domain.models.cost_snapshot import CostSource, SessionCostSnapshot


def test_cost_source_none() -> None:
    """All-zero snapshot collapses to NONE (no attribution claim)."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.0,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.0,
    )
    assert snap.cost_source == CostSource.NONE


def test_cost_source_direct() -> None:
    """Only direct > 0 → DIRECT."""
    snap = SessionCostSnapshot(
        direct_cost_usd=1.5,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.0,
    )
    assert snap.cost_source == CostSource.DIRECT


def test_cost_source_coordinator() -> None:
    """Only coordinator > 0 → COORDINATOR_SUBAGENT."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.0,
        coordinator_child_cost_usd=2.25,
        research_child_cost_usd=0.0,
    )
    assert snap.cost_source == CostSource.COORDINATOR_SUBAGENT


def test_cost_source_research() -> None:
    """Only research > 0 → RESEARCH_SUBAGENT."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.0,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.75,
    )
    assert snap.cost_source == CostSource.RESEARCH_SUBAGENT


def test_cost_source_mixed_direct_plus_coordinator() -> None:
    """Direct + coordinator both > 0 → MIXED."""
    snap = SessionCostSnapshot(
        direct_cost_usd=1.0,
        coordinator_child_cost_usd=0.5,
        research_child_cost_usd=0.0,
    )
    assert snap.cost_source == CostSource.MIXED


def test_cost_source_mixed_direct_plus_research() -> None:
    """Direct + research both > 0 → MIXED (covers second 2-of-3 combo)."""
    snap = SessionCostSnapshot(
        direct_cost_usd=1.0,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.25,
    )
    assert snap.cost_source == CostSource.MIXED


def test_cost_source_mixed_coordinator_plus_research() -> None:
    """Coordinator + research both > 0 → MIXED (covers third 2-of-3 combo)."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.0,
        coordinator_child_cost_usd=0.4,
        research_child_cost_usd=0.6,
    )
    assert snap.cost_source == CostSource.MIXED


def test_cost_source_mixed_all_three() -> None:
    """All three > 0 → MIXED."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.25,
        coordinator_child_cost_usd=0.5,
        research_child_cost_usd=0.75,
    )
    assert snap.cost_source == CostSource.MIXED


def test_total_cost_sums_all_three() -> None:
    """``total_cost_usd`` is the unweighted sum of the three dimensions."""
    snap = SessionCostSnapshot(
        direct_cost_usd=1.0,
        coordinator_child_cost_usd=2.0,
        research_child_cost_usd=4.0,
    )
    assert snap.total_cost_usd == pytest.approx(7.0)


def test_total_cost_zero_when_all_zero() -> None:
    """Empty-attribution snapshot reports zero total (no float drift)."""
    snap = SessionCostSnapshot(
        direct_cost_usd=0.0,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.0,
    )
    assert snap.total_cost_usd == 0.0


def test_frozen_dataclass_immutable() -> None:
    """``frozen=True`` blocks post-construction mutation of any dimension."""
    snap = SessionCostSnapshot(
        direct_cost_usd=1.0,
        coordinator_child_cost_usd=0.0,
        research_child_cost_usd=0.0,
    )
    with pytest.raises(FrozenInstanceError):
        snap.direct_cost_usd = 99.0  # type: ignore[misc]


def test_cost_source_string_values() -> None:
    """StrEnum values match wire contract consumed by frontend."""
    assert CostSource.NONE.value == "none"
    assert CostSource.DIRECT.value == "direct"
    assert CostSource.COORDINATOR_SUBAGENT.value == "coordinator_subagent"
    assert CostSource.RESEARCH_SUBAGENT.value == "research_subagent"
    assert CostSource.MIXED.value == "mixed"
