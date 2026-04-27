"""B4 M0 Phase C: CostRecord frozen dataclass contract.

Locks the shape of the domain-level cost row so downstream B4 components
(CostCallbackHandler, CostAggregationService, GET /cost serializers) can rely
on a stable set of fields. Frozen so instances cannot be mutated mid-flight
between the handler and the persister.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.models.cost_record import CostRecord, CostStatus


def _sample(**overrides: object) -> CostRecord:
    defaults: dict[str, object] = {
        "id": str(uuid4()),
        "session_id": "sess-abc",
        "user_id": "user-xyz",
        "run_id": str(uuid4()),
        "node_name": "planner",
        "step_ix": 3,
        "attempt_ix": 0,
        "model": "gpt-4o",
        "provider": "openai_official",
        "input_tokens": 42,
        "output_tokens": 17,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "total_usd": Decimal("0.0123456789"),
        "pricing_version": "abc123def456",
        "cost_status": CostStatus.ACTUAL,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return CostRecord(**defaults)  # type: ignore[arg-type]


class TestCostRecordDomainModel:
    def test_can_construct_with_all_fields(self) -> None:
        record = _sample()
        assert record.input_tokens == 42
        assert record.cost_status == CostStatus.ACTUAL
        assert record.total_usd == Decimal("0.0123456789")

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        record = _sample()
        with pytest.raises(FrozenInstanceError):
            record.input_tokens = 99  # type: ignore[misc]

    def test_created_at_must_be_tz_aware(self) -> None:
        """Issue OV-3: naive datetimes are rejected at construction time.

        The aggregation sorts by (created_at, step_ix) tuples; mixing naive
        and aware datetimes raises TypeError mid-sort and masks real data.
        Enforce tz-awareness at the domain boundary.
        """
        naive = datetime(2026, 4, 24, 12, 0, 0)
        with pytest.raises(ValueError, match="timezone"):
            _sample(created_at=naive)


class TestCostStatus:
    def test_has_four_states(self) -> None:
        assert {s.value for s in CostStatus} == {
            "actual",
            "estimated",
            "partial",
            "unknown",
        }

    def test_row_level_values_are_strings(self) -> None:
        assert CostStatus.ACTUAL.value == "actual"
        assert CostStatus.ESTIMATED.value == "estimated"
        assert CostStatus.PARTIAL.value == "partial"
        assert CostStatus.UNKNOWN.value == "unknown"
