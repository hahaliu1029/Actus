"""[PR-9b-B Task B3] Integration tests for DbCostRollupService.

Real Postgres required (manus_test DB; see api/tests/integration/conftest.py).
These tests verify INV-B3 semantics end-to-end:

  - aggregate(...) SUMs cost_records rows that match BOTH child_session_ids
    AND sessions.coordinator_run_id; mismatched-run rows excluded.
  - missing_children contains child_session_ids that contributed ZERO rows
    (either ledger empty or sessions.coordinator_run_id mismatched).
  - rollup_to_parent(...) is idempotent on repeated idempotency_key
    (supersedes A3's MetricHookCostRollupService coverage).

NOTE: Marked ``coordinator_recovery`` so the integration runner picks them
up alongside the PR-9b-A coordinator suite. Marked ``anyio`` because the
project's async test runtime is anyio (pytest-anyio), not pytest-asyncio.
The module-level marker dispatches all async test functions through anyio.
"""
from __future__ import annotations

import uuid as _uuid
from decimal import Decimal

import pytest
from sqlalchemy import text

from app.application.services.db_cost_rollup_service import DbCostRollupService
from app.infrastructure.models.cost_record_orm import CostRecordModel
from app.infrastructure.models.session import SessionModel
from app.infrastructure.models.user import UserModel

pytestmark = [
    pytest.mark.integration,
    pytest.mark.coordinator_recovery,
    pytest.mark.anyio,
]


async def _seed_child_with_cost(
    db_session,
    *,
    user_id: str,
    parent_session_id: str,
    coordinator_run_id: str,
    total_usd: Decimal,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> str:
    """Insert child SessionModel + one CostRecordModel; return session_id."""
    sid = f"sess-b3-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=sid,
            user_id=user_id,
            parent_session_id=parent_session_id,
            coordinator_run_id=coordinator_run_id,
            status="pending",
            title="b3 child",
        )
    )
    await db_session.flush()
    db_session.add(
        CostRecordModel(
            id=str(_uuid.uuid4()),
            session_id=sid,
            user_id=user_id,
            run_id=f"run-{_uuid.uuid4().hex[:12]}",
            node_name="executor_node",
            step_ix=0,
            attempt_ix=0,
            model="gpt-4",
            provider="openai_official",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            reasoning_tokens=0,
            total_usd=total_usd,
            pricing_version="v1",
            cost_status="actual",
        )
    )
    await db_session.flush()
    return sid


async def test_aggregate_sums_children_matching_coordinator_run(
    db_session, async_session_factory,
):
    """Three children all tagged with the same coordinator_run_id contribute
    rows; aggregate SUMs the total_usd + token counts; missing_children empty.
    """
    uid = str(_uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"b3_{uid[:8]}", password_hash="x"))
    parent_sid = f"sess-b3-parent-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(id=parent_sid, user_id=uid, status="pending", title="parent")
    )
    await db_session.flush()

    run_id = f"run-b3-{_uuid.uuid4().hex[:12]}"
    c1 = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.10"),
        input_tokens=100, output_tokens=50,
    )
    c2 = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.20"),
        input_tokens=200, output_tokens=100,
    )
    c3 = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.30"),
        input_tokens=300, output_tokens=150,
    )
    await db_session.commit()

    svc = DbCostRollupService(async_session_factory=async_session_factory)
    result = await svc.aggregate(
        coordinator_run_id=run_id, child_session_ids=[c1, c2, c3],
    )
    assert result.cost.total_input_tokens == 600
    assert result.cost.total_output_tokens == 300
    # AggregateResult.cost.total_usd is a float (B3 applies float() coercion at
    # the CostAggregate boundary), so assert with approx — `== Decimal("0.60")`
    # is False against a float 0.6 and would fail on real Postgres.
    assert result.cost.total_usd == pytest.approx(0.60)
    assert result.missing_children == ()


async def test_aggregate_excludes_mismatched_coordinator_run(
    db_session, async_session_factory,
):
    """INV-B3: a child whose sessions.coordinator_run_id != caller-provided
    run_id MUST be excluded from SUM and surface in missing_children.
    """
    uid = str(_uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"b3_{uid[:8]}", password_hash="x"))
    parent_sid = f"sess-b3-parent-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(id=parent_sid, user_id=uid, status="pending", title="parent")
    )
    await db_session.flush()

    good_run = f"run-good-{_uuid.uuid4().hex[:12]}"
    bad_run = f"run-bad-{_uuid.uuid4().hex[:12]}"
    good_child = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=good_run, total_usd=Decimal("0.10"),
    )
    bad_child = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=bad_run, total_usd=Decimal("99.99"),
    )
    await db_session.commit()

    svc = DbCostRollupService(async_session_factory=async_session_factory)
    # Caller names good_run but passes BOTH children — the bad-run child must
    # be excluded by the JOIN filter and end up in missing_children.
    result = await svc.aggregate(
        coordinator_run_id=good_run, child_session_ids=[good_child, bad_child],
    )
    assert result.cost.total_usd == pytest.approx(0.10)
    assert bad_child in result.missing_children
    assert good_child not in result.missing_children


async def test_aggregate_empty_ledger_child_in_missing_children(
    db_session, async_session_factory,
):
    """A child that has a valid sessions row + matching coordinator_run_id but
    NO cost_records ledger rows MUST surface in missing_children with cost=0.
    """
    uid = str(_uuid.uuid4())
    db_session.add(UserModel(id=uid, username=f"b3_{uid[:8]}", password_hash="x"))
    parent_sid = f"sess-b3-parent-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(id=parent_sid, user_id=uid, status="pending", title="parent")
    )
    await db_session.flush()

    run_id = f"run-b3-{_uuid.uuid4().hex[:12]}"
    child_with_cost = await _seed_child_with_cost(
        db_session, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.10"),
    )
    empty_child_id = f"sess-b3-empty-{_uuid.uuid4().hex[:12]}"
    db_session.add(
        SessionModel(
            id=empty_child_id, user_id=uid,
            parent_session_id=parent_sid,
            coordinator_run_id=run_id, status="pending", title="empty",
        )
    )
    await db_session.commit()

    svc = DbCostRollupService(async_session_factory=async_session_factory)
    result = await svc.aggregate(
        coordinator_run_id=run_id,
        child_session_ids=[child_with_cost, empty_child_id],
    )
    assert result.cost.total_usd == pytest.approx(0.10)
    assert empty_child_id in result.missing_children


async def test_rollup_to_parent_idempotent(
    db_session, async_session_factory,
):
    """Replaces A3 INV-A9 coverage: repeated invocations of rollup_to_parent
    with the same idempotency_key MUST be a no-op (metric sink fired once).
    """
    sink_calls: list[dict] = []

    class _RecordingSink:
        def record(self, **kwargs):
            sink_calls.append(dict(kwargs))

    svc = DbCostRollupService(
        async_session_factory=async_session_factory,
        metric_sink=_RecordingSink(),
    )
    for _ in range(3):
        await svc.rollup_to_parent(
            idempotency_key="dupe-key",
            parent_session_id="p",
            cost={"total_input_tokens": 10, "total_output_tokens": 5,
                  "total_usd": Decimal("0.01"), "tool_call_count": 1},
            source="result_ready",
        )
    assert len(sink_calls) == 1
