"""[PR-9b-B Task B3] Integration tests for DbCostRollupService.

Real Postgres required (manus_test DB; see api/tests/integration/conftest.py).
These tests verify INV-B3 semantics end-to-end:

  - aggregate(...) SUMs cost_records rows that match BOTH child_session_ids
    AND sessions.coordinator_run_id; mismatched-run rows excluded.
  - missing_children contains child_session_ids that contributed ZERO rows
    (either ledger empty or sessions.coordinator_run_id mismatched).
  - rollup_to_parent(...) is idempotent on repeated idempotency_key
    (supersedes A3's MetricHookCostRollupService coverage).

LOCATION (codex R3-F1, HIGH): this module lives under ``tests/integration/``
— NOT ``tests/application/services/`` — because it is marked
``coordinator_recovery`` and CI's recovery pass (``pytest -m "coordinator_recovery
and not sandbox ..."`` run from ``api/`` over ALL of ``tests/``) selects it. The
``async_session_factory`` (and the ``coordinator_truncation`` / ``fresh_test_user``
fixtures) it consumes are defined in ``tests/integration/conftest.py`` +
``tests/integration/coordinator_fixtures.py`` and only resolve for tests at or
below ``tests/integration/`` (pytest walks UP the tree, never sideways). Placed
in the application-services tree the recovery pass errored with
``fixture 'async_session_factory' not found`` at setup.

TRANSACTION BOUNDARY (B3 "deferred to C8" note): ``DbCostRollupService`` opens
its OWN sessions via the injected ``async_session_factory`` and COMMITs. The
``db_session`` fixture's ``begin()...rollback()`` isolation would therefore hide
the seed rows from the service's independent session (PostgreSQL read-committed).
So these tests seed through ``async_session_factory`` with EXPLICIT commits and
rely on the ``coordinator_truncation`` fixture (TRUNCATE before+after) for
cleanup — mirroring ``test_coordinator_fixture_harness_smoke.py::
test_truncation_clears_coordinator_tables``. ``fresh_test_user`` supplies the
committed FK parent for ``sessions.user_id`` / ``cost_records.user_id``.

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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.coordinator_recovery,
    pytest.mark.anyio,
]


async def _seed_parent(
    async_session_factory,
    *,
    user_id: str,
) -> str:
    """Insert + COMMIT a parent SessionModel row; return parent_session_id.

    Seeds via ``async_session_factory`` (its own session + commit) so the rows
    are visible to ``DbCostRollupService``'s independent session — the
    ``db_session`` rollback-isolation fixture would hide them (see module
    docstring). ``sessions`` NOT NULL columns carry server_defaults, so a raw
    INSERT of id/user_id/created_at/updated_at suffices.
    """
    parent_sid = f"sess-b3-parent-{_uuid.uuid4().hex[:12]}"
    async with async_session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO sessions(id, user_id, status, title, "
                "created_at, updated_at) "
                "VALUES (:sid, :uid, 'pending', 'parent', NOW(), NOW())"
            ),
            {"sid": parent_sid, "uid": user_id},
        )
        await s.commit()
    return parent_sid


async def _seed_child_with_cost(
    async_session_factory,
    *,
    user_id: str,
    parent_session_id: str,
    coordinator_run_id: str,
    total_usd: Decimal,
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> str:
    """Insert + COMMIT a child SessionModel + one CostRecordModel; return id.

    Committed via ``async_session_factory`` for the same visibility reason as
    ``_seed_parent``.
    """
    sid = f"sess-b3-{_uuid.uuid4().hex[:12]}"
    async with async_session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO sessions(id, user_id, parent_session_id, worker_type, "
                "coordinator_run_id, tool_filter_preset, status, title, "
                "created_at, updated_at) "
                "VALUES (:sid, :uid, :parent, 'subagent', :run_id, 'coordinator_step', "
                "'pending', 'b3 child', "
                "NOW(), NOW())"
            ),
            {
                "sid": sid,
                "uid": user_id,
                "parent": parent_session_id,
                "run_id": coordinator_run_id,
            },
        )
        await s.execute(
            text(
                "INSERT INTO cost_records("
                "  id, session_id, user_id, run_id, node_name, step_ix, "
                "  attempt_ix, model, provider, input_tokens, output_tokens, "
                "  cache_read_tokens, cache_write_tokens, reasoning_tokens, "
                "  total_usd, pricing_version, cost_status"
                ") VALUES ("
                "  :id, :sid, :uid, :run_id, 'executor_node', 0, 0, 'gpt-4', "
                "  'openai_official', :input_tokens, :output_tokens, 0, 0, 0, "
                "  :total_usd, 'v1', 'actual'"
                ")"
            ),
            {
                "id": str(_uuid.uuid4()),
                "sid": sid,
                "uid": user_id,
                "run_id": f"run-{_uuid.uuid4().hex[:12]}",
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_usd": total_usd,
            },
        )
        await s.commit()
    return sid


async def _seed_empty_child(
    async_session_factory,
    *,
    user_id: str,
    parent_session_id: str,
    coordinator_run_id: str,
) -> str:
    """Insert + COMMIT a child SessionModel with NO cost_records; return id."""
    sid = f"sess-b3-empty-{_uuid.uuid4().hex[:12]}"
    async with async_session_factory() as s:
        await s.execute(
            text(
                "INSERT INTO sessions(id, user_id, parent_session_id, worker_type, "
                "coordinator_run_id, tool_filter_preset, status, title, "
                "created_at, updated_at) "
                "VALUES (:sid, :uid, :parent, 'subagent', :run_id, 'coordinator_step', "
                "'pending', 'empty', "
                "NOW(), NOW())"
            ),
            {
                "sid": sid,
                "uid": user_id,
                "parent": parent_session_id,
                "run_id": coordinator_run_id,
            },
        )
        await s.commit()
    return sid


async def test_aggregate_sums_children_matching_coordinator_run(
    async_session_factory, fresh_test_user, coordinator_truncation,
):
    """Three children all tagged with the same coordinator_run_id contribute
    rows; aggregate SUMs the total_usd + token counts; missing_children empty.
    """
    uid = str(fresh_test_user.id)
    parent_sid = await _seed_parent(async_session_factory, user_id=uid)

    run_id = f"run-b3-{_uuid.uuid4().hex[:12]}"
    c1 = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.10"),
        input_tokens=100, output_tokens=50,
    )
    c2 = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.20"),
        input_tokens=200, output_tokens=100,
    )
    c3 = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.30"),
        input_tokens=300, output_tokens=150,
    )

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
    async_session_factory, fresh_test_user, coordinator_truncation,
):
    """INV-B3: a child whose sessions.coordinator_run_id != caller-provided
    run_id MUST be excluded from SUM and surface in missing_children.
    """
    uid = str(fresh_test_user.id)
    parent_sid = await _seed_parent(async_session_factory, user_id=uid)

    good_run = f"run-good-{_uuid.uuid4().hex[:12]}"
    bad_run = f"run-bad-{_uuid.uuid4().hex[:12]}"
    good_child = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=good_run, total_usd=Decimal("0.10"),
    )
    bad_child = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=bad_run, total_usd=Decimal("99.99"),
    )

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
    async_session_factory, fresh_test_user, coordinator_truncation,
):
    """A child that has a valid sessions row + matching coordinator_run_id but
    NO cost_records ledger rows MUST surface in missing_children with cost=0.
    """
    uid = str(fresh_test_user.id)
    parent_sid = await _seed_parent(async_session_factory, user_id=uid)

    run_id = f"run-b3-{_uuid.uuid4().hex[:12]}"
    child_with_cost = await _seed_child_with_cost(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id, total_usd=Decimal("0.10"),
    )
    empty_child_id = await _seed_empty_child(
        async_session_factory, user_id=uid, parent_session_id=parent_sid,
        coordinator_run_id=run_id,
    )

    svc = DbCostRollupService(async_session_factory=async_session_factory)
    result = await svc.aggregate(
        coordinator_run_id=run_id,
        child_session_ids=[child_with_cost, empty_child_id],
    )
    assert result.cost.total_usd == pytest.approx(0.10)
    assert empty_child_id in result.missing_children


async def test_rollup_to_parent_idempotent(
    async_session_factory, coordinator_truncation,
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
