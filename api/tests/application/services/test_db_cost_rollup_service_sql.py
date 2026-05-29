"""[PR-9b-B Task B3] Unit SQL-shape tests for DbCostRollupService.

Synchronous introspection of the SQL the aggregate(...) method builds. These
tests do NOT require Postgres — they exercise the SQL-shape contract:

INV-B3 SQL invariants (plan §B3 spec):
  - Query joins cost_records with sessions on session_id == sessions.id.
  - Filter clauses include sessions.coordinator_run_id == :run_id and
    cost_records.session_id ∈ child_session_ids.
  - Aggregates use SUM(...) over token + total_usd columns; result
    grouped/joined per child so missing children surface zero rows.
  - The query relies on the cost_records.total_usd column name (NOT
    cost_usd) — INV-B1 hard-locks that.

Integration coverage at test_db_cost_rollup_service.py (4 anyio tests)
verifies actual SUM behaviour against real Postgres.
"""
from __future__ import annotations

import inspect
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.application.services import db_cost_rollup_service as _mod
from app.application.services.db_cost_rollup_service import DbCostRollupService


def test_aggregate_query_text_joins_cost_records_with_sessions():
    """The aggregate query MUST join cost_records to sessions so that
    sessions.coordinator_run_id can validate child_session_ids belong to
    the named run. A child whose row's session_id has a mismatched
    sessions.coordinator_run_id MUST be excluded from the SUM and
    surface in missing_children.
    """
    sql = DbCostRollupService.AGGREGATE_SQL
    sql_lower = sql.lower()
    assert "cost_records" in sql_lower
    assert "sessions" in sql_lower
    # Join predicate references both sides of the relationship.
    assert "session_id" in sql_lower
    # coordinator_run_id is the validation filter, not just a select column.
    assert "coordinator_run_id" in sql_lower


def test_aggregate_query_filters_by_coordinator_run_id_and_session_ids():
    """INV-B3: filter clauses must include BOTH
       sessions.coordinator_run_id == :coordinator_run_id AND
       cost_records.session_id ∈ :child_session_ids
       so mismatched-run rows are excluded.
    """
    sql = DbCostRollupService.AGGREGATE_SQL
    sql_lower = sql.lower()
    assert ":coordinator_run_id" in sql_lower
    # child_session_ids — either ANY(:child_session_ids) (Postgres array) or
    # an expanded IN (...) form. Both shapes bind the same parameter name.
    assert ":child_session_ids" in sql_lower


def test_aggregate_query_uses_total_usd_column_not_cost_usd():
    """INV-B1 hard-lock: the cost_records ledger uses total_usd (Numeric).
    A regression that renames to cost_usd here would silently SUM zero
    on every call until somebody reads the wire.
    """
    sql = DbCostRollupService.AGGREGATE_SQL
    sql_lower = sql.lower()
    assert "total_usd" in sql_lower
    assert "cost_usd" not in sql_lower


def test_aggregate_query_sums_required_cost_components():
    """The Protocol returns CostAggregate(total_input_tokens,
    total_output_tokens, total_usd, tool_call_count). The query must SUM
    each of those columns (tool_call_count may be derived as COUNT(*) if
    cost_records lacks a column for it).
    """
    sql = DbCostRollupService.AGGREGATE_SQL
    sql_lower = sql.lower()
    assert "sum(" in sql_lower
    assert "input_tokens" in sql_lower
    assert "output_tokens" in sql_lower
    assert "total_usd" in sql_lower


def test_aggregate_groups_by_session_id_so_missing_children_visible():
    """The reducer reads ``missing_children`` from AggregateResult to
    construct ``diagnostics_summary='cost_unavailable: <ids>'`` per
    INV-B3. To surface zero-row children the query must GROUP BY
    session_id (so each child gets its own row, including zero-row
    children via the application-side set diff).
    """
    sql = DbCostRollupService.AGGREGATE_SQL
    sql_lower = sql.lower()
    assert "group by" in sql_lower
    assert "session_id" in sql_lower


def test_aggregate_returns_cost_aggregate_with_float_total_usd():
    """Code-review fix-1: explicit ``float(...)`` coercion at the
    CostAggregate boundary.

    ``CostAggregate.total_usd`` is typed as ``float``
    (mailbox_envelope.py:167) but the SUM(cost_records.total_usd) we read
    out of Postgres is ``Decimal`` (cost_record_orm.py:112 declares
    ``Numeric(28, 10)``). Without an explicit ``float(...)`` the wire
    boundary leans on Pydantic v2's implicit Decimal → float coercion —
    a Pydantic v3+ behavior change would silently break the contract.

    Source-introspection guard so a future refactor that drops the
    coercion produces a load-bearing test failure rather than a runtime
    regression discovered on the wire.
    """
    src = inspect.getsource(DbCostRollupService.aggregate)
    assert "float(" in src, (
        "DbCostRollupService.aggregate must explicitly call float(...) at "
        "the CostAggregate(total_usd=...) boundary; do NOT rely on Pydantic "
        "v2's implicit Decimal → float coercion."
    )


# ── rollup_to_parent idempotency + bounded-LRU eviction (unit, NO Postgres) ──
#
# rollup_to_parent only touches self._seen_keys + self._metric_sink.record, so
# it is exercised with a fake sink and a MagicMock session factory (the factory
# is never called on this path). These tests live in the UNIT file (not the
# Postgres-backed integration file) precisely because no DB is needed.
#
# anyio (NOT asyncio) is the project's async test runtime; the anyio_backend
# fixture is provided session-wide by tests/conftest.py. The module marker only
# dispatches the async tests below — the synchronous SQL-shape tests above are
# unaffected.
pytestmark = pytest.mark.anyio


class _RecordingSink:
    """Fake metric sink that records every record(...) call's kwargs."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def record(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


def _make_service(metric_sink):
    # The session factory is unused by rollup_to_parent — a MagicMock proves
    # the push path never reaches Postgres.
    return DbCostRollupService(
        async_session_factory=MagicMock(),
        metric_sink=metric_sink,
    )


async def _roll(svc, key: str) -> None:
    await svc.rollup_to_parent(
        idempotency_key=key,
        parent_session_id="parent-1",
        cost={
            "total_input_tokens": 10,
            "total_output_tokens": 5,
            "total_usd": Decimal("0.01"),
            "tool_call_count": 1,
        },
        source="result_ready",
    )


async def test_rollup_to_parent_idempotent_within_cap():
    """(a) Idempotency-within-cap: the SAME idempotency_key submitted twice
    fires the metric sink exactly once. This is the INV-A9 dedup contract a
    supervisor PEL redelivery relies on.
    """
    sink = _RecordingSink()
    svc = _make_service(sink)

    await _roll(svc, "dupe-key")
    await _roll(svc, "dupe-key")

    assert len(sink.calls) == 1


async def test_rollup_to_parent_evicts_oldest_when_cap_exceeded(monkeypatch):
    """(b) Eviction: with the cap shrunk to 2, inserting 3 distinct keys evicts
    the OLDEST (k1). Re-submitting k1 fires the sink AGAIN (proving eviction),
    while a still-cached key (k3) stays deduped. The cache never exceeds the cap.

    Against the OLD unbounded ``set[str]`` this fails: the set never evicts, so
    the re-submission of k1 is a dedup no-op and the sink is NOT re-fired.
    """
    monkeypatch.setattr(_mod, "_MAX_SEEN_KEYS", 2)
    sink = _RecordingSink()
    svc = _make_service(sink)

    # k1 evicted once k3 lands (cap=2 holds the 2 most-recent: k2, k3).
    await _roll(svc, "k1")
    await _roll(svc, "k2")
    await _roll(svc, "k3")
    assert len(sink.calls) == 3
    assert len(svc._seen_keys) <= 2

    # k1 was evicted → re-submission is a MISS → sink fires again (4th call).
    await _roll(svc, "k1")
    assert len(sink.calls) == 4
    assert len(svc._seen_keys) <= 2

    # k3 is still cached → re-submission is a HIT → no new sink call.
    await _roll(svc, "k3")
    assert len(sink.calls) == 4
    assert len(svc._seen_keys) <= 2


async def test_rollup_to_parent_dedup_hit_refreshes_lru(monkeypatch):
    """[PR-9b-B codex F4 — LOW] True-LRU refresh on a dedup HIT.

    A dedup HIT must ``move_to_end`` the re-seen key so eviction is
    recency-based, NOT FIFO-by-first-insertion. Scenario (cap=2):

      1. insert k1, k2  → cache = [k1, k2] (k1 oldest)
      2. HIT k1         → refresh: cache = [k2, k1] (k2 now oldest)
      3. insert k3      → evicts the OLDEST = k2; k1 SURVIVES

    Proof: after step 3, re-submitting k1 is a HIT (no new sink call) while
    re-submitting k2 is a MISS (sink fires). Against the OLD FIFO behavior
    (no move_to_end) k1 would have been evicted at step 3 and k2 retained,
    flipping both assertions.

    Also confirms the FIFO-cap-size invariant is preserved (cache never
    exceeds the cap).
    """
    monkeypatch.setattr(_mod, "_MAX_SEEN_KEYS", 2)
    sink = _RecordingSink()
    svc = _make_service(sink)

    await _roll(svc, "k1")          # miss → sink fires (1)
    await _roll(svc, "k2")          # miss → sink fires (2)
    assert len(sink.calls) == 2
    assert len(svc._seen_keys) <= 2

    # HIT k1 → dedup no-op for the sink BUT refreshes k1 to the LRU tail.
    await _roll(svc, "k1")
    assert len(sink.calls) == 2     # still a HIT — no new sink call
    assert len(svc._seen_keys) <= 2

    # Insert k3 → evicts the OLDEST, which is now k2 (k1 was refreshed).
    await _roll(svc, "k3")          # miss → sink fires (3)
    assert len(sink.calls) == 3
    assert len(svc._seen_keys) <= 2

    # k1 SURVIVED the eviction (LRU refresh worked) → re-submit is a HIT.
    await _roll(svc, "k1")
    assert len(sink.calls) == 3, (
        "k1 must survive eviction after its LRU refresh — a dedup HIT must "
        "move_to_end, not leave the key at its original FIFO position."
    )

    # k2 was the one evicted → re-submit is a MISS → sink fires again.
    await _roll(svc, "k2")
    assert len(sink.calls) == 4
    assert len(svc._seen_keys) <= 2
