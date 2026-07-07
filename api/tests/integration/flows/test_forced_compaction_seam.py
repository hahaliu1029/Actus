import pytest

from app.domain.services.execution_metrics import ExecutionMetrics

pytestmark = [pytest.mark.anyio, pytest.mark.integration]  # host DB; CI-only


async def test_forced_initial_compaction_runs_and_yields_events(
    planner_react_with_compactor, memory_at_85_percent
):
    """§12 test 11 (seam-level): a compactable Memory + force → real GradualCompactor
    Level-2 compaction, yields ContextStatus + Compaction events; metric counted;
    _last_compaction_result cleared before yield (no run-end double-emit)."""
    flow = planner_react_with_compactor
    flow._force_initial_compaction = True
    flow._execution_metrics = ExecutionMetrics()
    # Persistence note: memory_at_85_percent triggers level-2 → _check_overflow
    # persists via save_memory + record_compaction. Per the fixture's WARNING
    # (conftest.py:254-270) commit the seed user/session first (separate committed
    # UoW) OR set flow._uow_factory to a no-op UoW — use whichever the sibling B6
    # integration tests use.

    events = [ev async for ev in flow._run_forced_initial_compaction(memory_at_85_percent)]

    kinds = [type(e).__name__ for e in events]
    assert "ContextStatusEvent" in kinds
    assert "CompactionEvent" in kinds  # real Level-2 compaction happened at the seam
    assert flow._execution_metrics.compaction_count == 1
    assert flow._last_compaction_result is None  # cleared before yield
