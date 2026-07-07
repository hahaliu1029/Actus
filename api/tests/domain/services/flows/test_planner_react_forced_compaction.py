"""B11 §8: forced initial compaction seam (unit, bypass-init).

Two local (no LLM / DB / graph) facets of the seam are unit-tested in this file:
(1) event *production* — the extracted helper in isolation via object.__new__ +
a stubbed _check_overflow (tests below); (2) the *ordering* invariant — a
source-structure assertion that the seam call precedes planner detection in
invoke() (test_forced_compaction_seam_precedes_planner_detection_in_invoke —
run by the same Steps 2/4 that run this file). The bypass-init fixtures cannot
drive a full invoke() (no _main_graph /
_memory_config), so SPEC §12 test 11's runtime yield-ordering is locked
structurally instead — see Documented Deviation #7. Together (produces the
events) ∧ (seam runs before the planner) ⟹ forced ContextStatus/Compaction
events reach the flow generator before the first planner event.
"""
from __future__ import annotations

import pytest

from app.domain.services.execution_metrics import ExecutionMetrics
from app.domain.services.flows.planner_react import PlannerReActFlow
from app.domain.services.graphs.compaction import CompactionResult


def _bare_flow() -> PlannerReActFlow:
    flow = object.__new__(PlannerReActFlow)
    flow._overflow_config = None  # deterministic build_compaction_events (window=0)
    flow._execution_metrics = ExecutionMetrics()
    flow._last_compaction_result = None
    return flow


@pytest.mark.anyio
async def test_forced_compaction_yields_events_counts_metric_clears_result():
    flow = _bare_flow()
    known = CompactionResult(
        messages=(),
        level_applied=2,
        tokens_before=1000,
        tokens_after=600,
        summary_injected=True,
        messages_removed=5,
        usage_ratio_after=0.5,
        compaction_id="c-1",
    )

    async def fake_check(memory, *, force):
        assert force is True
        flow._last_compaction_result = known  # mimic real _check_overflow (:1122)
        return known

    flow._check_overflow = fake_check  # type: ignore[assignment]

    events = [ev async for ev in flow._run_forced_initial_compaction(None)]

    assert [type(e).__name__ for e in events] == ["ContextStatusEvent", "CompactionEvent"]
    assert events[1].compaction_id == "c-1"
    assert events[1].level == 2
    assert flow._execution_metrics.compaction_count == 1
    # Cleared → run-end second _check_overflow won't double-emit via runner helper.
    assert flow._last_compaction_result is None


@pytest.mark.anyio
async def test_forced_compaction_noop_when_check_returns_none():
    flow = _bare_flow()

    async def fake_check(memory, *, force):
        return None  # overflow guard disabled / nothing to compact

    flow._check_overflow = fake_check  # type: ignore[assignment]

    events = [ev async for ev in flow._run_forced_initial_compaction(None)]
    assert events == []
    assert flow._execution_metrics.compaction_count == 0


def test_forced_compaction_seam_precedes_planner_detection_in_invoke():
    """§12 test 11 (ordering half): the forced-compaction seam must run before
    the first planner event is emitted in PlannerReActFlow.invoke, so forced
    ContextStatus/Compaction events reach the flow generator ahead of any planner
    event. The bypass-init fixtures can't drive a full invoke() (no _main_graph /
    _memory_config), so we lock the ordering at the source level: the seam call
    precedes planner detection in invoke()'s source. invoke() is a straight-line
    async generator between the two (no planner event is yielded in between), so
    source order == yield order (Documented Deviation #7). Anchor
    `_run_planner_for_detection` is invoke()'s earliest planner-emitting call;
    seam-before-it ⟹ seam-before-every-planner-path (the
    main-graph astream path is later in source). This is synchronous source
    inspection — no anyio, DB, or graph needed, so it runs locally.
    """
    import inspect

    from app.domain.services.flows.planner_react import PlannerReActFlow

    src = inspect.getsource(PlannerReActFlow.invoke)
    assert "_run_forced_initial_compaction(" in src, (
        "forced-compaction seam not yet wired into invoke() (Step 3d)"
    )
    seam = src.index("_run_forced_initial_compaction(")
    planner = src.index("_run_planner_for_detection(")
    assert seam < planner, (
        "seam must precede planner detection in invoke() so forced compaction "
        "events reach the flow generator before the first planner event"
    )
