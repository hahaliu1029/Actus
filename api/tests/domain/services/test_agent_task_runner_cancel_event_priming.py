"""PR-9b-A audit round-1 P1 (Fix 3 — INV-A6) — AgentTaskRunner primes the
planner's per-run ``_cancel_event`` so the 18-key coordinator cfg projection
in ``PlannerReActFlow._build_config()`` injects a real ``asyncio.Event``
(not ``None``) under INV-A6 ("each non-None").

A5 introduced ``PlannerReActFlow._cancel_event = None`` as a deferred
placeholder; no caller wrote it, so the cfg projection emitted
``cancel_event=None`` and broke the invariant downstream consumers
(runner_starter / orchestrator / PatchApplier) rely on. This test exercises
the priming helper directly via attribute-style construction (bypassing the
heavy AgentTaskRunner ctor) so the contract is locked without an integration
fixture.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner


def _make_runner_stub(
    *,
    coord_deps: object | None,
    flow: object | None,
) -> AgentTaskRunner:
    """Construct a minimally-initialized AgentTaskRunner that just carries the
    two attributes the priming helper inspects. Avoids the full ctor's
    dependency graph (LLM / sandbox / DB / Redis / supervisor / ...).
    """
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._coord_deps_for_planner = coord_deps  # type: ignore[attr-defined]
    runner._flow = flow  # type: ignore[attr-defined]
    return runner


def test_priming_writes_fresh_asyncio_event_when_coord_deps_real() -> None:
    """INV-A6 — when coord_deps is non-None, the helper MUST write a fresh
    ``asyncio.Event`` to ``self._flow._cancel_event`` so the cfg projection
    sees a real event instance (not ``None``).
    """
    flow = SimpleNamespace(_cancel_event=None)
    runner = _make_runner_stub(coord_deps=MagicMock(name="real_coord_deps"), flow=flow)

    runner._prime_planner_cancel_event_for_coord_deps()

    assert isinstance(flow._cancel_event, asyncio.Event), (
        "priming must allocate an asyncio.Event when coord_deps is real; "
        f"got {flow._cancel_event!r}"
    )
    assert not flow._cancel_event.is_set(), (
        "freshly-primed event must start unset so the first cancel "
        "signal is observable"
    )


def test_priming_allocates_a_new_event_per_call() -> None:
    """Each ``invoke()`` enters this helper; the helper MUST allocate a fresh
    Event so a previously-cancelled / previously-set Event from an earlier
    invoke cannot leak forward and short-circuit the new run.
    """
    flow = SimpleNamespace(_cancel_event=None)
    runner = _make_runner_stub(coord_deps=MagicMock(), flow=flow)

    runner._prime_planner_cancel_event_for_coord_deps()
    first = flow._cancel_event
    first.set()  # simulate cancellation during the previous invoke

    runner._prime_planner_cancel_event_for_coord_deps()
    second = flow._cancel_event

    assert isinstance(second, asyncio.Event)
    assert second is not first, "must allocate a fresh Event per invoke"
    assert not second.is_set(), (
        "freshly-allocated Event must start unset even if a previous "
        "instance was set / cancelled"
    )


def test_priming_is_noop_when_coord_deps_is_none() -> None:
    """Legacy / non-coordinator path: ``coord_deps_for_planner is None`` means
    ``PlannerReActFlow._build_config()`` skips the 18 coordinator cfg keys
    wholesale (via the ``_NullCoordinatorRuntimeDeps`` isinstance guard).
    Writing a cancel_event would be wasted allocation; the helper MUST no-op.
    """
    flow = SimpleNamespace(_cancel_event=None)
    runner = _make_runner_stub(coord_deps=None, flow=flow)

    runner._prime_planner_cancel_event_for_coord_deps()

    assert flow._cancel_event is None, (
        "priming must no-op when coord_deps_for_planner is None"
    )


def test_priming_handles_mock_flow_without_cancel_event_attr() -> None:
    """Test mocks sometimes swap ``self._flow`` for a bare object without a
    ``_cancel_event`` attribute. The priming helper is best-effort wiring;
    it MUST silently no-op rather than raise ``AttributeError`` and break
    the test fixture.
    """
    flow = object()  # no _cancel_event
    runner = _make_runner_stub(coord_deps=MagicMock(), flow=flow)

    # Must not raise.
    runner._prime_planner_cancel_event_for_coord_deps()


def test_priming_handles_missing_flow_attr() -> None:
    """``__new__``-bypass construction without ``_flow`` set at all — the
    helper MUST no-op rather than crash. Mirrors the defensive pattern used
    by ``_maybe_spawn_mailbox_supervisor`` (getattr-guarded reads).
    """
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._coord_deps_for_planner = MagicMock()  # type: ignore[attr-defined]

    # Must not raise.
    runner._prime_planner_cancel_event_for_coord_deps()


def test_primed_event_flows_through_build_config_into_cfg() -> None:
    """End-to-end contract — after priming, ``PlannerReActFlow._build_config()``
    projects the SAME Event instance into ``cfg['configurable']['cancel_event']``.

    This pins the threading: ``AgentTaskRunner`` writer -> planner attribute ->
    cfg key -> dispatch_node -> orchestrator.run(cancel_event=...) /
    worker_node waiter / PatchApplier.apply(cancel_event=...). NOTE (C2b
    budget D9): child runners do NOT share this event — dispatch creates a
    fresh per-child event for each runner_starter.start; parent cancel
    reaches children via the orchestrator's CANCEL_REQUEST envelope fan-out.
    """
    from app.application.services.coordinator_runtime_deps import (
        _CoordinatorRuntimeDeps,
    )
    from app.domain.models.app_config import AgentConfig
    from app.domain.services.flows.planner_react import PlannerReActFlow

    # Build a real planner with a real (sentinel-filled) coord_deps so the
    # cfg projection enters the non-null branch.
    field_names = (
        "parallel_execution_subgraph", "session_service", "rehydrate_service",
        "child_runner_starter", "mailbox_publisher", "mailbox_subscriber",
        "envelope_factory", "orchestrator_factory", "terminal_waiter",
        "probe_quota", "coordinator_limits", "session_repository",
        "patch_reducer_service", "patch_applier_deps", "artifact_storage",
        "cost_rollup_service", "coordinator_envelope_store",
        "parent_sandbox_adapter_factory",
    )
    sentinels = {f: MagicMock(name=f) for f in field_names}
    # [finish-core §5.2 G2] _build_config() invokes this factory to wrap the
    # raw parent SandboxHandle into a ParentSandboxPort, so the dep MUST be
    # callable. A bare MagicMock is callable; an explicit side_effect keeps the
    # output distinct from the raw handle (matches production semantics).
    sentinels["parent_sandbox_adapter_factory"] = MagicMock(
        name="parent_sandbox_adapter_factory",
        side_effect=lambda h: MagicMock(name="wrapped_parent_sandbox"),
    )
    coord_deps = _CoordinatorRuntimeDeps(**sentinels)

    flow = PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(
            max_iterations=10, max_retries=3, max_search_results=5,
        ),
        session_id="test-session",
        browser=MagicMock(),
        sandbox=MagicMock(name="parent_sandbox"),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        _coord_deps=coord_deps,
    )

    # Pre-priming: cancel_event is None — the projection would emit None.
    assert flow._cancel_event is None

    # Drive the priming helper via the runner's writer.
    runner = _make_runner_stub(coord_deps=coord_deps, flow=flow)
    runner._prime_planner_cancel_event_for_coord_deps()

    # The cfg projection now sees the primed Event (same instance — not a
    # copy, not None).
    cfg = flow._build_config()
    projected = cfg["configurable"]["cancel_event"]
    assert isinstance(projected, asyncio.Event)
    assert projected is flow._cancel_event, (
        "cfg['cancel_event'] must be the SAME Event instance the runner "
        "primed onto flow._cancel_event — the PARENT-RUN consumers "
        "(orchestrator cancel-watch / worker_node waiter / PatchApplier) "
        "observe this object identity; children get their own per-child "
        "events from dispatch (C2b budget D9)"
    )


@pytest.mark.parametrize("none_field", ["_coord_deps_for_planner"])
def test_priming_threshold_matches_cfg_projection_skip_branch(none_field: str) -> None:
    """Symmetry check: the priming branch is gated on the SAME predicate
    that ``PlannerReActFlow._build_config()`` uses to skip the 18 coordinator
    cfg keys. When ``coord_deps_for_planner is None``, the planner's cfg
    projection skips ``cancel_event`` injection entirely, so priming is a
    no-op (don't allocate, don't surprise the test fixture).
    """
    flow = SimpleNamespace(_cancel_event=None)
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    setattr(runner, none_field, None)
    runner._flow = flow  # type: ignore[attr-defined]

    runner._prime_planner_cancel_event_for_coord_deps()
    assert flow._cancel_event is None
