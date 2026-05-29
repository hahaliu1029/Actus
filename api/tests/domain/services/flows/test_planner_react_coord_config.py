"""PR-9b-A INV-A6 — _build_config produces all 18 coordinator cfg keys
when _coord_deps is non-null, each non-None.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.application.services.coordinator_runtime_deps import (
    _CoordinatorRuntimeDeps,
)
from app.domain.models.app_config import AgentConfig


EXPECTED_KEYS = {
    "parallel_execution_subgraph",
    "session_service",
    "rehydrate_service",
    "child_runner_starter",
    "mailbox_publisher",
    "mailbox_subscriber",
    "envelope_factory",
    "orchestrator_factory",
    "terminal_waiter",
    "probe_quota",
    "coordinator_limits",
    "session_repository",
    "cancel_event",
    "patch_reducer_service",
    "patch_applier_deps",
    "parent_sandbox",
    "artifact_storage",
    "cost_rollup_service",
}


def _build_flow_with_real_coord_deps():
    """Construct PlannerReActFlow with sentinel _coord_deps + minimal stubs.

    The intent is to expose _build_config() output, NOT to exercise planner.
    """
    from app.domain.services.flows.planner_react import PlannerReActFlow

    sentinels = {f: MagicMock(name=f) for f in (
        "parallel_execution_subgraph", "session_service", "rehydrate_service",
        "child_runner_starter", "mailbox_publisher", "mailbox_subscriber",
        "envelope_factory", "orchestrator_factory", "terminal_waiter",
        "probe_quota", "coordinator_limits", "session_repository",
        "patch_reducer_service", "patch_applier_deps", "artifact_storage",
        "cost_rollup_service", "coordinator_envelope_store",
    )}
    coord_deps = _CoordinatorRuntimeDeps(**sentinels)

    flow = PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(
            max_iterations=10, max_retries=3, max_search_results=5,
        ),
        session_id="test-session",
        browser=MagicMock(),
        sandbox=MagicMock(),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        _coord_deps=coord_deps,
    )
    # Per-run attrs are set on the instance after construction (matches the
    # production wire: AgentTaskRunner / main_graph._run_parallel_backend
    # populate these per invoke; here we exercise the cfg-building contract).
    flow._sandbox = MagicMock(name="parent_sandbox")
    flow._cancel_event = MagicMock(name="cancel_event")
    return flow, sentinels


def test_build_config_contains_all_18_coord_keys():
    flow, sentinels = _build_flow_with_real_coord_deps()
    cfg = flow._build_config()
    configurable = cfg["configurable"]
    for key in EXPECTED_KEYS:
        assert key in configurable, f"missing key: {key}"
        assert configurable[key] is not None, f"None for key: {key}"


def test_build_config_does_not_inject_event_queue():
    """event_queue is merged by GraphEventBridge at invocation time, NOT here."""
    flow, _ = _build_flow_with_real_coord_deps()
    cfg = flow._build_config()
    assert "event_queue" not in cfg["configurable"]


def test_build_config_threads_per_run_objects():
    """cancel_event and parent_sandbox come from per-run flow attrs, not _coord_deps."""
    flow, _ = _build_flow_with_real_coord_deps()
    cfg = flow._build_config()
    assert cfg["configurable"]["cancel_event"] is flow._cancel_event
    assert cfg["configurable"]["parent_sandbox"] is flow._sandbox
