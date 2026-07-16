"""PR-9b-A INV-A10 — _NullCoordinatorRuntimeDeps must trigger ZERO side-effect
real-ctor calls AND _build_config must SKIP all coordinator config keys.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.application.services.coordinator_runtime_deps import (
    _NullCoordinatorRuntimeDeps,
)
from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig


COORD_KEYS = {
    "parallel_execution_subgraph", "session_service", "rehydrate_service",
    "child_runner_starter", "mailbox_publisher", "mailbox_subscriber",
    "envelope_factory", "orchestrator_factory", "terminal_waiter",
    "probe_quota", "coordinator_limits", "session_repository",
    "cancel_event", "patch_reducer_service", "patch_applier_deps",
    "parent_sandbox", "artifact_storage", "cost_rollup_service",
    "coordinator_wait_guard_factory",
    "coordinator_liveness_service",
}


def _make_default_flow():
    from app.domain.services.flows.planner_react import PlannerReActFlow

    return PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(
            max_iterations=10, max_retries=3, max_search_results=5,
        ),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
    )


def test_null_deps_zero_side_effects_on_flow_construction():
    """INV-A10 — patching Redis, SQLAlchemy, Docker, and ALL lifespan-owned
    coordinator-builder symbols with side_effect=AssertionError; construct
    PlannerReActFlow with default (_NullCoordinatorRuntimeDeps); assert no
    AssertionError was raised by any patched ctor.
    """
    # Patch targets verified at A5 impl time (Step 0). All seven exist.
    targets = [
        ("redis.asyncio", "from_url"),
        ("sqlalchemy.ext.asyncio", "create_async_engine"),
        ("docker", "DockerClient"),
        # Lifespan-owned coordinator builders:
        ("app.infrastructure.repositories.db_coordinator_result_envelope_store_repository",
         "DbCoordinatorResultEnvelopeStoreRepository.__init__"),
        # [B3] A3's MetricHookCostRollupService was superseded by
        # DbCostRollupService (pull-cost authority + push-only hook).
        ("app.application.services.db_cost_rollup_service",
         "DbCostRollupService.__init__"),
        ("app.application.services.coordinator_child_runner_starter",
         "DefaultCoordinatorChildRunnerStarter.__init__"),
        ("app.application.services.coordinator_envelope_factory",
         "CoordinatorEnvelopeFactory.__init__"),
    ]
    sentinels = []
    for mod, attr in targets:
        s = patch(f"{mod}.{attr}",
                  side_effect=AssertionError(f"{mod}.{attr} called"))
        s.__enter__()
        sentinels.append(s)
    try:
        flow = _make_default_flow()
        assert isinstance(flow._coord_deps, _NullCoordinatorRuntimeDeps)
    finally:
        for s in reversed(sentinels):
            s.__exit__(None, None, None)


def test_build_config_omits_coord_keys_when_null_deps():
    flow = _make_default_flow()
    cfg = flow._build_config()
    cfg_keys = set(cfg["configurable"].keys())
    intersection = COORD_KEYS & cfg_keys
    assert intersection == set(), (
        f"null deps must SKIP coord keys; found injected: {intersection}"
    )


def test_ordinary_null_deps_flow_still_creates_watchdog():
    """Null deps alone does not mean coordinator child (legacy/subagent path)."""
    from app.domain.services.execution_watchdog import ExecutionWatchdog

    flow = _make_default_flow()
    cfg = flow._build_config()

    assert isinstance(cfg["configurable"]["execution_watchdog"], ExecutionWatchdog)


def test_coordinator_child_permission_context_disables_graph_watchdog():
    """The child-scope context is the production coordinator-child marker."""
    from app.domain.services.execution_watchdog import ExecutionControl

    flow = _make_default_flow()
    flow.set_child_permission_context(MagicMock(name="child_permission_context"))
    cfg = flow._build_config()

    assert cfg["configurable"]["execution_watchdog"] is None
    assert isinstance(cfg["configurable"]["execution_control"], ExecutionControl)


def test_mailbox_managed_research_child_disables_graph_watchdog():
    """Mailbox heartbeat/orphan ownership replaces the graph idle timer."""
    flow = _make_default_flow()
    flow.set_mailbox_liveness_managed()

    cfg = flow._build_config()

    assert cfg["configurable"]["execution_watchdog"] is None
