"""[S4 §5 / Task 3.1] Coordinator DI — ``team_repository`` + ``skill_repository``
are tail-defaulted ``object = None`` fields on ``_CoordinatorRuntimeDeps`` and
matching ``@property`` stubs on ``_NullCoordinatorRuntimeDeps``.

Dormant DI plumbing: NOTHING consumes them in this task. They are projected into
``cfg["configurable"]`` by ``_build_config`` (separately covered by
``test_planner_react_coord_config.py``) and constructed at the composition root
(covered by ``test_coordinator_composition_root_wiring.py``).
"""
from app.application.services.coordinator_runtime_deps import (
    _CoordinatorRuntimeDeps,
    _NullCoordinatorRuntimeDeps,
)


def _min_deps(**kw):
    # 18 required positional + 2 pre-existing tail defaults; construct with
    # sentinels and assert the 2 NEW tail defaults exist + default None.
    fields = [
        "parallel_execution_subgraph", "session_service", "rehydrate_service",
        "child_runner_starter", "mailbox_publisher", "mailbox_subscriber",
        "envelope_factory", "orchestrator_factory", "terminal_waiter",
        "probe_quota", "coordinator_limits", "session_repository",
        "patch_reducer_service", "patch_applier_deps", "artifact_storage",
        "cost_rollup_service", "coordinator_envelope_store",
        "parent_sandbox_adapter_factory",
    ]
    base = {f: object() for f in fields}
    base.update(kw)
    return _CoordinatorRuntimeDeps(**base)


def test_team_and_skill_repository_default_none():
    d = _min_deps()
    assert d.team_repository is None
    assert d.skill_repository is None


def test_team_and_skill_repository_settable():
    tr, sr = object(), object()
    d = _min_deps(team_repository=tr, skill_repository=sr)
    assert d.team_repository is tr
    assert d.skill_repository is sr


def test_null_deps_expose_both_properties():
    n = _NullCoordinatorRuntimeDeps()
    assert n.team_repository is None
    assert n.skill_repository is None
