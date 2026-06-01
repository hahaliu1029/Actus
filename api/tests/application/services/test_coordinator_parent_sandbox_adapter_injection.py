"""[finish-core §5.2 G2 / F3.1] parent_sandbox adapter-factory injection.

[INV-F2.1] ``_build_config``'s coord branch must set
``cfg["configurable"]["parent_sandbox"]`` to the OUTPUT of the injected
adapter factory (a ``ParentSandboxPort``), NOT the raw ``SandboxHandle``.
This keeps the domain flow (``planner_react.py``) free of any
``app.infrastructure`` import for the wrapping — the factory is injected as
a coordinator runtime dep instead.

[INV-F2.2] regression guard: ``_build_config`` (and the parent_sandbox
wiring it owns) must not introduce a NEW ``app.infrastructure`` import into
the domain flow. (A pre-existing B2-recovery import at planner_react.py:574
is a separate, documented Clean-Architecture exception — see test below.)
"""
from __future__ import annotations


def test_build_config_wraps_sandbox_in_adapter_port_not_raw():
    """[INV-F2.1] cfg['parent_sandbox'] is the adapter-factory output (a
    ParentSandboxPort), NOT the raw SandboxHandle. Build a REAL flow (so
    _build_config runs), then swap in a coord_deps carrying a factory mock."""
    from unittest.mock import MagicMock
    from app.application.services.coordinator_runtime_deps import _CoordinatorRuntimeDeps
    from tests.domain.services.flows.test_planner_react_coord_config import (
        _build_flow_with_real_coord_deps,
    )
    wrapped = MagicMock(name="adapter_port")
    factory = MagicMock(return_value=wrapped)
    s = object()
    coord = _CoordinatorRuntimeDeps(
        parallel_execution_subgraph=s, session_service=s, rehydrate_service=s,
        child_runner_starter=s, mailbox_publisher=s, mailbox_subscriber=s,
        envelope_factory=s, orchestrator_factory=s, terminal_waiter=s, probe_quota=s,
        coordinator_limits=s, session_repository=s, patch_reducer_service=s,
        patch_applier_deps=s, artifact_storage=s, cost_rollup_service=s,
        coordinator_envelope_store=s, parent_sandbox_adapter_factory=factory,
    )
    flow, _ = _build_flow_with_real_coord_deps()  # real flow → _build_config runs
    flow._coord_deps = coord
    cfg = flow._build_config()
    assert cfg["configurable"]["parent_sandbox"] is wrapped
    factory.assert_called_once_with(flow._sandbox)


def test_planner_react_no_new_infrastructure_import_in_build_config():
    """[INV-F2.2] domain flow imports nothing NEW from app.infrastructure as a
    result of this task.

    NOTE: planner_react.py ALREADY contains ONE pre-existing
    ``app.infrastructure`` import at line ~574 (``wrap_with_recovery``, the
    B2 recovery wrap, added 2026-04-27 — a documented Clean-Architecture
    exception predating this task). This guard asserts the count has NOT
    grown: the parent_sandbox adapter wrapping must go through the injected
    factory, never a direct import here. If this assert fails because the
    count grew, a new violation was introduced — fix it (inject, don't
    import). If the pre-existing exception is ever removed, tighten this to
    ``== 0``.
    """
    import pathlib
    import re

    src = pathlib.Path("app/domain/services/flows/planner_react.py").read_text()
    infra_imports = re.findall(
        r"^\s*(?:import\s+app\.infrastructure|from\s+app\.infrastructure)",
        src,
        flags=re.MULTILINE,
    )
    # Exactly the single pre-existing B2-recovery exception, nothing new.
    assert len(infra_imports) == 1, (
        f"expected exactly 1 pre-existing app.infrastructure import "
        f"(B2 recovery wrap), found {len(infra_imports)}: {infra_imports}. "
        "The parent_sandbox adapter must be injected via "
        "parent_sandbox_adapter_factory, NOT imported in the domain flow."
    )
