"""C2b §4.1 — PlannerReActFlow injects child_permission_context + SSM into the
child graph cfg (mirror the cancel_event child seam). Root sessions never get
the cpc key (INV-2)."""
from __future__ import annotations

import asyncio

from unittest.mock import MagicMock

from app.application.services.coordinator_runtime_deps import (
    _NullCoordinatorRuntimeDeps,
)
from tests.domain.services.flows.test_planner_react_coord_config import (
    _build_flow_with_real_coord_deps,
)


def _child_flow():
    flow, _ = _build_flow_with_real_coord_deps()
    flow._coord_deps = _NullCoordinatorRuntimeDeps()  # make it a CHILD
    flow._cancel_event = None
    flow._cancel_event_externally_injected = False
    flow._child_permission_context = None
    flow._session_state_machine = MagicMock(name="ssm")
    return flow


def test_set_child_permission_context_stores():
    flow = _child_flow()
    cpc = MagicMock(name="cpc")
    flow.set_child_permission_context(cpc)
    assert flow._child_permission_context is cpc


def test_build_config_injects_cpc_and_ssm_on_child_branch():
    flow = _child_flow()
    flow.set_cancel_event(asyncio.Event())  # child branch trigger
    cpc = MagicMock(name="cpc")
    ssm = flow._session_state_machine
    flow.set_child_permission_context(cpc)
    cfg = flow._build_config()
    assert cfg["configurable"]["child_permission_context"] is cpc
    assert cfg["configurable"]["session_state_machine"] is ssm


def test_build_config_omits_cpc_when_unset_even_with_cancel_event():
    flow = _child_flow()
    flow.set_cancel_event(asyncio.Event())
    cfg = flow._build_config()
    assert "child_permission_context" not in cfg["configurable"]


def test_build_config_root_path_never_injects_cpc():
    """A non-child flow (coord_deps real, no cpc) must not carry the cpc key
    (INV-2 — root behavior byte-identical)."""
    flow, _ = _build_flow_with_real_coord_deps()  # real coord_deps → 'if' branch
    flow._child_permission_context = None
    cfg = flow._build_config()
    assert "child_permission_context" not in cfg["configurable"]
