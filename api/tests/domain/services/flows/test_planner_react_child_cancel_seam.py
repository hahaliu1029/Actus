"""Tests for the C2 finish-core child cancel seam (F1.2).

A coordinator CHILD PlannerReActFlow has `_coord_deps` set to the Null sentinel
(a child must not be a nested coordinator), so `_build_config()` normally omits
all coordinator keys including ``cancel_event``. The invoke-adapter, however,
needs to forward the coordinator's per-work-unit ``cancel_event`` into the child
so react_graph cancel checkpoints observe a parent-initiated cancel.

This module verifies the seam:
  * ``PlannerReActFlow.set_cancel_event`` records the event and marks it as
    externally injected.
  * ``_build_config`` injects ONLY ``cancel_event`` on the Null-coord-deps path
    when one was set, and omits it otherwise.
  * ``AgentTaskRunner._prime_planner_cancel_event_for_coord_deps`` never clobbers
    an adapter-injected event.
"""

from __future__ import annotations

import asyncio


def _make_child_flow():
    """A real PlannerReActFlow whose coord_deps is then set to Null (a child)."""
    from app.application.services.coordinator_runtime_deps import (
        _NullCoordinatorRuntimeDeps,
    )

    # Reuse the working real-args helper from test_planner_react_coord_config.py
    # (it builds a flow on which _build_config() runs without AttributeError).
    from tests.domain.services.flows.test_planner_react_coord_config import (
        _build_flow_with_real_coord_deps,
    )

    flow, _ = _build_flow_with_real_coord_deps()
    flow._coord_deps = _NullCoordinatorRuntimeDeps()  # make it a CHILD (null coord)
    flow._cancel_event = None
    flow._cancel_event_externally_injected = False
    return flow


def test_set_cancel_event_marks_externally_injected():
    flow = _make_child_flow()
    ev = asyncio.Event()
    flow.set_cancel_event(ev)
    assert flow._cancel_event is ev
    assert flow._cancel_event_externally_injected is True


def test_build_config_injects_cancel_event_on_null_coord_deps_when_set():
    flow = _make_child_flow()
    ev = asyncio.Event()
    flow.set_cancel_event(ev)
    cfg = flow._build_config()
    assert cfg["configurable"]["cancel_event"] is ev


def test_build_config_omits_cancel_event_on_null_coord_deps_when_unset():
    flow = _make_child_flow()
    cfg = flow._build_config()
    assert "cancel_event" not in cfg["configurable"]


def test_prime_does_not_overwrite_externally_injected_event():
    from app.domain.services.agent_task_runner import AgentTaskRunner

    flow = _make_child_flow()
    ev = asyncio.Event()
    flow.set_cancel_event(ev)
    runner = object.__new__(AgentTaskRunner)
    runner._coord_deps_for_planner = None  # child -> prime returns early anyway
    runner._flow = flow
    runner._prime_planner_cancel_event_for_coord_deps()
    assert flow._cancel_event is ev  # not clobbered
