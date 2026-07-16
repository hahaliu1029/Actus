"""SPM PR-1b Task 10 — runner / flow ctor accessor contract (compile-time forcing).

The migration DELETES the legacy ``sandbox=`` / ``browser=`` ctor params and
REQUIRES the new ``sandbox_accessor=`` / ``browser_accessor=`` params. A pure
``inspect.signature`` assertion locks the contract so a regression that re-adds
a raw-handle param (silently re-coupling the runner/flow to a concrete handle)
fails loudly.
"""
from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.anyio


class TestRunnerAccessorCtor:
    def test_ctor_requires_accessors_not_raw_handles(self):
        """Legacy ``sandbox=`` / ``browser=`` deleted; accessor kwargs present."""
        from app.domain.services.agent_task_runner import AgentTaskRunner

        params = inspect.signature(AgentTaskRunner.__init__).parameters
        assert "sandbox_accessor" in params and "browser_accessor" in params
        assert "sandbox" not in params and "browser" not in params


class TestFlowAccessorCtor:
    def test_ctor_requires_accessors_not_raw_handles(self):
        """The flow is the second raw-handle consumption chain (r2) — same rule."""
        from app.domain.services.flows.planner_react import PlannerReActFlow

        params = inspect.signature(PlannerReActFlow.__init__).parameters
        assert "sandbox_accessor" in params and "browser_accessor" in params
        assert "sandbox" not in params and "browser" not in params
