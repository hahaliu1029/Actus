"""PE-0 Phase 7 — PlannerReActFlow PE/SSM wiring signature tests.

Plan lines 4710-4733. Uses inspect.signature so no real LLM / DB needed.
"""
from __future__ import annotations

import inspect

import pytest

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.services.flows.planner_react import PlannerReActFlow


def test_planner_react_flow_has_permission_engine_kwarg():
    """PlannerReActFlow.__init__ must accept 'permission_engine' as a
    keyword-only-compatible parameter with default None."""
    sig = inspect.signature(PlannerReActFlow.__init__)
    assert "permission_engine" in sig.parameters, (
        "PlannerReActFlow.__init__ is missing 'permission_engine' kwarg"
    )
    param = sig.parameters["permission_engine"]
    assert param.default is None, (
        "permission_engine default must be None, got %r" % param.default
    )


def test_planner_react_flow_has_session_state_machine_kwarg():
    """PlannerReActFlow.__init__ must accept 'session_state_machine' as a
    keyword-only-compatible parameter with default None."""
    sig = inspect.signature(PlannerReActFlow.__init__)
    assert "session_state_machine" in sig.parameters, (
        "PlannerReActFlow.__init__ is missing 'session_state_machine' kwarg"
    )
    param = sig.parameters["session_state_machine"]
    assert param.default is None, (
        "session_state_machine default must be None, got %r" % param.default
    )


def test_planner_react_flow_stores_pe_and_ssm():
    """PlannerReActFlow stores permission_engine as _permission_engine and
    session_state_machine as _session_state_machine on the instance."""
    from unittest.mock import MagicMock

    from app.domain.models.app_config import AgentConfig

    fake_pe = MagicMock()
    fake_ssm = MagicMock()

    flow = PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(max_iterations=10, max_retries=3, max_search_results=5),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        _allow_default_prompt_assembler=True,
        permission_engine=fake_pe,
        session_state_machine=fake_ssm,
    )

    assert flow._permission_engine is fake_pe
    assert flow._session_state_machine is fake_ssm


def test_planner_react_flow_defaults_pe_ssm_to_none():
    """When permission_engine and session_state_machine are not passed,
    both default to None on the instance."""
    from unittest.mock import MagicMock

    from app.domain.models.app_config import AgentConfig

    flow = PlannerReActFlow(
        uow_factory=MagicMock(),
        llm=MagicMock(),
        agent_config=AgentConfig(max_iterations=10, max_retries=3, max_search_results=5),
        session_id="test-session",
        browser_accessor=EagerBrowserAccessor(MagicMock()),
        sandbox_accessor=EagerSandboxAccessor(MagicMock()),
        search_engine=MagicMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        _allow_default_prompt_assembler=True,
    )

    assert flow._permission_engine is None
    assert flow._session_state_machine is None
