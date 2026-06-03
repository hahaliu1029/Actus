"""PE-4c — planner_react._build_config threads the master tool_confirmation
config into the LangGraph configurable; the per-source PE flags are retired,
so no per-source configurable keys are injected anymore."""

from __future__ import annotations

import inspect


def test_build_config_threads_master_tool_confirmation_config():
    """The configurable carries the master ``tool_confirmation_config`` object
    so the gate helper resolves ``enabled`` per call."""
    import app.domain.services.flows.planner_react as m
    src = inspect.getsource(m)
    assert '"tool_confirmation_config"' in src


def test_build_config_no_longer_injects_per_source_pe_keys():
    """PE-4c: the per-source configurable keys are gone (the flags are deleted)."""
    import app.domain.services.flows.planner_react as m
    src = inspect.getsource(m)
    assert '"permission_engine_native_enabled"' not in src
    assert '"permission_engine_skill_enabled"' not in src
    assert '"permission_engine_mcp_enabled"' not in src
    assert '"permission_engine_a2a_enabled"' not in src
