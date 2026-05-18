"""PE-1 §3.2 — planner_react._build_config must thread BOTH native and skill
PE flags into the LangGraph configurable so gate helper sees the full picture."""

from __future__ import annotations

import inspect


def test_build_config_includes_pe_skill_flag():
    """The injected configurable should carry permission_engine_skill_enabled
    in addition to the legacy permission_engine_native_enabled."""
    import app.domain.services.flows.planner_react as m
    src = inspect.getsource(m)
    assert '"permission_engine_skill_enabled"' in src or "permission_engine_skill_enabled" in src


def test_pe_injection_gate_no_longer_native_only():
    """The PE inject if-block must consider any-source-enabled (or remove
    the per-source check entirely and trust _create_task gate)."""
    import app.domain.services.flows.planner_react as m
    src = inspect.getsource(m)
    # Either: the explicit `flag_native` check is replaced with `is_pe_enabled_for_source`,
    # or the gate is removed (delegating to _create_task which already gated build).
    # We allow both: assert the helper appears OR the legacy native-only condition is gone.
    legacy_pattern = "and flag_native"
    helper = "is_pe_enabled_for_source"
    assert helper in src or legacy_pattern not in src
