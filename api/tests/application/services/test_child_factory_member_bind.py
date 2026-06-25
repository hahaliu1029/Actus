# Unit-test ONLY the bind-union computation. The pure helper
# _apply_member_skill_bind(tool_filter, cpc, preset) lives in the factory so we
# avoid constructing the full runner. [S4 §11]
from app.application.services.child_agent_runner_factory import (
    _apply_member_skill_bind,
)
from app.domain.models.tool_filter_presets import COORDINATOR_STEP_PRESET


class _Cpc:
    def __init__(self, tools):
        self.member_skill_tools = frozenset(tools)


def test_member_tools_unioned_for_coordinator_preset():
    out = _apply_member_skill_bind(
        frozenset({"file_read"}), _Cpc({"skill_a_x"}), COORDINATOR_STEP_PRESET
    )
    assert out == frozenset({"file_read", "skill_a_x"})


def test_no_union_for_other_preset():
    out = _apply_member_skill_bind(
        frozenset({"file_read"}), _Cpc({"skill_a_x"}), "subagent_research"
    )
    assert out == frozenset({"file_read"})


def test_empty_member_tools_is_identity():
    base = frozenset({"file_read"})
    assert (
        _apply_member_skill_bind(base, _Cpc(set()), COORDINATOR_STEP_PRESET) == base
    )


def test_none_tool_filter_passthrough():
    assert (
        _apply_member_skill_bind(None, _Cpc({"skill_a_x"}), COORDINATOR_STEP_PRESET)
        is None
    )
