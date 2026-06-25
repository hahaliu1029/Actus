"""C2-full S4 Task 3.5 — the merge branch in ``_build_work_units_from_requests``.

When a team is selected (``team_member_map is not None``) AND a work unit carries
a roster ``role``, the built ``WorkUnit`` merges that member's
``MemberCapability`` (system_prompt, member skill tools, member-authoritative
shell_mode, gate unions). With no team / no role, the build stays byte-identical
to pre-S4 (INV-0).
"""
import pytest

from app.domain.models.work_unit import WorkUnitRequest
from app.domain.services.graphs.parallel_execution_subgraph import (
    _build_work_units_from_requests,
)
from app.domain.services.team_expander import MemberCapability, TeamCapabilityError


def _cap(**kw):
    base = dict(system_prompt="persona", member_skill_tools=frozenset({"skill_repo_map_go"}),
                member_skill_slugs=("repo-map",), native_skill_slugs=(), shell_mode=False)
    base.update(kw)
    return MemberCapability(**base)


def _req(**kw):
    base = dict(objective="o", phase="exploration", allowed_tools=["file_read"])
    base.update(kw)
    return WorkUnitRequest(**base)


def test_no_role_is_inv0_verbatim():
    # allowed_tools kept verbatim (NOT dict.fromkeys), no member fields
    req = _req(allowed_tools=["file_read", "file_read"])  # intentional dup
    units = _build_work_units_from_requests([req], "hash16", 0, team_member_map={"explorer": _cap()})
    u = units[0]
    assert u.role is None
    assert u.allowed_tools == ["file_read", "file_read"]   # duplicates preserved
    assert u.system_prompt is None
    assert u.member_skill_tools == frozenset()
    assert u.member_skill_slugs == ()


def test_role_emitted_with_none_map_is_inv0_no_raise():
    # [codex-R7 — INV-0 critical] flag-OFF/no-team ⇒ team_member_map is None; an
    # LLM-emitted role (the field is in the planner schema unconditionally) is a
    # DEAD no-op, NOT a TeamCapabilityError.
    from app.domain.services.graphs.parallel_execution_subgraph import (
        _serialize_spawn_manifest,
    )
    req = _req(role="explorer", allowed_tools=["file_read"])
    units = _build_work_units_from_requests([req], "h", 0, team_member_map=None)
    u = units[0]
    assert u.system_prompt is None
    assert u.member_skill_tools == frozenset()
    # serializes identically to a no-role pre-S4 unit (role not in the manifest)
    plain = _build_work_units_from_requests(
        [_req(allowed_tools=["file_read"])], "h", 0, team_member_map=None,
    )[0]
    assert _serialize_spawn_manifest(u) == _serialize_spawn_manifest(plain)


def test_merge_branch_applies_member_capability():
    req = _req(role="explorer", allowed_tools=["file_read"])
    units = _build_work_units_from_requests([req], "hash16", 0,
                                            team_member_map={"explorer": _cap()})
    u = units[0]
    assert u.role == "explorer"
    assert u.system_prompt == "persona"
    assert "skill_repo_map_go" in u.allowed_tools          # gate union
    assert u.member_skill_tools == frozenset({"skill_repo_map_go"})
    assert u.member_skill_slugs == ("repo-map",)


def test_merge_allowed_tools_member_part_sorted_deterministic():
    # [codex-R1-F5] member tool names append in sorted order (frozenset iteration
    # is process-dependent; the manifest is content-addressed).
    cap = _cap(member_skill_tools=frozenset({"skill_b_y", "skill_a_x"}))
    req = _req(role="explorer", allowed_tools=["file_read"])
    units = _build_work_units_from_requests([req], "h", 0, team_member_map={"explorer": cap})
    assert units[0].allowed_tools == ["file_read", "skill_a_x", "skill_b_y"]


def test_role_not_in_map_fails_closed():
    req = _req(role="ghost")
    with pytest.raises(TeamCapabilityError):
        _build_work_units_from_requests([req], "hash16", 0, team_member_map={"explorer": _cap()})


def test_shell_conflict_member_not_shell_capable_raises():
    # write task needs shell (proposed_trees) but member shell_mode=False ⇒ reject,
    # NOT an uncaught ValueError at work_unit.py:204-208 (R4-#4).
    req = _req(role="impl", phase="write",
               proposed_trees=[{"prefix": "workspace/sub", "ops": ["add"]}])
    with pytest.raises(TeamCapabilityError):
        _build_work_units_from_requests([req], "hash16", 0,
                                        team_member_map={"impl": _cap(shell_mode=False)})


def test_shell_capable_member_gets_shell_gate_union():
    req = _req(role="impl", phase="write",
               proposed_trees=[{"prefix": "workspace/sub", "ops": ["add"]}])
    units = _build_work_units_from_requests([req], "hash16", 0,
                                            team_member_map={"impl": _cap(shell_mode=True)})
    u = units[0]
    assert u.shell_mode is True
    # F4: the 5 shell tools must be in allowed_tools (gate step-1 precedes shell-release)
    assert {"shell_execute", "shell_read_output"} <= set(u.allowed_tools)
