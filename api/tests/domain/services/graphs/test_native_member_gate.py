# api/tests/domain/services/graphs/test_native_member_gate.py
import pytest

from app.domain.models.work_unit import WorkUnit
from app.domain.services.graphs.parallel_execution_subgraph import (
    _enforce_native_member_gate,
)
from app.domain.services.team_expander import MemberCapability, TeamCapabilityError


def _cap_native():
    return MemberCapability(system_prompt="p", member_skill_tools=frozenset({"skill_nat_go"}),
                            member_skill_slugs=("nat",), native_skill_slugs=("nat",), shell_mode=True)


def _wu(**kw):
    base = dict(work_unit_id="w.a0.0", objective="o", phase="write",
                write_tree_lease=[], shell_mode=True, role="r",
                member_skill_tools=frozenset({"skill_nat_go"}), member_skill_slugs=("nat",))
    # a write unit needs a lease; give a path lease
    from app.domain.models.work_unit import PathLease
    base["write_lease"] = [PathLease(path="workspace/a.py", op="add")]
    base.update(kw)
    return WorkUnit(**base)


def test_native_in_shell_write_unit_allowed():
    units = [_wu(shell_mode=True, phase="write")]
    out = _enforce_native_member_gate(units, {"r": _cap_native()})
    assert out == units  # unchanged


def test_native_in_typed_only_unit_fails_closed():
    # post-coercion shell_mode False (flag-off demotion or never-shell) ⇒ reject
    units = [_wu(shell_mode=False, phase="write", write_tree_lease=[])]
    with pytest.raises(TeamCapabilityError):
        _enforce_native_member_gate(units, {"r": _cap_native()})


def test_no_native_member_is_identity():
    cap = MemberCapability(system_prompt="p", member_skill_tools=frozenset(),
                           member_skill_slugs=(), native_skill_slugs=(), shell_mode=False)
    units = [_wu(shell_mode=False, phase="write", write_tree_lease=[],
                 member_skill_tools=frozenset(), member_skill_slugs=())]
    assert _enforce_native_member_gate(units, {"r": cap}) == units


def test_no_team_map_is_identity():
    units = [_wu(role=None, member_skill_tools=frozenset(), member_skill_slugs=(),
                 shell_mode=False, phase="write", write_tree_lease=[])]
    assert _enforce_native_member_gate(units, None) == units


def test_both_dispatch_sites_invoke_native_gate():
    # [codex-R3-F5] both call sites (:569 rehydrate, :610 first-time) must call the
    # gate AFTER _coerce — helper-only tests pass even if one branch is missed.
    import ast
    import pathlib
    tree = ast.parse(
        pathlib.Path("app/domain/services/graphs/parallel_execution_subgraph.py").read_text()
    )
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_enforce_native_member_gate"
    ]
    assert len(calls) >= 2, f"both dispatch sites must invoke the native gate, found {len(calls)}"
