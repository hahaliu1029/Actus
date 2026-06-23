# api/tests/domain/services/permission/test_child_scope_gate_shell_mode_contract.py
# [S2 §9(c)] PR-5 Task 5.4 — dual-caller contract LOCK (no production code).
# Both production gate-invocation sites (DefaultPermissionEngine.evaluate and
# react_graph._enforce_child_scope_or_raise) must honor the shell-mode un-block
# transitively through ChildScopeGate.check_in_scope — the single flag read site
# for the two RUNTIME tool-call gate callers (the factory bind-time widen and the
# planner teaching read the same flag independently elsewhere). A FAIL here is a
# caller-side wiring drift, which is the regression this locks.
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.session import SessionStatus
from app.domain.services.permission import child_scope_gate as gate_mod
from app.domain.services.permission.child_permission_context import (
    ChildBudget, ChildPermissionContext, SpawnManifest,
)
from app.domain.services.permission.child_scope_violation import (
    ChildScopeViolation,
)
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.tool_call_spec import ToolCallSpec
from app.domain.services.risk_assessor import RiskAssessment, RiskLevel

anyio_backend = "asyncio"
pytestmark = pytest.mark.anyio


def _cpc(*, shell_mode: bool, allowed=frozenset({"shell_execute"})):
    return ChildPermissionContext(
        parent_session_id="p1", child_session_id="c1", coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=allowed, path_leases=(), runtime_caps=frozenset(),
            tree_leases=(), shell_mode=shell_mode,
        ),
        session_mode_revision=1,
        budget=ChildBudget(max_tool_calls=100, max_token_cost_usd=1.0,
                           max_wallclock_seconds=600),
        shell_mode=shell_mode,
    )


# ── Caller A: DefaultPermissionEngine.evaluate (REAL evaluate, not a hand-rolled
#    re-implementation of the prologue). The flag-ON shell path must reach
#    AllowSuccess through evaluate(); the flag-OFF path must let the child-scope
#    prologue raise ChildScopeViolation. Calling evaluate() for real is the
#    point: a hand-call of check_in_scope would pass even if evaluate() stopped
#    invoking the gate / passed the wrong ctx (false coverage). ──
def _NONE_RISK(tool_name: str) -> RiskAssessment:
    # Source-derived assessment forcing the §7a AUTO/no-policy + safe-risk
    # AllowSuccess short-circuit (so the test never needs a real writer/queue).
    return RiskAssessment(
        tool_name=tool_name, tool_args={}, static_level=RiskLevel.NONE,
        dynamic_level=RiskLevel.NONE, final_level=RiskLevel.NONE,
        risk_reason="test", matched_patterns=[], suggested_alternative=None,
        primary_arg="", dir_arg=None, arg_digest="",
    )


async def _evaluate_shell(*, shell_mode: bool):
    from app.domain.services.permission.default_engine import (
        DefaultPermissionEngine,
    )
    from app.domain.services.permission.child_scope_gate import ChildScopeGate

    engine = DefaultPermissionEngine.__new__(DefaultPermissionEngine)
    engine._child_scope_gate = ChildScopeGate()
    engine._decision_recorder = lambda *a, **kw: None
    engine._record_decision = MagicMock()
    engine._build_attrs = MagicMock(return_value={})
    # Policy → None ⇒ the §7a AUTO/no-policy branch is reachable.
    engine._get_policy = AsyncMock(return_value=None)
    # Source dispatch: a fake "native" source whose assess_risk returns a
    # NONE-risk assessment (so risk gate stays safe → AllowSuccess).
    fake_source = MagicMock()
    fake_source.assess_risk = AsyncMock(return_value=_NONE_RISK("shell_execute"))
    engine._sources = {"native": fake_source}
    # Stage P.1 reader → "no_match" so we fall through to the risk gate.
    engine._reader = MagicMock()
    engine._reader.check = AsyncMock(return_value="no_match")
    # Writer must never be touched on the AllowSuccess path; fail loudly if it is.
    engine._writer = MagicMock()
    engine._writer.write = AsyncMock(
        side_effect=AssertionError("writer must not be called on AllowSuccess")
    )

    # Real ToolCallSpec (frozen dataclass) — evaluate() does
    # dataclasses.replace(call, risk_assessment=...), which requires a real
    # dataclass instance, not a MagicMock.
    call = ToolCallSpec(
        tool_name="shell_execute", tool_args={}, tool_source="native",
        user_id="u1", session_id="c1",
    )
    cpc = _cpc(shell_mode=shell_mode)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING, session_mode_revision=1,
        child_permission_context=cpc,
    )
    # REAL call — exercises the prologue (gate) + the full evaluate() pipeline.
    return await engine.evaluate(call, ctx)


async def test_caller_a_pe_prologue_passes_when_flag_on_shell_mode(monkeypatch):
    from app.domain.models.tool_result import AllowSuccess

    monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    outcome = await _evaluate_shell(shell_mode=True)  # no ChildScopeViolation
    # Reaching AllowSuccess proves evaluate() ran the gate with the right ctx
    # AND continued past the prologue to the AUTO short-circuit.
    assert isinstance(outcome, AllowSuccess)


async def test_caller_a_pe_prologue_blocks_when_flag_off(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                        lambda: False)
    with pytest.raises(ChildScopeViolation):
        await _evaluate_shell(shell_mode=True)


# ── Caller B: react_graph._enforce_child_scope_or_raise ──
async def _enforce_shell(*, shell_mode: bool):
    from app.domain.services.graphs.react_graph import (
        _enforce_child_scope_or_raise,
    )
    cpc = _cpc(shell_mode=shell_mode)
    ssm = AsyncMock()
    ssm.get_mode_with_revision = AsyncMock(
        return_value=(SessionStatus.RUNNING, 1)
    )
    configurable = {
        "child_permission_context": cpc,
        "session_state_machine": ssm,
        "session_id": "c1",
        "request_id": "req-1",
    }
    ai = AIMessage(content="", tool_calls=[{
        "id": "tc1", "name": "shell_execute", "args": {},
    }])
    state = {"messages": [ai], "completed_tool_call_prefix": []}
    await _enforce_child_scope_or_raise(state, configurable)


async def test_caller_b_react_guard_passes_when_flag_on_shell_mode(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                        lambda: True)
    await _enforce_shell(shell_mode=True)  # no raise


async def test_caller_b_react_guard_blocks_when_flag_off(monkeypatch):
    monkeypatch.setattr(gate_mod, "is_coordinator_shell_mode_enabled",
                        lambda: False)
    with pytest.raises(ChildScopeViolation):
        await _enforce_shell(shell_mode=True)
