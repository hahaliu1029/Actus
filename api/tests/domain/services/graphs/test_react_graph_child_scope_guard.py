"""C2b §4.3 — _enforce_child_scope_or_raise: tool_node-entry child-scope guard."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from app.domain.models.session import SessionStatus
from app.domain.models.work_unit import PathLease
from app.domain.services.graphs import react_graph as rg
from app.domain.services.permission.child_permission_context import (
    ChildBudget, ChildPermissionContext, SpawnManifest,
)
from app.domain.services.permission.child_scope_gate import (
    ChildScopeGate, ScopeDecision,
)
from app.domain.services.permission.child_scope_violation import ChildScopeViolation

pytestmark = pytest.mark.anyio


def _cpc(*, allowed, leases=(), rev=6, max_tool_calls=100,
         lease_expiry=None) -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1", child_session_id="c1", coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=frozenset(allowed), path_leases=tuple(leases),
            runtime_caps=frozenset(),
        ),
        session_mode_revision=rev,
        budget=ChildBudget(max_tool_calls=max_tool_calls, max_token_cost_usd=1.0,
                           max_wallclock_seconds=600),
        lease_expiry=lease_expiry,
    )


def _state(tool_calls, completed=()):
    return {
        "messages": [AIMessage(content="", tool_calls=list(tool_calls))],
        "completed_tool_call_prefix": list(completed),
    }


def _configurable(cpc, *, live_rev=6, mode=SessionStatus.RUNNING):
    ssm = MagicMock()
    ssm.get_mode_with_revision = AsyncMock(return_value=(mode, live_rev))
    return {
        "child_permission_context": cpc,
        "session_state_machine": ssm,
        "session_id": "c1",
        "user_id": "u1",
    }


def _tc(name, *, id="tc1", **args):
    return {"id": id, "name": name, "args": args, "type": "tool_call"}


def test_module_level_gate_is_child_scope_gate():
    assert isinstance(rg._child_scope_gate, ChildScopeGate)


async def test_noop_when_no_cpc():
    state = _state([_tc("file_read", path="/a")])
    # root: no child_permission_context in configurable
    await rg._enforce_child_scope_or_raise(state, {"session_id": "root1"})  # no raise


async def test_in_scope_read_passes():
    cpc = _cpc(allowed={"file_read"})
    state = _state([_tc("file_read", path="/a")])
    await rg._enforce_child_scope_or_raise(state, _configurable(cpc))  # no raise


async def test_out_of_allowlist_raises():
    cpc = _cpc(allowed={"file_read"})
    state = _state([_tc("shell_execute", command="ls")])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.OUT_OF_TOOL_ALLOWLIST
    assert ei.value.tool_name == "shell_execute"


async def test_hard_blocked_raises_even_if_in_allowlist():
    cpc = _cpc(allowed={"message_ask_user"})
    state = _state([_tc("message_ask_user", text="hi")])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.HARD_BLOCKED


async def test_write_out_of_path_lease_raises():
    cpc = _cpc(allowed={"file_write"}, leases=())  # no lease for the target
    state = _state([_tc("file_write", filepath="/forbidden", content="x")])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.OUT_OF_PATH_LEASE
    assert ei.value.target_path == "/forbidden"


async def test_write_in_lease_passes():
    lease = PathLease(path="ok.py", op="modify", base_digest="d")
    cpc = _cpc(allowed={"file_write"}, leases=(lease,))
    state = _state([_tc("file_write", filepath="ok.py", content="x")])
    await rg._enforce_child_scope_or_raise(state, _configurable(cpc))  # no raise


async def test_op_mismatch_raises():
    """file_delete against a modify lease → OP_MISMATCH (mirror
    test_child_scope_gate.py::test_delete_with_modify_lease_op_mismatch)."""
    lease = PathLease(path="/x", op="modify", base_digest="abc")
    cpc = _cpc(allowed={"file_delete"}, leases=(lease,))
    state = _state([_tc("file_delete", path="/x")])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.OP_MISMATCH


async def test_lease_expired_raises():
    """An expired lease_expiry → LEASE_EXPIRED (gate step 5, before the
    revision check; live_rev==baseline so drift would NOT fire here)."""
    past = datetime.now(timezone.utc) - timedelta(minutes=10)
    cpc = _cpc(allowed={"file_read"}, lease_expiry=past)
    state = _state([_tc("file_read", path="/a")])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.LEASE_EXPIRED


async def test_revision_drift_raises():
    cpc = _cpc(allowed={"file_read"}, rev=6)
    state = _state([_tc("file_read", path="/a")])
    # live read returns rev 7 ≠ cpc baseline 6 → drift
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc, live_rev=7))
    assert ei.value.decision == ScopeDecision.REVISION_DRIFT


async def test_completed_calls_are_skipped():
    cpc = _cpc(allowed={"file_read"})
    # the only call is out-of-scope BUT already completed → skipped → no raise
    state = _state([_tc("shell_execute", id="done1", command="ls")],
                   completed=["done1"])
    await rg._enforce_child_scope_or_raise(state, _configurable(cpc))  # no raise


async def test_unknown_tool_name_maps_to_out_of_allowlist():
    """An unknown tool name (resolve_tool_source raises ToolSourceUnknownError)
    must reach the gate as OUT_OF_TOOL_ALLOWLIST, not crash before the gate."""
    cpc = _cpc(allowed={"file_read"})
    state = _state([_tc("totally_made_up_tool", x=1)])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.OUT_OF_TOOL_ALLOWLIST


async def test_mixed_batch_is_atomic_first_violation_raises():
    """call#1 in-scope + call#2 out-of-scope → guard raises at entry; neither
    executes (the guard runs before any per-call dispatch)."""
    cpc = _cpc(allowed={"file_read"})
    state = _state([
        _tc("file_read", id="t1", path="/a"),
        _tc("shell_execute", id="t2", command="ls"),
    ])
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.tool_name == "shell_execute"


async def test_pre_approved_replay_call_is_still_guarded():
    """[R2 P2-a / §4.3 deferred-c] A call that PE pre-approval would replay
    (id ∈ approved_tool_call_ids) is NOT exempt from the child-scope guard:
    the guard only skips completed_tool_call_prefix, so a pre-approved
    out-of-scope call STILL raises. Proves replay/approval does not bypass."""
    cpc = _cpc(allowed={"file_read"})
    state = {
        "messages": [AIMessage(content="", tool_calls=[_tc("shell_execute", command="ls")])],
        "completed_tool_call_prefix": [],
        "approved_tool_call_ids": ["tc1"],     # pre-approved — but still guarded
        "pe_resume_outcomes": {"tc1": object()},  # resume replay slot — ignored too
    }
    with pytest.raises(ChildScopeViolation) as ei:
        await rg._enforce_child_scope_or_raise(state, _configurable(cpc))
    assert ei.value.decision == ScopeDecision.OUT_OF_TOOL_ALLOWLIST


async def test_guard_ssm_read_failure_propagates_raw():
    """[R2 P3 / §4.2 R3-P3] The guard's OWN get_mode_with_revision failure is
    NOT caught here — it propagates as a generic exception (→ the runner's
    generic catch-all → FAILED, still fail-closed), and is deliberately NOT
    converted to ChildScopeViolation (which would imply replannable
    NEEDS_AUTHORIZATION). Locks the §4.2 R3-P3 error posture."""
    cpc = _cpc(allowed={"file_read"})
    ssm = MagicMock()
    ssm.get_mode_with_revision = AsyncMock(side_effect=RuntimeError("ssm down"))
    configurable = {
        "child_permission_context": cpc, "session_state_machine": ssm,
        "session_id": "c1", "user_id": "u1",
    }
    state = _state([_tc("file_read", path="/a")])
    with pytest.raises(RuntimeError, match="ssm down"):
        await rg._enforce_child_scope_or_raise(state, configurable)


def test_tool_node_calls_guard_before_pe_dispatch():
    """[R2 P2-a] Lock the wiring: tool_node must call _enforce_child_scope_or_raise
    BEFORE the PE dispatch branch (so the guard covers the whole batch regardless
    of PE/legacy routing). Source-level assertion — driving the full graph would
    need extensive fixture wiring (mirrors the existing react_graph AST tests)."""
    import inspect
    src = inspect.getsource(rg)
    guard_i = src.find("_enforce_child_scope_or_raise(state, configurable)")
    pe_i = src.find("_pe_result = await _pe_dispatch(state, config)")
    assert guard_i != -1, "tool_node must call _enforce_child_scope_or_raise"
    assert pe_i != -1, "tool_node PE dispatch call anchor not found"
    assert guard_i < pe_i, "child-scope guard must run BEFORE the PE dispatch branch"
