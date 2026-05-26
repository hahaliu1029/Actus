"""C2 PR-2 §5.4 — DefaultPermissionEngine child scope prologue tests."""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.session import SessionStatus
from app.domain.services.permission.child_permission_context import (
    ChildBudget,
    ChildPermissionContext,
    SpawnManifest,
)
from app.domain.services.permission.child_scope_gate import ScopeDecision
from app.domain.services.permission.child_scope_violation import ChildScopeViolation
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.default_engine import DefaultPermissionEngine
from app.domain.services.permission.tool_call_spec import ToolCallSpec

pytestmark = pytest.mark.anyio


def _cctx() -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1",
        child_session_id="c1",
        coordinator_run_id="r1",
        work_unit_id="wu1",
        spawn_manifest=SpawnManifest(
            allowed_tools=frozenset({"file_read"}),
            path_leases=(),
            runtime_caps=frozenset(),
        ),
        session_mode_revision=1,
        budget=ChildBudget(
            max_tool_calls=10,
            max_token_cost_usd=1.0,
            max_wallclock_seconds=60,
        ),
    )


def _call(tool_name: str = "file_read", tool_args=None) -> ToolCallSpec:
    return ToolCallSpec(
        tool_name=tool_name,
        tool_args=tool_args or {},
        tool_source="native",
        user_id="u1",
        session_id="c1",
        tool_call_id="tc-1",
    )


@pytest.fixture
def engine() -> DefaultPermissionEngine:
    """Bare-collaborator engine — gate prologue is independent of these."""
    return DefaultPermissionEngine(
        uow_factory=MagicMock(),
        writer=MagicMock(),
        queue=MagicMock(),
        session_machine=MagicMock(),
        reader=MagicMock(),
        escalation_registry={},
        sources={},
        decision_recorder=lambda *a, **kw: None,
    )


async def test_root_session_skips_gate(engine: DefaultPermissionEngine) -> None:
    """ctx.child_permission_context=None => prologue is no-op; downstream may raise but NOT ChildScopeViolation."""
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
    )
    try:
        await engine.evaluate(_call(), ctx)
    except ChildScopeViolation:
        pytest.fail("Root session must skip child gate")
    except Exception:
        pass  # downstream failures (e.g. UnsupportedSource, MagicMock interplay) allowed


async def test_child_in_scope_continues(engine: DefaultPermissionEngine) -> None:
    """When gate returns IN_SCOPE, no ChildScopeViolation; downstream may raise."""
    gate_mock = AsyncMock(return_value=ScopeDecision.IN_SCOPE)
    engine._child_scope_gate = MagicMock(check_in_scope=gate_mock)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=_cctx(),
    )
    try:
        await engine.evaluate(_call(), ctx)
    except ChildScopeViolation:
        pytest.fail("IN_SCOPE must not raise ChildScopeViolation")
    except Exception:
        pass  # downstream failure is OK; we only assert gate non-raise
    gate_mock.assert_awaited_once()


async def test_child_out_of_scope_raises(engine: DefaultPermissionEngine) -> None:
    """When gate returns OUT_OF_PATH_LEASE, evaluate must raise ChildScopeViolation BEFORE any source loop."""
    gate_mock = AsyncMock(return_value=ScopeDecision.OUT_OF_PATH_LEASE)
    engine._child_scope_gate = MagicMock(check_in_scope=gate_mock)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=_cctx(),
    )
    with pytest.raises(ChildScopeViolation) as exc_info:
        await engine.evaluate(_call("file_write", {"path": "/forbidden"}), ctx)
    assert exc_info.value.decision == ScopeDecision.OUT_OF_PATH_LEASE
    assert exc_info.value.tool_name == "file_write"
    assert exc_info.value.target_path == "/forbidden"


async def test_child_hard_blocked_raises(engine: DefaultPermissionEngine) -> None:
    """HARD_BLOCKED decision => ChildScopeViolation with carried decision."""
    gate_mock = AsyncMock(return_value=ScopeDecision.HARD_BLOCKED)
    engine._child_scope_gate = MagicMock(check_in_scope=gate_mock)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=_cctx(),
    )
    with pytest.raises(ChildScopeViolation) as exc_info:
        await engine.evaluate(_call("shell_execute", {}), ctx)
    assert exc_info.value.decision == ScopeDecision.HARD_BLOCKED


async def test_child_budget_exhausted_raises(engine: DefaultPermissionEngine) -> None:
    gate_mock = AsyncMock(return_value=ScopeDecision.BUDGET_EXHAUSTED)
    engine._child_scope_gate = MagicMock(check_in_scope=gate_mock)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=_cctx(),
    )
    with pytest.raises(ChildScopeViolation) as exc_info:
        await engine.evaluate(_call(), ctx)
    assert exc_info.value.decision == ScopeDecision.BUDGET_EXHAUSTED


async def test_prologue_runs_before_source_loop(engine: DefaultPermissionEngine) -> None:
    """If gate denies, evaluate must NOT touch ANY of: writer, queue, reader, ssm, or uow_factory."""
    engine._writer.write = AsyncMock(side_effect=AssertionError("writer.write must NOT be called when gate denies"))
    engine._writer.write_audit_only = AsyncMock(side_effect=AssertionError("writer.write_audit_only must NOT be called when gate denies"))
    engine._queue.store = AsyncMock(side_effect=AssertionError("queue.store must NOT be called when gate denies"))
    engine._queue.read = AsyncMock(side_effect=AssertionError("queue.read must NOT be called when gate denies"))
    engine._reader.check = AsyncMock(side_effect=AssertionError("reader.check must NOT be called when gate denies"))
    engine._ssm.get_mode_with_revision = AsyncMock(side_effect=AssertionError("ssm.get_mode_with_revision must NOT be called when gate denies"))
    # uow_factory is a callable returning an async context manager; tripping the callable is enough
    engine._uow_factory = MagicMock(side_effect=AssertionError("uow_factory must NOT be called when gate denies"))
    gate_mock = AsyncMock(return_value=ScopeDecision.REVISION_DRIFT)
    engine._child_scope_gate = MagicMock(check_in_scope=gate_mock)
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=_cctx(),
    )
    with pytest.raises(ChildScopeViolation):
        await engine.evaluate(_call(), ctx)
    engine._writer.write.assert_not_called()
    engine._queue.store.assert_not_called()
    engine._reader.check.assert_not_called()
    engine._ssm.get_mode_with_revision.assert_not_called()
    engine._uow_factory.assert_not_called()
