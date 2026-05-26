"""C2 PR-2 §5.4 — EvaluationContext.child_permission_context optional field."""
import pytest
from app.domain.models.session import SessionStatus
from app.domain.services.permission.context import EvaluationContext
from app.domain.services.permission.child_permission_context import (
    ChildBudget, ChildPermissionContext, ChildRuntimeCap, SpawnManifest,
)


def _child_ctx() -> ChildPermissionContext:
    return ChildPermissionContext(
        parent_session_id="p1",
        child_session_id="c1",
        coordinator_run_id="p1:abcd1234abcd1234:a1",
        work_unit_id="abcd1234abcd1234.a1.0",
        spawn_manifest=SpawnManifest(
            allowed_tools=frozenset({"file_read"}),
            path_leases=(),
            runtime_caps=frozenset({ChildRuntimeCap.NO_RESPAWN}),
        ),
        session_mode_revision=1,
        budget=ChildBudget(
            max_tool_calls=10,
            max_token_cost_usd=0.10,
            max_wallclock_seconds=60,
        ),
    )


def test_default_none():
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
    )
    assert ctx.child_permission_context is None


def test_accepts_child_ctx():
    c = _child_ctx()
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
        child_permission_context=c,
    )
    assert ctx.child_permission_context is c


def test_child_ctx_field_is_frozen():
    """frozen dataclass — direct assignment must raise FrozenInstanceError."""
    import dataclasses
    ctx = EvaluationContext(
        session_mode=SessionStatus.RUNNING,
        session_mode_revision=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.child_permission_context = _child_ctx()  # type: ignore[misc]
