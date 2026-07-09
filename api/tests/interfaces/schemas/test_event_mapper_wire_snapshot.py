"""C7 PR1 — INV-C7-4：旧 SSE wire schema snapshot（EventMapper 输出 shape 逐事件 pin，R1#11）。

首次生成 fixture：UPDATE_WIRE_SNAPSHOT=1 uv run pytest <本文件>
之后任何改动使存量事件的 (wire event 名, data 键集合) 变化 → 本测试红。
"""
import json
import os
from pathlib import Path

import pytest

from app.domain.models.event import (
    CompactionEvent,
    ContextStatusEvent,
    ControlAction,
    ControlEvent,
    CoordinatorApplyEvent,
    CoordinatorDispatchEvent,
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    CoordinatorWorkerSpawnedEvent,
    DoneEvent,
    ErrorEvent,
    ExecutionStateChangedEvent,
    ExecutionStatePayload,
    FinishingEvent,
    HealthEvent,
    HealthStatus,
    MessageEvent,
    OwnerConflictEvent,
    OwnerConflictPayload,
    PlanEvent,
    SandboxStateChangedEvent,
    SessionModeChangedEvent,
    StepEvent,
    TitleEvent,
    ToolConfirmationEvent,
    ToolEvent,
    WaitEvent,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.plan import Plan, Step
from app.interfaces.schemas.event import EventMapper

FIXTURE = Path(__file__).parent / "fixtures" / "event_mapper_wire_snapshot.json"

# 全部 23 个存量 Event 联合成员的最小实例（LifecycleEvent 有意不在此表——它是新增项）
LEGACY_EVENT_FACTORIES = {
    "plan": lambda: PlanEvent(plan=Plan(id="p1", steps=[Step(id="s1")])),
    "title": lambda: TitleEvent(title="t"),
    "step": lambda: StepEvent(step=Step(id="s1")),
    "message": lambda: MessageEvent(),
    "tool": lambda: ToolEvent(tool_call_id="tc1", tool_name="shell", function_name="shell_execute", function_args={}),
    "wait": lambda: WaitEvent(),
    "control": lambda: ControlEvent(action=ControlAction.STARTED),
    "error": lambda: ErrorEvent(),
    "context_status": lambda: ContextStatusEvent(),
    "compaction": lambda: CompactionEvent(),
    "finishing": lambda: FinishingEvent(),
    "health": lambda: HealthEvent(status=HealthStatus.HEALTHY, reason="r"),
    "tool_confirmation": lambda: ToolConfirmationEvent(
        tool_call_id="tc1", tool_name="shell", tool_args={}, risk_level="high",
        risk_reason="r", matched_patterns=[], timeout_seconds=60,
    ),
    "sandbox_state_changed": lambda: SandboxStateChangedEvent(old_state="active", new_state="suspended", generation=1),
    "session_mode_changed": lambda: SessionModeChangedEvent(to="running", reason="test"),
    "execution_state_changed": lambda: ExecutionStateChangedEvent(payload=ExecutionStatePayload(
        execution_mode="foreground", execution_phase="running", retry_budget_remaining=3,
    )),
    "owner_conflict": lambda: OwnerConflictEvent(payload=OwnerConflictPayload(
        current_owner_connection_id="a", conflicting_connection_id="b", session_id="s",
    )),
    "coordinator_dispatch": lambda: CoordinatorDispatchEvent(
        step_id="s1", work_unit_count=1, work_unit_ids=["wu1"], phases=["exploration"],
    ),
    "coordinator_worker_spawned": lambda: CoordinatorWorkerSpawnedEvent(
        objective="o", phase="exploration", allowed_tools=[], write_lease_count=0,
    ),
    "coordinator_reduce": lambda: CoordinatorReduceEvent(
        group_outcome=GroupOutcome.SUCCESS, per_worker_outcomes={"wu1": ResultReadyOutcome.SUCCESS},
        diagnostics_summary="", cost_total=CostAggregate(),
    ),
    "coordinator_apply": lambda: CoordinatorApplyEvent(apply_status="applied"),
    "coordinator_sibling_cancel": lambda: CoordinatorSiblingCancelEvent(
        triggered_by_work_unit_id="wu1", triggered_by_outcome=ResultReadyOutcome.FAILED,
        cancelled_work_unit_ids=[], reason="fail_fast",
    ),
    "done": lambda: DoneEvent(),
}


def _current_shapes() -> dict:
    shapes = {}
    for name, factory in sorted(LEGACY_EVENT_FACTORIES.items()):
        sse = EventMapper.event_to_sse_event(factory())
        shapes[name] = {
            "wire_event": sse.event,
            "sse_class": type(sse).__name__,
            "data_keys": sorted(json.loads(sse.to_sse_data_json()).keys()),
        }
    return shapes


def test_factories_cover_all_legacy_union_members():
    from typing import get_args
    from app.domain.models.event import Event
    union_types = {t.model_fields["type"].default for t in get_args(get_args(Event)[0])}
    assert union_types - {"lifecycle"} == set(LEGACY_EVENT_FACTORIES.keys())


def test_legacy_wire_shapes_frozen():
    current = _current_shapes()
    if os.environ.get("UPDATE_WIRE_SNAPSHOT") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(current, indent=2, ensure_ascii=False, sort_keys=True))
        pytest.skip("snapshot updated — rerun without UPDATE_WIRE_SNAPSHOT")
    assert FIXTURE.exists(), "run once with UPDATE_WIRE_SNAPSHOT=1 to create the baseline"
    frozen = json.loads(FIXTURE.read_text())
    assert current == frozen, "legacy SSE wire shape drifted — INV-C7-4 violation"


def test_legacy_status_enums_frozen():
    # INV-C7-4 前半：存量枚举值 frozen 断言（F1-F3 词表零改动）
    from app.domain.models.event import PlanEventStatus, StepEventStatus, ToolEventStatus
    from app.domain.models.plan import ExecutionStatus
    assert {m.value for m in PlanEventStatus} == {"created", "updated", "completed"}
    assert {m.value for m in StepEventStatus} == {"started", "completed", "failed"}
    assert {m.value for m in ToolEventStatus} == {"calling", "running", "called"}
    assert {m.value for m in ExecutionStatus} == {"pending", "running", "completed", "failed"}
