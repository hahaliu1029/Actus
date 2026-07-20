"""C7 PR5 — INV-C7-5 双源终态 in-process 去重 + AND flag 组合 + query-count + §12-8 child 正向（spec §9/§8/§12-8）。"""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.domain.models.app_config import LifecycleRuntimeConfig
from app.domain.models.event import (
    CoordinatorReduceEvent,
    CoordinatorSiblingCancelEvent,
    PlanEvent,
    StepEvent,
    ToolEvent,
)
from app.domain.models.mailbox_envelope import CostAggregate, ResultReadyOutcome as RO
from app.domain.models.patch_apply_plan import GroupOutcome
from app.domain.models.plan import Plan, Step
from app.domain.services.lifecycle_projector import LifecycleProjector

from tests.domain.services.test_agent_task_runner_lifecycle_hook import (
    _FakeRedis, _FakeUoW, _Task, _make_runner,
)


def _wire_subagent(runner, *, master_on: bool, sub_on: bool):
    runner._lifecycle_runtime = LifecycleRuntimeConfig(
        lifecycle_events_enabled=master_on,
        lifecycle_subagent_events_enabled=sub_on,
    )
    runner._uow.session.child_rows = [
        SimpleNamespace(work_unit_id="wu-b", id="child-b"),
    ]
    runner._uow.session.lookup_calls = 0

    async def _lookup(run_id: str):
        runner._uow.session.lookup_calls += 1
        return {r.work_unit_id: r.id for r in runner._uow.session.child_rows}

    runner._lifecycle_projector = LifecycleProjector(
        parent_session_id="sess-1",
        subagent_enabled=lambda: runner._lifecycle_runtime.lifecycle_subagent_events_enabled,
        child_lookup=_lookup,
    )
    return runner


def _sibling_cancel():
    return CoordinatorSiblingCancelEvent(
        triggered_by_work_unit_id="wu-a", triggered_by_outcome=RO.FAILED,
        cancelled_work_unit_ids=["wu-b"], reason="fail_fast",
        coordinator_run_id="run-1", parent_session_id="sess-1",
    )


def _reduce_cancelled():
    return CoordinatorReduceEvent(
        group_outcome=GroupOutcome.CANCELLED, per_worker_outcomes={"wu-b": RO.CANCELLED},
        diagnostics_summary="", cost_total=CostAggregate(),
        coordinator_run_id="run-1", parent_session_id="sess-1",
    )


def _lifecycle(task):
    return [json.loads(r) for r in task.output_stream.events if json.loads(r)["type"] == "lifecycle"]


@pytest.mark.anyio
async def test_double_source_terminal_emitted_once():
    # §12-5：sibling-cancel（fail-fast 实时）与 reduce（批量时刻）双源同 child
    # 终态——emitter in-process 首发者胜（跨重启重复由前端 reducer 权威去重）
    runner, task = _wire_subagent(_make_runner(True), master_on=True, sub_on=True), _Task()
    await runner._put_and_add_event(task, _sibling_cancel())
    await runner._put_and_add_event(task, _reduce_cancelled())
    terminals = [p for p in _lifecycle(task) if p["unit_id"] == "child-b"]
    assert len(terminals) == 1
    assert (terminals[0]["event"], terminals[0]["reason"]) == ("cancelled", "sibling_cancel")
    assert runner._uow.session.lookup_calls == 2      # 每 source 一次批查（仍各查一次——去重在 emit 层）


@pytest.mark.anyio
async def test_master_off_subagent_on_emits_nothing():
    # R10#A9：master-off + subagent-on 必须零发射零批查
    runner, task = _wire_subagent(_make_runner(False), master_on=False, sub_on=True), _Task()
    await runner._put_and_add_event(task, _sibling_cancel())
    assert _lifecycle(task) == []
    assert runner._uow.session.lookup_calls == 0


@pytest.mark.anyio
async def test_master_on_subagent_off_emits_nothing_for_coordinator():
    runner, task = _wire_subagent(_make_runner(True), master_on=True, sub_on=False), _Task()
    await runner._put_and_add_event(task, _reduce_cancelled())
    assert _lifecycle(task) == []
    assert runner._uow.session.lookup_calls == 0


@pytest.mark.anyio
async def test_child_runner_emits_plan_step_tool_lifecycle():
    # §12-8 positive（R9#P2c 锁孤儿承诺）：child runner 的 plan/step/tool 照常投影。
    # root gate 只属 task 层 helper（T10）——若未来有人给咽喉 hook 加 root 门控，
    # 本测试变红（正是 spec §12-8 要防的「root gate 放得过高误抑制 child 全部投影」）。
    runner, task = _make_runner(True), _Task()
    runner._is_root_session = AsyncMock(return_value=False)   # 显式 child 身份
    await runner._put_and_add_event(task, PlanEvent(plan=Plan(id="plan-1")))
    await runner._put_and_add_event(task, StepEvent(step=Step(id="s1")))
    await runner._put_and_add_event(
        task,
        ToolEvent(tool_call_id="tc1", tool_name="shell",
                  function_name="shell_execute", function_args={}),
    )
    assert [(p["lifecycle_type"], p["event"]) for p in _lifecycle(task)] == [
        ("plan", "started"), ("step", "started"), ("tool", "started"),  # §4.3 CALLING→started
    ]
    runner._is_root_session.assert_not_awaited()   # hook 路径根本不查 root（结构佐证）
