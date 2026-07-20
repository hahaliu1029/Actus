"""C7 PR2 — §12-1：flag-off 下咽喉输出与 pre-C7 行为 byte-identical。"""
import json

import pytest

from app.domain.models.event import (
    DoneEvent, ErrorEvent, MessageEvent, PlanEvent, PlanEventStatus,
    StepEvent, StepEventStatus,
)
from app.domain.models.plan import Plan, Step

# 复用 Task 7 的 fakes（同目录 import）
from tests.domain.services.test_agent_task_runner_lifecycle_hook import (
    _Task, _make_runner,
)


@pytest.mark.anyio
async def test_flag_off_stream_is_sources_verbatim():
    runner, task = _make_runner(False), _Task()
    sources = [
        PlanEvent(plan=Plan(id="p1", steps=[Step(id="s1")]), status=PlanEventStatus.CREATED),
        StepEvent(step=Step(id="s1"), status=StepEventStatus.STARTED),
        MessageEvent(message="hello"),
        StepEvent(step=Step(id="s1", success=True), status=StepEventStatus.COMPLETED),
        PlanEvent(plan=Plan(id="p1"), status=PlanEventStatus.COMPLETED),
        ErrorEvent(error="x"),
        DoneEvent(),
    ]
    for ev in sources:
        await runner._put_and_add_event(task, ev)
    assert len(task.output_stream.events) == len(sources)
    for raw, src in zip(task.output_stream.events, sources):
        assert json.loads(raw)["type"] == src.type
    assert all(json.loads(r)["type"] != "lifecycle" for r in task.output_stream.events)


@pytest.mark.anyio
async def test_flag_on_smoke_projects_plan_and_step_only():
    runner, task = _make_runner(True), _Task()
    await runner._put_and_add_event(task, MessageEvent(message="hi"))
    await runner._put_and_add_event(
        task, StepEvent(step=Step(id="s1", success=False), status=StepEventStatus.COMPLETED)
    )
    types = [json.loads(r)["type"] for r in task.output_stream.events]
    assert types == ["message", "step", "lifecycle"]
    lc = json.loads(task.output_stream.events[-1])
    assert (lc["lifecycle_type"], lc["event"], lc["reason"]) == ("step", "failed", "step_failed")
