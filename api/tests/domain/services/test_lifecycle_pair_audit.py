"""C7 PR7 — INV-C7-6：全源配对审计（不承诺相邻，配对靠 source_event_id/source_seq）
+ 并发 adversarial（两协程交错，per-source 配对仍完整）。"""
import asyncio
import json

import pytest

from app.domain.models.event import (
    DoneEvent, MessageEvent, PlanEvent, PlanEventStatus,
    StepEvent, StepEventStatus, ToolEvent, ToolEventStatus,
)
from app.domain.models.plan import Plan, Step

from tests.domain.services.test_agent_task_runner_lifecycle_hook import (
    _Task, _make_runner,
)

# §4 白名单源的脚本化序列（PR2-3 范围；subagent 配对已在 T14 锁）：
# (source_factory, 期望投影出的 lifecycle event 值列表)
SCRIPT = [
    (lambda: PlanEvent(plan=Plan(id="p1", steps=[Step(id="s1")]), status=PlanEventStatus.CREATED), ["started"]),
    (lambda: StepEvent(step=Step(id="s1"), status=StepEventStatus.STARTED), ["started"]),
    (lambda: ToolEvent(tool_call_id="tc1", tool_name="shell", function_name="shell_execute",
                       function_args={}, status=ToolEventStatus.CALLING), ["started"]),
    (lambda: MessageEvent(message="thinking"), []),
    (lambda: ToolEvent(tool_call_id="tc1", tool_name="shell", function_name="shell_execute",
                       function_args={},
                       artifact={"tool_call_id": "tc1", "tool_name": "shell",
                                 "tool_source": "native",
                                 "outcome": {"variant": "allow_success"}},
                       status=ToolEventStatus.CALLED), ["completed"]),
    (lambda: StepEvent(step=Step(id="s1", success=True), status=StepEventStatus.COMPLETED), ["completed"]),
    (lambda: PlanEvent(plan=Plan(id="p1"), status=PlanEventStatus.COMPLETED), ["completed"]),
    (lambda: DoneEvent(), []),
]


def _split(task: _Task):
    raw = [json.loads(r) for r in task.output_stream.events]
    return [e for e in raw if e["type"] != "lifecycle"], [e for e in raw if e["type"] == "lifecycle"]


@pytest.mark.asyncio
async def test_every_whitelisted_source_gets_paired_lifecycle():
    runner, task = _make_runner(True), _Task()
    for factory, _ in SCRIPT:
        await runner._put_and_add_event(task, factory())
    sources, lifecycles = _split(task)
    expected_total = sum(len(exp) for _, exp in SCRIPT)
    assert len(lifecycles) == expected_total          # 漏发/多发都在这里翻红（§11 pair-emit 风险）
    src_by_id = {s["id"]: s for s in sources if s.get("id")}
    for lc in lifecycles:
        src = src_by_id.get(lc["source_event_id"])
        assert src is not None, f"orphan lifecycle: {lc}"
        assert lc["source_seq"] == src["seq"]
        assert lc["seq"] > src["seq"]                  # 同 seq 空间且严格更大；相邻性不断言


@pytest.mark.asyncio
async def test_concurrent_emitters_keep_pairing_intact():
    # adversarial（R1#3/INV-C7-6）：两个协程各推一半事件，await 点交错；
    # 配对经 source_event_id 仍一一对应，且每对 lifecycle.seq > source.seq。
    runner, task = _make_runner(True), _Task()

    async def _pump(events):
        for ev in events:
            await runner._put_and_add_event(task, ev)
            await asyncio.sleep(0)                     # 强制让出，制造交错

    plans = [PlanEvent(plan=Plan(id=f"pa-{i}"), status=PlanEventStatus.CREATED) for i in range(5)]
    steps = [StepEvent(step=Step(id=f"sb-{i}"), status=StepEventStatus.STARTED) for i in range(5)]
    await asyncio.gather(_pump(plans), _pump(steps))

    sources, lifecycles = _split(task)
    assert len(lifecycles) == 10
    src_by_id = {s["id"]: s for s in sources}
    for lc in lifecycles:
        src = src_by_id[lc["source_event_id"]]
        assert lc["seq"] > src["seq"]
        # 语义配对正确（plan 源→plan lifecycle；step 源→step lifecycle）
        assert lc["lifecycle_type"] == src["type"]
