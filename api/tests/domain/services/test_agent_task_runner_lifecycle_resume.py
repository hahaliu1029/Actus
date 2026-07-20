"""C7 PR4 — resume() 入口 task lifecycle：终态对齐 + postprocess-cancelled 负向（§4.4/§12-7c）。"""
import json
from unittest.mock import AsyncMock

import pytest

from app.domain.services.agent_task_runner import AgentTaskRunner

from tests.domain.services.test_agent_task_runner_lifecycle_hook import _Task
from tests.domain.services.test_agent_task_runner_lifecycle_task_layer import (
    _lifecycle_payloads, _make_runner,
)


class _FakeFlow:
    """空事件流 + 可配置 deferred 状态的最小 flow 替身。"""

    def __init__(self, deferred: bool) -> None:
        self._deferred_final_state = object() if deferred else None
        self._deferred_summaries = None

    def resume(self, command):
        async def _gen():
            if False:
                yield None
        return _gen()


def _prime_resume(runner: AgentTaskRunner, *, deferred: bool, postprocess_cancelled: bool = False):
    runner._flow = _FakeFlow(deferred)
    runner._require_state_machine = lambda: AsyncMock()
    runner._run_postprocess_or_cancel = AsyncMock(return_value=postprocess_cancelled)
    runner._build_compaction_events_if_any = lambda: []
    runner._snapshot_metrics = lambda: {}
    runner._set_terminal_status_with_notifications = AsyncMock()
    runner._was_timed_out = False


@pytest.mark.anyio
async def test_resume_normal_completion_emits_completed():
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=True)
    await runner.resume(task, command=None)
    events = [p["event"] for p in _lifecycle_payloads(task)]
    assert events == ["progress", "completed"]           # finishing → completed
    assert _lifecycle_payloads(task)[0]["reason"] == "finishing"


@pytest.mark.anyio
async def test_resume_timed_out_emits_failed_watchdog():
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=True)
    runner._was_timed_out = True
    await runner.resume(task, command=None)
    terminal = _lifecycle_payloads(task)[-1]
    assert (terminal["event"], terminal["reason"]) == ("failed", "watchdog_timeout")


@pytest.mark.anyio
async def test_resume_no_deferred_emits_completed():
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=False)
    await runner.resume(task, command=None)
    assert [p["event"] for p in _lifecycle_payloads(task)] == ["completed"]


@pytest.mark.anyio
async def test_resume_postprocess_cancelled_emits_no_terminal():
    # §12-7(c) 负向：回流 RUNNING，终态由后续 invoke() 覆盖
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=True, postprocess_cancelled=True)
    await runner.resume(task, command=None)
    events = [p["event"] for p in _lifecycle_payloads(task)]
    assert "completed" not in events and "failed" not in events and "cancelled" not in events
    assert events == ["progress"]                        # 只有 finishing progress


@pytest.mark.anyio
async def test_resume_postprocess_exception_emits_completed_postprocess_failed():
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=True)
    runner._run_postprocess_or_cancel = AsyncMock(side_effect=RuntimeError("pp boom"))
    await runner.resume(task, command=None)
    terminal = _lifecycle_payloads(task)[-1]
    assert (terminal["event"], terminal["reason"]) == ("completed", "postprocess_failed")


@pytest.mark.anyio
async def test_resume_generic_exception_emits_nothing_terminal():
    # R7：兜底 except 无终态状态写——lifecycle 镜像（零终态发射）
    runner, task = _make_runner(), _Task()
    _prime_resume(runner, deferred=False)

    class _BoomFlow(_FakeFlow):
        def resume(self, command):
            async def _gen():
                raise RuntimeError("flow boom")
                if False:
                    yield None
            return _gen()

    runner._flow = _BoomFlow(False)
    await runner.resume(task, command=None)
    assert all(p["event"] not in {"completed", "failed", "cancelled"} for p in _lifecycle_payloads(task))
