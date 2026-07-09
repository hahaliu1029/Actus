"""C7 PR4 — task 层 runner 本地发射：helper 语义 + root gate + 分支布点（spec §4.4）。"""
import json
from unittest.mock import AsyncMock

import pytest

from app.domain.models.app_config import LifecycleRuntimeConfig
from app.domain.models.lifecycle import LifecycleEventKind as K
from app.domain.services.agent_task_runner import AgentTaskRunner

from tests.domain.services.test_agent_task_runner_lifecycle_hook import (
    _FakeRedis, _FakeUoW, _Task,
)


def _make_runner(*, flag_on: bool = True, is_root: bool = True) -> AgentTaskRunner:
    runner = AgentTaskRunner.__new__(AgentTaskRunner)
    runner._session_id = "sess-1"
    runner._uow = _FakeUoW()
    runner._event_seq_client = _FakeRedis()
    runner._event_seq_ttl_seconds = 3600
    runner._lifecycle_runtime = LifecycleRuntimeConfig(lifecycle_events_enabled=flag_on)
    runner._lifecycle_projector = None          # task 层不走投影器
    runner._lifecycle_terminal_emitted = set()
    runner._lifecycle_task_epoch = 0
    runner._lifecycle_postprocess_failed = False
    runner._retry_lifecycle_context = None
    runner._is_root_session = AsyncMock(return_value=is_root)
    return runner


def _lifecycle_payloads(task: _Task) -> list[dict]:
    return [json.loads(r) for r in task.output_stream.events if json.loads(r)["type"] == "lifecycle"]


class TestEmitHelper:
    @pytest.mark.asyncio
    async def test_root_emits_task_lifecycle(self):
        runner, task = _make_runner(), _Task()
        await runner._emit_task_lifecycle(task, K.STARTED)
        payloads = _lifecycle_payloads(task)
        assert [(p["lifecycle_type"], p["event"], p["state"], p["unit_id"]) for p in payloads] == [
            ("task", "started", "running", "sess-1")
        ]

    @pytest.mark.asyncio
    async def test_child_runner_never_emits_task_lifecycle(self):
        # §12-7(b) root-only gate：child 的执行观测归 subagent 层
        runner, task = _make_runner(is_root=False), _Task()
        await runner._emit_task_lifecycle(task, K.STARTED)
        await runner._emit_task_lifecycle(task, K.COMPLETED)
        assert _lifecycle_payloads(task) == []

    @pytest.mark.asyncio
    async def test_flag_off_zero_emission(self):
        runner, task = _make_runner(flag_on=False), _Task()
        await runner._emit_task_lifecycle(task, K.COMPLETED)
        assert task.output_stream.events == []          # INV-C7-3：连 root 查询都不做
        runner._is_root_session.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_terminal_dedup_in_process(self):
        runner, task = _make_runner(), _Task()
        await runner._emit_task_lifecycle(task, K.COMPLETED)
        await runner._emit_task_lifecycle(task, K.FAILED, reason="runner_error")
        payloads = _lifecycle_payloads(task)
        assert len(payloads) == 1 and payloads[0]["event"] == "completed"  # INV-C7-5 同 (unit,epoch) 终态首发者胜

    @pytest.mark.asyncio
    async def test_epoch_attached_to_all_task_events(self):
        runner, task = _make_runner(), _Task()
        runner._lifecycle_task_epoch = 1
        await runner._emit_task_lifecycle(task, K.PROGRESS, reason="finishing")
        assert _lifecycle_payloads(task)[0]["epoch"] == 1

    @pytest.mark.asyncio
    async def test_never_raises_on_internal_failure(self):
        runner, task = _make_runner(), _Task()
        runner._is_root_session = AsyncMock(side_effect=RuntimeError("db down"))
        await runner._emit_task_lifecycle(task, K.STARTED)   # 不得向调用分支泄漏异常
        assert _lifecycle_payloads(task) == []


class TestStartedRetriedMutex:
    @pytest.mark.asyncio
    async def test_no_context_emits_started(self):
        runner, task = _make_runner(), _Task()
        await runner._emit_task_started_or_retried(task)
        assert [p["event"] for p in _lifecycle_payloads(task)] == ["started"]

    @pytest.mark.asyncio
    async def test_context_emits_retried_and_suppresses_started(self):
        # §12-7(d)：retry 路径下 task.started 零发射（互斥）
        from app.domain.models.lifecycle import RetryLifecycleContext
        runner, task = _make_runner(), _Task()
        runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=2)
        runner._lifecycle_task_epoch = 1
        await runner._emit_task_started_or_retried(task)
        payloads = _lifecycle_payloads(task)
        assert [p["event"] for p in payloads] == ["retried"]
        assert payloads[0]["epoch"] == 1
        assert payloads[0]["reason"] == "retry_from_suspend"
        assert payloads[0]["detail"] == {
            "trigger": "user", "previous_state": "suspended",
            "retry_budget_remaining": 2, "original_outcome": None, "note": None,
        }
        assert runner._retry_lifecycle_context is None       # 消费一次即清
