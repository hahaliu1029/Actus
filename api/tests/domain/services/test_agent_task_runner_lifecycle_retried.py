"""C7 PR4 — retried 首事件 + epoch 持久派生 + started 互斥负向（spec §5/§12-4/§12-7d）。"""
import json

import pytest

from app.domain.models.lifecycle import RetryLifecycleContext

from tests.domain.services.test_agent_task_runner_lifecycle_hook import _Task
from tests.domain.services.test_agent_task_runner_lifecycle_task_layer import (
    _lifecycle_payloads, _make_runner,
)


@pytest.mark.anyio
async def test_retried_first_event_with_epoch_and_no_started():
    runner, task = _make_runner(), _Task()
    runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=2)
    runner._lifecycle_task_epoch = 1                      # 3 - 2（_create_task 派生）
    await runner._emit_task_started_or_retried(task)
    payloads = _lifecycle_payloads(task)
    assert [p["event"] for p in payloads] == ["retried"]  # §12-7(d): started 零发射
    assert payloads[0]["epoch"] == 1
    assert payloads[0]["state"] == "running"              # retried 即 reopening edge（§5）


@pytest.mark.anyio
async def test_second_retry_epoch_two():
    runner, task = _make_runner(), _Task()
    runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=1)
    runner._lifecycle_task_epoch = 2
    await runner._emit_task_started_or_retried(task)
    assert _lifecycle_payloads(task)[0]["epoch"] == 2


@pytest.mark.anyio
async def test_post_retry_terminal_carries_current_epoch():
    from app.domain.models.lifecycle import LifecycleEventKind as K
    runner, task = _make_runner(), _Task()
    runner._lifecycle_task_epoch = 1
    await runner._emit_task_lifecycle(task, K.COMPLETED)
    assert _lifecycle_payloads(task)[0]["epoch"] == 1


@pytest.mark.anyio
async def test_flag_off_context_consumed_but_nothing_emitted():
    # R7#P3c：flag-off 下 context 传了无人消费——wire 零差异，context 仍被清
    runner, task = _make_runner(flag_on=False), _Task()
    runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=2)
    await runner._emit_task_started_or_retried(task)
    assert task.output_stream.events == []
    assert runner._retry_lifecycle_context is None
