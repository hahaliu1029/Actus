"""C7 PR7 — R10#A3：epoch 源自持久 budget 列，pod 重启不回退（禁内存计数的验收面）。"""
import json

import pytest

from app.domain.models.lifecycle import RETRY_BUDGET_INITIAL, RetryLifecycleContext

from tests.domain.services.test_agent_task_runner_lifecycle_hook import _Task
from tests.domain.services.test_agent_task_runner_lifecycle_task_layer import (
    _lifecycle_payloads, _make_runner,
)


def _epoch_for(remaining: int) -> int:
    return max(0, RETRY_BUDGET_INITIAL - remaining)


@pytest.mark.anyio
async def test_epoch_monotonic_across_simulated_restarts():
    # 「重启」= 全新 runner 实例；epoch 只依赖持久列 remaining（3→2→1）
    seen = []
    for remaining in (2, 1):
        runner, task = _make_runner(), _Task()   # fresh instance ≈ new pod
        runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=remaining)
        runner._lifecycle_task_epoch = _epoch_for(remaining)
        await runner._emit_task_started_or_retried(task)
        seen.append(_lifecycle_payloads(task)[0]["epoch"])
    assert seen == [1, 2]
    assert seen == sorted(seen)                   # 单调不回退


@pytest.mark.anyio
async def test_post_retry_followup_task_inherits_epoch_from_persistent_column():
    # retry 之后的普通 follow-up task（无 context）也携带当前 epoch（§5）
    runner, task = _make_runner(), _Task()
    runner._lifecycle_task_epoch = _epoch_for(2)  # _create_task 从 session 行派生
    await runner._emit_task_started_or_retried(task)   # 无 context → started
    payloads = _lifecycle_payloads(task)
    assert [p["event"] for p in payloads] == ["started"]
    assert payloads[0]["epoch"] == 1
