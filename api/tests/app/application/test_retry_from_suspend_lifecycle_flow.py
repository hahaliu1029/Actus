"""C7 PR7 — §12-4 收口：HTTP 语义入口 retry_from_suspend → claim → context 透传。
（真 HTTP/DB/沙箱链路标注「CI 验证」——本文件锁 application 编排层的可单测半段。）"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.agent_service import AgentService
from app.domain.models.lifecycle import RetryLifecycleContext
from app.domain.models.session import SandboxBindingState, SessionStatus


def _suspended_session():
    session = MagicMock()
    session.id = "sess-1"
    session.user_id = "user-1"
    session.status = SessionStatus.RUNNING
    session.execution_mode = "background"
    session.execution_phase = "suspended"
    session.retry_budget_remaining = 3
    session.expires_at = None
    session.suspended_reason = "bg_idle_timeout"
    session.sandbox_binding.state = SandboxBindingState.SUSPENDED
    return session


class _ClaimUoW:
    def __init__(self, claimed: tuple[int, int]) -> None:
        self.session = MagicMock()
        self.session.claim_background_retry_from_suspend = AsyncMock(return_value=claimed)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@asynccontextmanager
async def _mode_transition_fence():
    yield


@pytest.mark.asyncio
async def test_retry_flow_passes_typed_context_with_claimed_budget():
    svc = AgentService.__new__(AgentService)
    session = _suspended_session()
    svc._get_accessible_session = AsyncMock(return_value=session)
    svc._uow_factory = lambda: _ClaimUoW(claimed=(2, 1))
    svc._sandbox_lifecycle_service = MagicMock(resume=AsyncMock())
    svc._supervisor = MagicMock(resume=AsyncMock(return_value="rc-1"))
    svc._supervisor.mode_transition_fence = MagicMock(
        return_value=_mode_transition_fence()
    )
    svc._resume_task_with_handoff = AsyncMock()

    await svc.retry_from_suspend("sess-1", "user-1")

    svc._resume_task_with_handoff.assert_awaited_once()
    kwargs = svc._resume_task_with_handoff.await_args.kwargs
    ctx = kwargs["retry_lifecycle_context"]
    assert isinstance(ctx, RetryLifecycleContext)
    assert ctx.retry_budget_remaining == 2            # claim 后的持久剩余 → epoch=3-2=1
    assert ctx.trigger == "user" and ctx.previous_state == "suspended"


@pytest.mark.asyncio
async def test_retried_wire_payload_matches_fe_fixture():
    # 跨语言互锁：与 ui/src/lib/lifecycle/__tests__/wire-contract.test.ts 的 SAMPLE 字面一致
    import json
    from app.interfaces.schemas.event import EventMapper
    from tests.domain.services.test_agent_task_runner_lifecycle_hook import _Task
    from tests.domain.services.test_agent_task_runner_lifecycle_task_layer import _make_runner

    runner, task = _make_runner(), _Task()
    runner._retry_lifecycle_context = RetryLifecycleContext(retry_budget_remaining=2)
    runner._lifecycle_task_epoch = 1
    runner._session_id = "sess-1"
    await runner._emit_task_started_or_retried(task)

    from app.domain.models.event import Event
    from pydantic import TypeAdapter
    domain_ev = TypeAdapter(Event).validate_json(task.output_stream.events[0])
    sse = EventMapper.event_to_sse_event(domain_ev)
    assert sse.event == "lifecycle.task.retried"
    data = json.loads(sse.to_sse_data_json())
    assert data["epoch"] == 1
    assert data["reason"] == "retry_from_suspend"
    assert data["unit_id"] == "sess-1"
    assert data["detail"] == {
        "trigger": "user", "previous_state": "suspended",
        "retry_budget_remaining": 2, "original_outcome": None, "note": None,
    }
