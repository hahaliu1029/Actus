"""C4.1a PR-4 — SubagentResearchService._record_child_run seat 测试（spec §5.2 + §7 PR-4）。

隔离测 helper：project_research_result 投影正确 / repo=None no-op / record-raise
swallow。调用点「yield 之前」的顺序由 reviewer 对账 spec §5.2（见 plan 说明）。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.subagent_research_service import (
    ChildResult,
    SubagentResearchService,
)
from app.domain.models.subagent_worker import (
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)
from app.interfaces.schemas.subagent import ChildOutcome

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_deps():
    return {
        "session_service": MagicMock(),
        "agent_service": MagicMock(),
        "execution_supervisor": MagicMock(),
        "token_estimator": MagicMock(),
        "summary_llm": MagicMock(),
        "classifier": MagicMock(),
        "sandbox_lifecycle_service": MagicMock(),
        "quota_service": MagicMock(),
    }


def _child_result(**overrides) -> ChildResult:
    base = dict(
        child_id="ch-1",
        prompt="p",
        outcome=ChildOutcome.COMPLETED,
        final_answer="ans",
        transcript_tokens=1200,
        error_summary=None,
    )
    base.update(overrides)
    return ChildResult(**base)


async def test_record_projects_research_result(mock_deps):
    repo = AsyncMock()
    repo.record = AsyncMock(return_value=None)
    svc = SubagentResearchService(**mock_deps, subagent_run_repo=repo)

    await svc._record_child_run(_child_result(), "parent-1")

    repo.record.assert_awaited_once()
    run = repo.record.await_args.args[0]
    assert run.worker_runtime_type == WorkerRuntimeType.LOCAL
    assert run.lifecycle_state == WorkerLifecycleState.TERMINAL
    assert run.terminal_outcome == WorkerTerminalOutcome.SUCCESS
    assert run.summary == "ans"
    assert run.cost_summary is None
    assert run.cost_authoritative is False
    assert run.parent_session_id == "parent-1"
    assert run.child_session_id == "ch-1"
    assert run.source_ref == "ch-1"


async def test_record_waiting_maps_waiting_input(mock_deps):
    repo = AsyncMock()
    repo.record = AsyncMock(return_value=None)
    svc = SubagentResearchService(**mock_deps, subagent_run_repo=repo)

    await svc._record_child_run(
        _child_result(outcome=ChildOutcome.WAITING, final_answer=None), "parent-1",
    )

    run = repo.record.await_args.args[0]
    assert run.lifecycle_state == WorkerLifecycleState.WAITING_INPUT
    assert run.terminal_outcome is None


async def test_record_repo_none_is_noop(mock_deps):
    # flag OFF → subagent_run_repo=None → helper no-op（INV-C4.1-1）。
    svc = SubagentResearchService(**mock_deps)  # 不传 subagent_run_repo
    await svc._record_child_run(_child_result(), "parent-1")  # 不得抛


async def test_record_swallows_repo_error(mock_deps):
    # INV-C4.1-2：record raise → swallow（probe 续）。
    repo = AsyncMock()
    repo.record = AsyncMock(side_effect=RuntimeError("db down"))
    svc = SubagentResearchService(**mock_deps, subagent_run_repo=repo)

    await svc._record_child_run(
        _child_result(outcome=ChildOutcome.FAILED, final_answer=None,
                      error_summary="boom"),
        "parent-1",
    )  # 不得抛
    repo.record.assert_awaited_once()
