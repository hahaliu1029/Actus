"""C4.1a PR-1 — SubagentRunRepository ABC 契约测试（spec §3.3 + §7 PR-1）。"""
from __future__ import annotations

import inspect

import pytest

from app.domain.models.subagent_run_record import SubagentRunRecord
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)
from app.domain.repositories.subagent_run_repository import SubagentRunRepository


def test_repo_is_abstract() -> None:
    with pytest.raises(TypeError):
        SubagentRunRepository()  # type: ignore[abstract]


def test_record_is_coroutine_abstractmethod() -> None:
    assert getattr(SubagentRunRepository.record, "__isabstractmethod__", False) is True
    assert inspect.iscoroutinefunction(SubagentRunRepository.record)


def test_list_is_coroutine_abstractmethod() -> None:
    assert (
        getattr(SubagentRunRepository.list_by_parent_session, "__isabstractmethod__", False)
        is True
    )
    assert inspect.iscoroutinefunction(SubagentRunRepository.list_by_parent_session)


@pytest.mark.asyncio
async def test_concrete_subclass_instantiable_and_runs() -> None:
    recorded: list[SubagentRunResult] = []

    class _Fake(SubagentRunRepository):
        async def record(self, result: SubagentRunResult) -> None:
            recorded.append(result)

        async def list_by_parent_session(
            self, parent_session_id: str
        ) -> list[SubagentRunRecord]:
            return []

    repo = _Fake()
    assert isinstance(repo, SubagentRunRepository)
    result = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        parent_session_id="p",
        child_session_id="c",
    )
    await repo.record(result)
    assert recorded == [result]
    assert await repo.list_by_parent_session("p") == []
