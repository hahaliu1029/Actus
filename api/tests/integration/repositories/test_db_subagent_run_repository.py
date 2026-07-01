"""C4.1a PR-2 — DbSubagentRunRepository 集成测试（需 PostgreSQL；spec §4/§7 PR-2）。

autouse `_migrate`（tests/integration/conftest.py）跑 alembic upgrade head →
建 subagent_runs 表。**集成未本地跑，CI 验证。**

Run:
    cd api && uv run pytest \
        tests/integration/repositories/test_db_subagent_run_repository.py -v
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.domain.models.mailbox_envelope import ArtifactRef, CostAggregate
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)
from app.infrastructure.repositories.db_subagent_run_repository import (
    DbSubagentRunRepository,
)

pytestmark = [pytest.mark.anyio, pytest.mark.integration]


@pytest.fixture
def repo(async_session_factory: async_sessionmaker) -> DbSubagentRunRepository:
    return DbSubagentRunRepository(session_factory=async_session_factory)


def _local_result(*, parent: str, child: str, **overrides) -> SubagentRunResult:
    base = dict(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        summary="ok",
        artifacts=[ArtifactRef(artifact_type="file", ref="/w/a.txt", description="d")],
        cost_summary=CostAggregate(
            total_input_tokens=10, total_output_tokens=20,
            total_usd=0.5, tool_call_count=2,
        ),
        cost_authoritative=True,
        parent_session_id=parent,
        child_session_id=child,
        source_ref="wu-1",
    )
    base.update(overrides)
    return SubagentRunResult(**base)


async def test_record_then_list_round_trip(repo):
    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    await repo.record(_local_result(parent=parent, child=child))

    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    rec = rows[0]
    assert rec.id
    assert rec.created_at is not None
    run = rec.run
    assert run.worker_runtime_type == WorkerRuntimeType.LOCAL
    assert run.terminal_outcome == WorkerTerminalOutcome.SUCCESS
    assert run.child_session_id == child
    assert run.cost_authoritative is True
    assert run.cost_summary is not None
    assert run.cost_summary.total_input_tokens == 10
    assert run.cost_summary.tool_call_count == 2
    assert len(run.artifacts) == 1
    assert run.artifacts[0].ref == "/w/a.txt"


async def test_record_is_idempotent_on_child_session_id(repo):
    # INV-C4.1-3：同 child_session_id 记两次 → 一行（redelivery swallow）。
    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    await repo.record(_local_result(parent=parent, child=child))
    await repo.record(_local_result(parent=parent, child=child, summary="second"))
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    assert rows[0].run.summary == "ok"  # 第一条留存，第二条被 swallow


async def test_research_result_cost_none_round_trip(repo):
    # research：cost_summary=None → 四 cost 列 NULL → 读回 cost_summary=None。
    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    r = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        summary="research answer",
        cost_summary=None,
        cost_authoritative=False,
        parent_session_id=parent,
        child_session_id=child,
        source_ref=child,
    )
    await repo.record(r)
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    assert rows[0].run.cost_summary is None
    assert rows[0].run.cost_authoritative is False


async def test_waiting_input_terminal_outcome_none_round_trip(repo):
    # 非终态：lifecycle=WAITING_INPUT / terminal_outcome=None round-trips（INV-C4-1）。
    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    r = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.WAITING_INPUT,
        terminal_outcome=None,
        parent_session_id=parent,
        child_session_id=child,
    )
    await repo.record(r)
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    assert rows[0].run.lifecycle_state == WorkerLifecycleState.WAITING_INPUT
    assert rows[0].run.terminal_outcome is None


async def test_summary_and_error_truncated_to_cap(repo):
    # R1#P3：超长 summary/error → 落库截断 → 读回 = 截断视图（post-cap round-trip）。
    from app.infrastructure.repositories.db_subagent_run_repository import (
        _MAX_SUBAGENT_RUN_SUMMARY_CHARS,
    )

    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    big = "x" * (_MAX_SUBAGENT_RUN_SUMMARY_CHARS + 500)
    r = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.FAILED,
        summary=big,
        error_summary=big,
        cost_authoritative=True,
        parent_session_id=parent,
        child_session_id=child,
    )
    await repo.record(r)
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    assert len(rows[0].run.summary) == _MAX_SUBAGENT_RUN_SUMMARY_CHARS
    assert len(rows[0].run.error_summary) == _MAX_SUBAGENT_RUN_SUMMARY_CHARS


async def test_artifacts_bounded_and_field_capped(repo):
    # R4#P3：artifacts 超量 + 超长 ref/description → 落库有界 → 读回 = 截断视图
    # （≤ _MAX_SUBAGENT_RUN_ARTIFACTS 条 + 每条 ref/description 截断）。
    from app.infrastructure.repositories.db_subagent_run_repository import (
        _MAX_SUBAGENT_RUN_ARTIFACTS,
        _MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS,
    )

    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    child = f"child-c41a-{uuid.uuid4().hex[:12]}"
    big = "r" * (_MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS + 100)
    many = [
        ArtifactRef(artifact_type="file", ref=big, description=big)
        for _ in range(_MAX_SUBAGENT_RUN_ARTIFACTS + 10)
    ]
    r = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        summary="ok",
        artifacts=many,
        cost_authoritative=True,
        parent_session_id=parent,
        child_session_id=child,
    )
    await repo.record(r)
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 1
    arts = rows[0].run.artifacts
    assert len(arts) == _MAX_SUBAGENT_RUN_ARTIFACTS
    assert all(len(a.ref) == _MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS for a in arts)
    assert all(
        len(a.description) == _MAX_SUBAGENT_RUN_ARTIFACT_FIELD_CHARS for a in arts
    )


async def test_remote_result_and_null_child_not_deduped(repo):
    # INV-C4.1-5：REMOTE result round-trip；child_session_id=None（多 NULL）不 dedup。
    parent = f"parent-c41a-{uuid.uuid4().hex[:12]}"
    remote = SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.REMOTE,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.UNKNOWN,
        summary="remote done",
        cost_summary=None,
        cost_authoritative=False,
        duration_seconds=1.5,
        duration_source="remote_reported",
        parent_session_id=parent,
        child_session_id=None,
        source_ref="a2a:agent-x",
    )
    await repo.record(remote)
    await repo.record(remote)  # child None → UNIQUE 容多 NULL → 两行
    rows = await repo.list_by_parent_session(parent)
    assert len(rows) == 2
    assert all(r.run.worker_runtime_type == WorkerRuntimeType.REMOTE for r in rows)
    assert rows[0].run.duration_source == "remote_reported"
    assert rows[0].run.child_session_id is None
