"""C2 PR-5 Task 5.8 — _run_parallel_backend apply-wiring tests.

Spec ref: §10.6 (orchestrator branch into PatchApplier on SUCCESS).

Coverage matrix:
- Non-SUCCESS group_outcome → no applier call; return reducer text
- SUCCESS + empty plan (exploration-only) → no applier call
- SUCCESS + applier ports missing (PR-5 cold-code) → no applier call;
  defensive fallback returns reducer text
- SUCCESS + plan + ports wired → applier.apply called; status branches
  (SUCCESS / ROLLBACK_PARTIAL / other) all hit the right return text
"""
from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.application.services.patch_applier import (
    ApplyDiagnostics,
    ApplyOutcome,
    AppliedFileFailure,
    AppliedFileRecord,
    ApplyStatus,
)
from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
from app.domain.models.patch_manifest import FilePatchEntry
from app.domain.services.graphs.main_graph import _run_parallel_backend

pytestmark = pytest.mark.anyio


_SHA_A = hashlib.sha256(b"a").hexdigest()


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _step() -> MagicMock:
    s = MagicMock()
    s.id = "step1"
    s.parallel_work_units = MagicMock()
    s.parallel_work_units.work_units = []
    return s


def _state() -> dict[str, Any]:
    return {"session_id": "sess1", "user_id": "u1", "root_session_id": "sess1"}


def _plan(file_count: int = 1) -> PatchApplyPlan:
    files = tuple(
        FilePatchEntry(
            path=f"d/f{i}.py", op="add",
            new_digest=_SHA_A, content_ref=f"r{i}", content_size=1,
        ) for i in range(file_count)
    )
    return PatchApplyPlan(
        coordinator_run_id="r1",
        files=files,
        total_size_bytes=file_count,
        file_count=file_count,
        source_work_unit_ids=("wu1",),
    )


def _subgraph_with_final(final: dict[str, Any]) -> MagicMock:
    sub = MagicMock()
    sub.ainvoke = AsyncMock(return_value=final)
    return sub


async def test_failed_group_outcome_skips_apply() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.FAILED,
        "apply_plan": None,
        "step_result_candidate": "并行执行失败：N 个 worker 中 K 个失败。",
    })
    applier = MagicMock()
    applier.apply = AsyncMock()
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "并行执行失败" in out
    applier.apply.assert_not_called()


async def test_success_empty_plan_skips_apply() -> None:
    """Exploration-only step (SUCCESS with no file ops) → no apply."""
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(file_count=0),
        "step_result_candidate": "并行执行完成：1 个 worker 完成，合并写入 0 个文件。",
    })
    applier = MagicMock()
    applier.apply = AsyncMock()
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "并行执行完成" in out
    applier.apply.assert_not_called()


async def test_success_with_plan_but_no_applier_port_skips_apply() -> None:
    """PR-5 cold-code: composition root hasn't wired patch_applier yet.
    The function must gracefully fall back to returning the reducer
    text without crashing."""
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "并行执行完成：1 个 worker 完成，合并写入 1 个文件。",
    })
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "并行执行完成" in out


async def test_success_with_applier_returns_success_text() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(file_count=2),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.SUCCESS,
        applied_files=(
            AppliedFileRecord(path="f0.py", op="add"),
            AppliedFileRecord(path="f1.py", op="add"),
        ),
        failed_at=None,
        rollback_status=None,
        diagnostics=ApplyDiagnostics(duration_ms=10),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    applier.apply.assert_awaited_once()
    assert "reducer-text" in out
    # [codex R4 P2] Apply outcome text is Chinese per project
    # convention; PR-6/PR-8 route via ApplyStatus enum, not strings.
    assert "应用成功" in out
    assert "2 个文件" in out


async def test_apply_rollback_partial_surfaces_failed_path() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.ROLLBACK_PARTIAL,
        applied_files=(),
        failed_at=AppliedFileFailure(path="bad.py", reason="disk full"),
        rollback_status="partial",
        diagnostics=ApplyDiagnostics(duration_ms=20),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    # [codex R4 P2] Operator text is Chinese; PR-6/PR-8 route via
    # ApplyStatus enum, not text matching.
    assert "应用回滚不完整" in out
    assert "bad.py" in out


async def test_apply_receives_cancel_event_from_config() -> None:
    """[codex R2 P1] _run_parallel_backend MUST thread
    ``config['configurable']['cancel_event']`` into ``applier.apply()``.

    Without this, the applier's cancel contract — which catches a
    parent-cancel between reducer and apply and aborts with rollback —
    only fires in unit tests; production main_graph would continue
    writing the parent sandbox after the parent had been cancelled.
    """
    import asyncio
    cancel = asyncio.Event()
    cancel.set()
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.APPLY_ABORTED,
        applied_files=(),
        failed_at=None,
        rollback_status=None,
        diagnostics=ApplyDiagnostics(duration_ms=1),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
        "cancel_event": cancel,
    }}
    await _run_parallel_backend(_state(), config, _step())
    apply_kwargs = applier.apply.await_args.kwargs
    assert apply_kwargs.get("cancel_event") is cancel


async def test_apply_other_failure_returns_status_text() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.DIGEST_DRIFT,
        applied_files=(),
        failed_at=AppliedFileFailure(path="x.py", reason="drift"),
        rollback_status="complete",
        diagnostics=ApplyDiagnostics(duration_ms=5),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "digest_drift" in out.lower()


# ── Structured success flag (NOT inferred from the summary string) ────────


async def test_outcome_success_false_on_failed_group() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.FAILED,
        "apply_plan": None,
        "step_result_candidate": "并行执行失败。",
    })
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is False


async def test_outcome_success_true_on_success_exploration_only() -> None:
    """SUCCESS with no files (exploration-only) → success, no apply needed."""
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(file_count=0),
        "step_result_candidate": "并行探索完成。",
    })
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is True


async def test_outcome_success_true_on_apply_success() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.SUCCESS,
        applied_files=(AppliedFileRecord(path="f0.py", op="add"),),
        failed_at=None, rollback_status=None,
        diagnostics=ApplyDiagnostics(duration_ms=1),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is True


async def test_outcome_success_false_on_apply_rollback_partial() -> None:
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.ROLLBACK_PARTIAL,
        applied_files=(),
        failed_at=AppliedFileFailure(path="bad.py", reason="disk full"),
        rollback_status="partial",
        diagnostics=ApplyDiagnostics(duration_ms=1),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is False


async def test_outcome_success_true_when_applier_ports_missing() -> None:
    """SUCCESS with files but the composition root hasn't wired the applier
    ports (PR-5 cold-code / misconfig) → the reducer's SUCCESS is the
    authoritative work signal, so success=True (the skipped apply is logged
    as a WARNING and never happens on the flag-ON production path)."""
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "并行执行完成：1 个 worker 完成，合并写入 1 个文件。",
    })
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is True


async def test_dispatch_path_contract_error_surfaces_as_failed_outcome() -> None:
    """[single-path contract — graceful surfacing] A planner-proposed bare /
    workspace-root path raises ``CoordinatorPathContractError`` from
    dispatch_node. ``_run_parallel_backend`` catches it and returns a graceful
    failed ``ParallelBackendOutcome`` — NOT an uncaught raise that would crash
    the whole agent run (executor_node has no try/except around the call)."""
    from app.domain.models.path_validation import CoordinatorPathContractError
    subgraph = MagicMock()
    subgraph.ainvoke = AsyncMock(
        side_effect=CoordinatorPathContractError(
            "bare filename rejected (no directory component): 'part_a.md'"
        )
    )
    config = {"configurable": {"parallel_execution_subgraph": subgraph}}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is False
    assert "part_a.md" in outcome.summary


async def test_outcome_success_false_on_apply_other_status() -> None:
    """Apply ends in a non-SUCCESS / non-ROLLBACK_PARTIAL status (e.g.
    DIGEST_DRIFT) → step failed."""
    subgraph = _subgraph_with_final({
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": _plan(),
        "step_result_candidate": "reducer-text",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.DIGEST_DRIFT,
        applied_files=(),
        failed_at=AppliedFileFailure(path="x.py", reason="drift"),
        rollback_status="complete",
        diagnostics=ApplyDiagnostics(duration_ms=1),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is False
