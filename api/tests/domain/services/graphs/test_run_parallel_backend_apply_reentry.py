"""C2 PR-7 Task 7.6 — _run_parallel_backend apply re-entry short-circuit tests.

Spec ref: §12.5 — when the subgraph's rehydrate path detected an
``already_applied`` audit row, the subgraph returns
``step_result_candidate = "ALREADY_APPLIED:{status}:{audit_id}"`` and
``goto=END``. main_graph must format the operator-facing message for each
status branch (success | rollback_partial | crash_mid_apply |
in_progress_recent) WITHOUT firing patch_applier.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.services.graphs.main_graph import _run_parallel_backend

pytestmark = pytest.mark.anyio


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


def _subgraph_already_applied(status: str, audit_id: int) -> MagicMock:
    """Mock subgraph returning an ALREADY_APPLIED step_result_candidate."""
    sub = MagicMock()
    sub.ainvoke = AsyncMock(return_value={
        "group_outcome": None,
        "apply_plan": None,
        "step_result_candidate": f"ALREADY_APPLIED:{status}:{audit_id}",
    })
    return sub


def _config_with_full_ports(subgraph: MagicMock) -> tuple[dict[str, Any], MagicMock]:
    """Wire all ports so we can assert applier is NEVER called even when present."""
    applier = MagicMock()
    applier.apply = AsyncMock()
    return {"configurable": {
        "parallel_execution_subgraph": subgraph,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}, applier


# ── 4 status branches ────────────────────────────────────────────────────


async def test_already_applied_success_returns_summary_no_apply() -> None:
    sub = _subgraph_already_applied("success", 42)
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "Apply already succeeded" in out
    assert "42" in out
    applier.apply.assert_not_called()


async def test_already_applied_rollback_partial_returns_no_retry_no_apply() -> None:
    sub = _subgraph_already_applied("rollback_partial", 7)
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "rollback_partial" in out
    assert "not auto-retrying" in out
    assert "7" in out
    applier.apply.assert_not_called()


async def test_already_applied_crash_mid_apply_warns_manual_recovery_no_apply() -> None:
    sub = _subgraph_already_applied("crash_mid_apply", 99)
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "crash" in out.lower() or "pod crash" in out
    assert "99" in out
    applier.apply.assert_not_called()


async def test_already_applied_in_progress_recent_waits_no_apply() -> None:
    sub = _subgraph_already_applied("in_progress_recent", 11)
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "Redis lock" in out or "lock" in out
    applier.apply.assert_not_called()


# ── Edge cases ──────────────────────────────────────────────────────────


async def test_already_applied_unknown_status_returns_verbatim() -> None:
    """Defensive: an unknown status sentinel falls through to verbatim return.

    Prevents a future status addition (in rehydrate_service) from silently
    breaking — operators see the raw sentinel in the step summary.
    """
    sub = _subgraph_already_applied("mystery_status", 5)
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert out == "ALREADY_APPLIED:mystery_status:5"
    applier.apply.assert_not_called()


async def test_malformed_already_applied_returns_unknown_audit_id() -> None:
    """If the sentinel is missing the audit_id segment, fall back to 'unknown'."""
    sub = MagicMock()
    sub.ainvoke = AsyncMock(return_value={
        "group_outcome": None,
        "apply_plan": None,
        "step_result_candidate": "ALREADY_APPLIED:success",
    })
    config, applier = _config_with_full_ports(sub)
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    assert "Apply already succeeded" in out
    assert "unknown" in out
    applier.apply.assert_not_called()


# ── Non-ALREADY_APPLIED passthrough ─────────────────────────────────────


async def test_non_already_applied_does_not_short_circuit() -> None:
    """Normal SUCCESS path with no ALREADY_APPLIED prefix → patch_applier branch runs.

    Regression guard: the new short-circuit must NOT eat regular step
    completions that happen to start with letters near 'A'.
    """
    from app.application.services.patch_applier import (
        ApplyDiagnostics, ApplyOutcome, ApplyStatus,
    )
    from app.domain.models.patch_apply_plan import GroupOutcome, PatchApplyPlan
    from app.domain.models.patch_manifest import FilePatchEntry
    import hashlib
    sha = hashlib.sha256(b"x").hexdigest()
    plan = PatchApplyPlan(
        coordinator_run_id="r1",
        files=(FilePatchEntry(
            path="f.py", op="add", new_digest=sha,
            content_ref="r", content_size=1,
        ),),
        total_size_bytes=1, file_count=1,
        source_work_unit_ids=("wu1",),
    )
    sub = MagicMock()
    sub.ainvoke = AsyncMock(return_value={
        "group_outcome": GroupOutcome.SUCCESS,
        "apply_plan": plan,
        "step_result_candidate": "并行执行完成：1 个 worker 完成。",
    })
    applier = MagicMock()
    applier.apply = AsyncMock(return_value=ApplyOutcome(
        status=ApplyStatus.SUCCESS,
        applied_files=(), failed_at=None, rollback_status=None,
        diagnostics=ApplyDiagnostics(duration_ms=1),
    ))
    config = {"configurable": {
        "parallel_execution_subgraph": sub,
        "patch_applier": applier,
        "parent_sandbox": AsyncMock(),
        "artifact_storage": AsyncMock(),
    }}
    out = (await _run_parallel_backend(_state(), config, _step())).summary
    applier.apply.assert_awaited_once()
    assert "应用成功" in out


# ── Structured success flag for the ALREADY_APPLIED short-circuit ─────────


@pytest.mark.parametrize(
    "status, expected_success",
    [
        ("success", True),
        ("rollback_partial", False),
        ("crash_mid_apply", False),
        ("in_progress_recent", False),
        ("mystery_status", False),
    ],
)
async def test_already_applied_outcome_success_flag(
    status: str, expected_success: bool,
) -> None:
    sub = _subgraph_already_applied(status, 1)
    config, _applier = _config_with_full_ports(sub)
    outcome = await _run_parallel_backend(_state(), config, _step())
    assert outcome.success is expected_success
