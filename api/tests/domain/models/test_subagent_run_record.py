"""C4.1a PR-1 — SubagentRunRecord 域读模型测试（spec §3.2 + §7 PR-1）。"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.domain.models.subagent_run_record import SubagentRunRecord
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)


def _make_result() -> SubagentRunResult:
    return SubagentRunResult(
        worker_runtime_type=WorkerRuntimeType.LOCAL,
        lifecycle_state=WorkerLifecycleState.TERMINAL,
        terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        summary="done",
        parent_session_id="parent-1",
        child_session_id="child-1",
        source_ref="wu-1",
    )


class TestSubagentRunRecord:
    def test_construct_embeds_run(self) -> None:
        ts = datetime(2026, 7, 1, tzinfo=timezone.utc)
        rec = SubagentRunRecord(id="rec-1", created_at=ts, run=_make_result())
        assert rec.id == "rec-1"
        assert rec.created_at == ts
        assert rec.run.child_session_id == "child-1"
        assert rec.run.worker_runtime_type == WorkerRuntimeType.LOCAL
        assert rec.run.terminal_outcome == WorkerTerminalOutcome.SUCCESS

    def test_is_frozen(self) -> None:
        rec = SubagentRunRecord(
            id="rec-1",
            created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
            run=_make_result(),
        )
        with pytest.raises(ValidationError):
            rec.id = "rec-2"  # type: ignore[misc]

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SubagentRunRecord(
                id="rec-1",
                created_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
                run=_make_result(),
                bogus="x",
            )
