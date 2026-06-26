"""C4 PR-1 — subagent_worker DTO 不变式测试（spec §3 + §7 PR-1）。"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models.subagent_worker import (
    WorkerRuntimeType,
    WorkerSpec,
)
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerTerminalOutcome,
)


class TestWorkerSpecRuntimeMatrix:
    """INV-C4-2：per-runtime 字段矩阵（spec §3.3）。"""

    def test_local_requires_child_session_id_and_forbids_remote_target(self) -> None:
        spec = WorkerSpec(
            worker_runtime_type=WorkerRuntimeType.LOCAL,
            parent_session_id="p",
            child_session_id="c",
            objective="do work",
        )
        assert spec.worker_runtime_type == WorkerRuntimeType.LOCAL
        assert spec.child_session_id == "c"
        assert spec.remote_target is None

    def test_local_missing_child_session_id_rejected(self) -> None:
        with pytest.raises(ValidationError, match="LOCAL worker requires child_session_id"):
            WorkerSpec(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                parent_session_id="p",
                objective="do work",
            )

    def test_local_with_remote_target_rejected(self) -> None:
        with pytest.raises(ValidationError, match="forbids remote_target"):
            WorkerSpec(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                parent_session_id="p",
                child_session_id="c",
                objective="do work",
                remote_target="agent-1",
            )

    def test_remote_requires_remote_target(self) -> None:
        spec = WorkerSpec(
            worker_runtime_type=WorkerRuntimeType.REMOTE,
            parent_session_id="p",
            objective="do work",
            remote_target="agent-1",
        )
        assert spec.remote_target == "agent-1"
        assert spec.child_session_id is None  # REMOTE: child optional

    def test_remote_missing_remote_target_rejected(self) -> None:
        with pytest.raises(ValidationError, match="REMOTE worker requires remote_target"):
            WorkerSpec(
                worker_runtime_type=WorkerRuntimeType.REMOTE,
                parent_session_id="p",
                objective="do work",
            )

    def test_skill_reserved_constructible_without_remote_target(self) -> None:
        # SKILL 是保留值：可构造（前向兼容），仅 forbid remote_target。
        spec_with_child = WorkerSpec(
            worker_runtime_type=WorkerRuntimeType.SKILL,
            parent_session_id="p",
            child_session_id="c",
            objective="do work",
        )
        assert spec_with_child.worker_runtime_type == WorkerRuntimeType.SKILL
        spec_no_child = WorkerSpec(
            worker_runtime_type=WorkerRuntimeType.SKILL,
            parent_session_id="p",
            objective="do work",
        )
        assert spec_no_child.child_session_id is None

    def test_skill_with_remote_target_rejected(self) -> None:
        with pytest.raises(ValidationError, match="forbids remote_target"):
            WorkerSpec(
                worker_runtime_type=WorkerRuntimeType.SKILL,
                parent_session_id="p",
                objective="do work",
                remote_target="agent-1",
            )

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            WorkerSpec(
                worker_runtime_type=WorkerRuntimeType.REMOTE,
                parent_session_id="p",
                objective="do work",
                remote_target="agent-1",
                bogus="x",
            )


class TestEnumValues:
    def test_runtime_values(self) -> None:
        assert WorkerRuntimeType.LOCAL.value == "local"
        assert WorkerRuntimeType.REMOTE.value == "remote"
        assert WorkerRuntimeType.SKILL.value == "skill"


class TestSubagentRunResultInvariant:
    """INV-C4-1：terminal_outcome is not None ⟺ lifecycle_state == TERMINAL。"""

    def test_terminal_with_outcome_ok(self) -> None:
        r = SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType.LOCAL,
            lifecycle_state=WorkerLifecycleState.TERMINAL,
            terminal_outcome=WorkerTerminalOutcome.SUCCESS,
        )
        assert r.terminal_outcome == WorkerTerminalOutcome.SUCCESS

    def test_waiting_input_without_outcome_ok(self) -> None:
        r = SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType.LOCAL,
            lifecycle_state=WorkerLifecycleState.WAITING_INPUT,
        )
        assert r.terminal_outcome is None

    def test_terminal_without_outcome_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lifecycle_state=TERMINAL requires terminal_outcome"):
            SubagentRunResult(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                lifecycle_state=WorkerLifecycleState.TERMINAL,
            )

    def test_outcome_without_terminal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="terminal_outcome may only be set"):
            SubagentRunResult(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                lifecycle_state=WorkerLifecycleState.WAITING_INPUT,
                terminal_outcome=WorkerTerminalOutcome.SUCCESS,
            )

    def test_defaults_are_honest_unknown(self) -> None:
        r = SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType.REMOTE,
            lifecycle_state=WorkerLifecycleState.TERMINAL,
            terminal_outcome=WorkerTerminalOutcome.UNKNOWN,
        )
        assert r.summary == ""
        assert r.artifacts == []
        assert r.cost_summary is None
        assert r.cost_authoritative is False
        assert r.duration_seconds is None
        assert r.duration_source == "unavailable"
        assert r.source_ref is None

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(ValidationError):
            SubagentRunResult(
                worker_runtime_type=WorkerRuntimeType.LOCAL,
                lifecycle_state=WorkerLifecycleState.WAITING_INPUT,
                bogus="x",
            )
