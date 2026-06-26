"""C4 PR-2 — subagent_worker_projection 投影 + 漂移测试（spec §6 + §7 PR-2）。"""
from __future__ import annotations

from typing import Optional

from app.application.services.subagent_worker_projection import (
    project_local_result,
    project_research_result,
)
from app.interfaces.schemas.subagent import ChildOutcome
from app.domain.models.mailbox_envelope import (
    ArtifactRef,
    CostAggregate,
    ResultReadyOutcome,
    ResultReadyPayload,
)
from app.domain.models.needs_authorization_details import NeedsAuthorizationDetails
from app.domain.models.subagent_worker import (
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerTerminalOutcome,
)


def _make_local_payload(outcome: ResultReadyOutcome) -> ResultReadyPayload:
    """构造合法 ResultReadyPayload（NEEDS_AUTHORIZATION 必带 details）。"""
    if outcome == ResultReadyOutcome.NEEDS_AUTHORIZATION:
        return ResultReadyPayload(
            summary="s",
            outcome=outcome,
            needs_authorization_details=NeedsAuthorizationDetails(
                reason="out_of_tool_allowlist",
                requested_tool="shell_execute",
            ),
        )
    return ResultReadyPayload(summary="s", outcome=outcome)


class TestProjectLocalResult:
    def test_covers_all_result_ready_outcomes(self) -> None:
        """漂移 + 精确映射双守卫。

        (1) 仍 `for outcome in ResultReadyOutcome` 迭代——新增未映射值 → KeyError（红）。
        (2) `_EXPECTED_LOCAL_OUTCOME` 是 **测试侧独立 oracle**（NOT 读实现的
            `_LOCAL_OUTCOME_MAP`）；逐成员断言 terminal_outcome 等于期望，
            把 map 抄错（如 FAILED→SUCCESS）也抓得到（不再 vacuous）。
        oracle 自身穷尽性：先断言 outcome 在期望表里，缺一个成员也红。
        """
        # 测试侧期望（手抄 spec §6.1，与实现 _LOCAL_OUTCOME_MAP 独立维护）
        _EXPECTED_LOCAL_OUTCOME: dict[ResultReadyOutcome, WorkerTerminalOutcome] = {
            ResultReadyOutcome.SUCCESS: WorkerTerminalOutcome.SUCCESS,
            ResultReadyOutcome.FAILED: WorkerTerminalOutcome.FAILED,
            ResultReadyOutcome.CANCELLED: WorkerTerminalOutcome.CANCELLED,
            ResultReadyOutcome.TIMED_OUT: WorkerTerminalOutcome.TIMED_OUT,
            ResultReadyOutcome.NEEDS_AUTHORIZATION: WorkerTerminalOutcome.NEEDS_AUTHORIZATION,
        }
        for outcome in ResultReadyOutcome:
            assert outcome in _EXPECTED_LOCAL_OUTCOME, (
                f"oracle 缺成员 {outcome!r}：补全测试侧期望表（保持 oracle 穷尽）"
            )
            payload = _make_local_payload(outcome)
            result = project_local_result(
                payload,
                parent_session_id="p",
                child_session_id="c",
                work_unit_id="wu-1",
            )
            # LOCAL 投影：全成员均 TERMINAL（spec §6.1）
            assert result.lifecycle_state == WorkerLifecycleState.TERMINAL
            # 精确映射：逐成员对账独立期望，挡 map 转抄错位
            assert result.terminal_outcome == _EXPECTED_LOCAL_OUTCOME[outcome]
            assert result.worker_runtime_type == WorkerRuntimeType.LOCAL

    def test_success_maps_fields(self) -> None:
        payload = ResultReadyPayload(
            summary="done",
            outcome=ResultReadyOutcome.SUCCESS,
            artifacts=[ArtifactRef(artifact_type="file", ref="/x", description="d")],
            cost_summary=CostAggregate(total_usd=0.5, total_output_tokens=10),
        )
        result = project_local_result(
            payload,
            parent_session_id="p",
            child_session_id="c",
            work_unit_id="wu-1",
        )
        assert result.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert result.summary == "done"
        assert result.cost_authoritative is True
        assert result.cost_summary is not None
        assert result.cost_summary.total_usd == 0.5
        assert len(result.artifacts) == 1
        assert result.artifacts[0].ref == "/x"
        assert result.duration_source == "unavailable"
        assert result.source_ref == "wu-1"
        assert result.parent_session_id == "p"
        assert result.child_session_id == "c"

    def test_needs_authorization_formats_error_summary(self) -> None:
        payload = _make_local_payload(ResultReadyOutcome.NEEDS_AUTHORIZATION)
        result = project_local_result(
            payload,
            parent_session_id="p",
            child_session_id="c",
            work_unit_id="wu-1",
        )
        assert result.terminal_outcome == WorkerTerminalOutcome.NEEDS_AUTHORIZATION
        assert (
            result.error_summary
            == "needs_authorization: out_of_tool_allowlist (tool=shell_execute)"
        )

    def test_never_pending_or_running(self) -> None:
        """INV-C4-4 producer-side：投影永不产出 PENDING/RUNNING。"""
        for outcome in ResultReadyOutcome:
            result = project_local_result(
                _make_local_payload(outcome),
                parent_session_id="p",
                child_session_id="c",
                work_unit_id="wu-1",
            )
            assert result.lifecycle_state not in {
                WorkerLifecycleState.PENDING,
                WorkerLifecycleState.RUNNING,
            }


class TestProjectResearchResult:
    def _project(self, outcome: ChildOutcome):
        return project_research_result(
            child_id="ch-1",
            outcome=outcome,
            final_answer="ans",
            transcript_tokens=1234,
            error_summary=None,
            parent_session_id="p",
            child_session_id="c",
        )

    def test_covers_all_child_outcomes(self) -> None:
        """漂移 + 精确映射双守卫。

        (1) 仍 `for outcome in ChildOutcome` 迭代——新增未映射值 → KeyError（红）。
        (2) `_EXPECTED_RESEARCH_OUTCOME` 是 **测试侧独立 oracle**（NOT 读实现的
            `_RESEARCH_OUTCOME_MAP`）；逐成员断言 (lifecycle_state, terminal_outcome)
            等于期望，把 map 抄错（如 WAITING→TERMINAL 或 FAILED→SUCCESS）也抓得到。
        oracle 自身穷尽性：先断言 outcome 在期望表里，缺一个成员也红。
        """
        # 测试侧期望（手抄 spec §6.2，与实现 _RESEARCH_OUTCOME_MAP 独立维护）
        _EXPECTED_RESEARCH_OUTCOME: dict[
            ChildOutcome, tuple[WorkerLifecycleState, Optional[WorkerTerminalOutcome]]
        ] = {
            ChildOutcome.COMPLETED: (
                WorkerLifecycleState.TERMINAL,
                WorkerTerminalOutcome.SUCCESS,
            ),
            ChildOutcome.FAILED: (
                WorkerLifecycleState.TERMINAL,
                WorkerTerminalOutcome.FAILED,
            ),
            ChildOutcome.TIMED_OUT: (
                WorkerLifecycleState.TERMINAL,
                WorkerTerminalOutcome.TIMED_OUT,
            ),
            ChildOutcome.CANCELLED: (
                WorkerLifecycleState.TERMINAL,
                WorkerTerminalOutcome.CANCELLED,
            ),
            ChildOutcome.WAITING: (WorkerLifecycleState.WAITING_INPUT, None),
        }
        for outcome in ChildOutcome:
            assert outcome in _EXPECTED_RESEARCH_OUTCOME, (
                f"oracle 缺成员 {outcome!r}：补全测试侧期望表（保持 oracle 穷尽）"
            )
            result = self._project(outcome)
            assert result.worker_runtime_type == WorkerRuntimeType.LOCAL
            assert result.source_ref == "ch-1"
            # 精确映射：逐成员对账 (lifecycle, terminal)，挡 map 转抄错位
            expected_lifecycle, expected_terminal = _EXPECTED_RESEARCH_OUTCOME[outcome]
            assert result.lifecycle_state == expected_lifecycle
            assert result.terminal_outcome == expected_terminal

    def test_completed_maps_to_success(self) -> None:
        result = self._project(ChildOutcome.COMPLETED)
        assert result.lifecycle_state == WorkerLifecycleState.TERMINAL
        assert result.terminal_outcome == WorkerTerminalOutcome.SUCCESS
        assert result.summary == "ans"

    def test_waiting_maps_to_waiting_input(self) -> None:
        result = self._project(ChildOutcome.WAITING)
        assert result.lifecycle_state == WorkerLifecycleState.WAITING_INPUT
        assert result.terminal_outcome is None

    def test_cost_is_unknown_not_zero(self) -> None:
        """transcript_tokens 不折入 cost；cost_summary=None。"""
        result = self._project(ChildOutcome.COMPLETED)
        assert result.cost_summary is None
        assert result.cost_authoritative is False
        assert result.duration_source == "unavailable"

    def test_final_answer_none_yields_empty_summary(self) -> None:
        result = project_research_result(
            child_id="ch-1",
            outcome=ChildOutcome.FAILED,
            final_answer=None,
            transcript_tokens=0,
            error_summary="boom",
            parent_session_id="p",
            child_session_id="c",
        )
        assert result.summary == ""
        assert result.error_summary == "boom"

    def test_never_pending_or_running(self) -> None:
        for outcome in ChildOutcome:
            result = self._project(outcome)
            assert result.lifecycle_state not in {
                WorkerLifecycleState.PENDING,
                WorkerLifecycleState.RUNNING,
            }


class TestReusedTypeShapeDrift:
    """spec §6.3：复用类型字段集一变即红（强制对账）。"""

    def test_artifact_ref_shape_unchanged(self) -> None:
        assert set(ArtifactRef.model_fields) == {"artifact_type", "ref", "description"}

    def test_cost_aggregate_shape_unchanged(self) -> None:
        assert set(CostAggregate.model_fields) == {
            "total_input_tokens",
            "total_output_tokens",
            "total_usd",
            "tool_call_count",
        }
