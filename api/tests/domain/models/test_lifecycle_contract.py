"""C7 PR1 — INV-C7-1 词表/派生表/受控 schema 契约测试（spec §2/§9）。"""
import pytest
from pydantic import ValidationError

from app.domain.models.lifecycle import (
    REASON_CODES,
    RETRY_BUDGET_INITIAL,
    STATE_FOR,
    SUPPORTED_EVENTS,
    TERMINAL_STATES,
    LifecycleCorrelationV1,
    LifecycleDetailV1,
    LifecycleEventKind,
    LifecycleState,
    LifecycleType,
    RetryLifecycleContext,
    is_terminal,
)

K = LifecycleEventKind
S = LifecycleState
T = LifecycleType

# spec §2 SUPPORTED_EVENTS 的字面复刻（测试与实现双写同一张表 = 抄错即红）
EXPECTED_SUPPORTED = {
    T.TASK: {K.STARTED, K.PROGRESS, K.COMPLETED, K.FAILED, K.CANCELLED, K.RETRIED},
    T.PLAN: {K.STARTED, K.PROGRESS, K.COMPLETED},
    T.STEP: {K.STARTED, K.COMPLETED, K.FAILED},
    T.TOOL: {K.STARTED, K.PROGRESS, K.COMPLETED, K.FAILED, K.CANCELLED},
    T.SUBAGENT: {K.STARTED, K.COMPLETED, K.FAILED, K.CANCELLED},
}

# spec §4 五表的 state 列字面复刻（INV-C7-1：STATE_FOR ↔ §4 一致性穷举）
EXPECTED_STATE_FOR = {
    (T.PLAN, K.STARTED): S.PENDING,      # §4.1 created→started/pending
    (T.PLAN, K.PROGRESS): S.RUNNING,
    (T.PLAN, K.COMPLETED): S.COMPLETED,
    (T.STEP, K.STARTED): S.RUNNING,      # §4.2
    (T.STEP, K.COMPLETED): S.COMPLETED,
    (T.STEP, K.FAILED): S.FAILED,
    (T.TOOL, K.STARTED): S.PENDING,      # §4.3 calling=排队
    (T.TOOL, K.PROGRESS): S.RUNNING,
    (T.TOOL, K.COMPLETED): S.COMPLETED,
    (T.TOOL, K.FAILED): S.FAILED,
    (T.TOOL, K.CANCELLED): S.CANCELLED,
    (T.TASK, K.STARTED): S.RUNNING,      # §4.4
    (T.TASK, K.PROGRESS): S.RUNNING,
    (T.TASK, K.COMPLETED): S.COMPLETED,
    (T.TASK, K.FAILED): S.FAILED,
    (T.TASK, K.CANCELLED): S.CANCELLED,
    (T.TASK, K.RETRIED): S.RUNNING,      # §5 retried 即 reopening edge
    (T.SUBAGENT, K.STARTED): S.RUNNING,  # §4.5 发射时行已 RUNNING-bound
    (T.SUBAGENT, K.COMPLETED): S.COMPLETED,
    (T.SUBAGENT, K.FAILED): S.FAILED,
    (T.SUBAGENT, K.CANCELLED): S.CANCELLED,
}


class TestVocabulary:
    def test_enum_member_counts_frozen(self):
        assert {m.value for m in T} == {"task", "plan", "step", "tool", "subagent"}
        assert {m.value for m in S} == {"pending", "running", "completed", "failed", "cancelled"}
        assert {m.value for m in K} == {"started", "progress", "completed", "failed", "cancelled", "retried"}

    def test_supported_events_matches_spec(self):
        assert {t: set(v) for t, v in SUPPORTED_EVENTS.items()} == EXPECTED_SUPPORTED
        assert sum(len(v) for v in SUPPORTED_EVENTS.values()) == 21

    def test_state_for_matches_spec_exhaustively(self):
        assert dict(STATE_FOR) == EXPECTED_STATE_FOR

    def test_state_for_keys_equal_supported_pairs(self):
        supported_pairs = {(t, k) for t, ks in SUPPORTED_EVENTS.items() for k in ks}
        assert set(STATE_FOR.keys()) == supported_pairs

    def test_terminal_states(self):
        assert TERMINAL_STATES == frozenset({S.COMPLETED, S.FAILED, S.CANCELLED})
        assert is_terminal(S.COMPLETED) and is_terminal(S.FAILED) and is_terminal(S.CANCELLED)
        assert not is_terminal(S.PENDING) and not is_terminal(S.RUNNING)

    def test_retry_budget_initial_matches_session_default(self):
        # R10#A3: epoch = RETRY_BUDGET_INITIAL - retry_budget_remaining（持久列）。
        # 常量必须与 Session 字段默认值一致，否则 epoch 起点漂移。
        from app.domain.models.session import Session
        assert Session.model_fields["retry_budget_remaining"].default == RETRY_BUDGET_INITIAL


class TestControlledSchemas:
    def test_detail_v1_rejects_unknown_fields(self):
        # 敏感信息禁入（spec §3.1）：白名单外字段一律 ValidationError
        with pytest.raises(ValidationError):
            LifecycleDetailV1(traceback="Exception: boom")
        with pytest.raises(ValidationError):
            LifecycleDetailV1(tool_args={"cmd": "rm -rf /"})
        with pytest.raises(ValidationError):
            LifecycleDetailV1(attempt=1)  # R12#P2: 无 attempt 字段——防 epoch 双源发散

    def test_detail_v1_whitelist(self):
        d = LifecycleDetailV1(
            trigger="user", previous_state="suspended",
            retry_budget_remaining=2, original_outcome="needs_authorization",
            note="plan_created_not_yet_executing",
        )
        assert d.retry_budget_remaining == 2

    def test_correlation_v1_rejects_unknown_fields(self):
        with pytest.raises(ValidationError):
            LifecycleCorrelationV1(objective="secret objective")

    def test_correlation_v1_whitelist(self):
        c = LifecycleCorrelationV1(
            work_unit_id="wu-1", coordinator_run_id="run-1", coordinator_attempt_ix=2,
        )
        assert c.coordinator_attempt_ix == 2

    def test_reason_codes_are_closed_vocabulary(self):
        # §4 各表 + §5 出现过的全部 reason code；自由文本禁止（helper 层校验，Task 2）
        assert REASON_CODES == frozenset({
            "plan_updated", "step_failed",
            "tool_error", "tool_timeout", "denied", "passthrough",
            "finishing", "waiting_confirmation", "watchdog_timeout",
            "user_cancel", "retry_from_suspend", "runner_error", "postprocess_failed",
            "worker_failed", "needs_authorization", "unknown_terminal_outcome",
            "waiting_unsupported", "sibling_cancel",
        })


class TestRetryLifecycleContext:
    def test_context_shape_and_defaults(self):
        ctx = RetryLifecycleContext(retry_budget_remaining=2)
        assert ctx.trigger == "user"
        assert ctx.previous_state == "suspended"
        assert ctx.retry_budget_remaining == 2
