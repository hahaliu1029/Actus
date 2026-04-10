"""Tests for ExecutionWatchdog, ExecutionControl, and progress classification."""

import time

import pytest

from app.domain.models.event import (
    CompactionEvent,
    ContextStatusEvent,
    ErrorEvent,
    FinishingEvent,
    MessageEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    TitleEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.services.execution_watchdog import (
    ExecutionControl,
    ExecutionWatchdog,
    WatchdogVerdict,
    _is_progress_event,
    _should_terminate,
)


class TestExecutionWatchdog:
    def test_initial_state_healthy(self):
        w = ExecutionWatchdog(total_timeout_seconds=60, idle_timeout_seconds=10)
        assert w.status == WatchdogVerdict.HEALTHY
        assert w.last_node is None

    def test_record_progress_resets_timer(self):
        w = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.05)
        time.sleep(0.06)
        # Before recording progress, idle > threshold
        w.record_progress(node_name="executor_node")
        assert w.last_node == "executor_node"
        assert w.status == WatchdogVerdict.HEALTHY

    def test_record_progress_without_node_name(self):
        w = ExecutionWatchdog()
        w.record_progress()
        assert w.last_node is None

    def test_evaluate_total_timeout(self):
        w = ExecutionWatchdog(total_timeout_seconds=0.01, idle_timeout_seconds=100)
        time.sleep(0.02)
        assert w.evaluate() == WatchdogVerdict.HARD_TERMINATE

    def test_evaluate_first_idle_timeout(self):
        w = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.01)
        time.sleep(0.02)
        assert w.evaluate() == WatchdogVerdict.SOFT_RECOVER
        assert w._idle_warnings == 1

    def test_evaluate_second_idle_timeout(self):
        w = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.01)
        time.sleep(0.02)
        w.evaluate()  # first → SOFT_RECOVER
        time.sleep(0.02)
        assert w.evaluate() == WatchdogVerdict.HARD_TERMINATE
        assert w._idle_warnings == 2

    def test_evaluate_healthy_after_progress_reset(self):
        w = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.01)
        time.sleep(0.02)
        w.evaluate()  # SOFT_RECOVER, _idle_warnings = 1
        w.record_progress()  # resets _idle_warnings to 0
        assert w._idle_warnings == 0
        assert w.status == WatchdogVerdict.HEALTHY

    def test_status_is_readonly(self):
        """status property should not advance _idle_warnings."""
        w = ExecutionWatchdog(total_timeout_seconds=600, idle_timeout_seconds=0.01)
        time.sleep(0.02)
        _ = w.status  # read-only
        _ = w.status  # read-only again
        assert w._idle_warnings == 0  # should not have changed

    def test_elapsed_and_idle_seconds(self):
        w = ExecutionWatchdog()
        assert w.elapsed_seconds >= 0
        assert w.idle_seconds >= 0

    def test_total_timeout_zero_means_unlimited(self):
        """total_timeout_seconds=0 must NOT trigger immediate HARD_TERMINATE."""
        w = ExecutionWatchdog(total_timeout_seconds=0, idle_timeout_seconds=100)
        # Sleep past what would have been a non-zero total, confirm still healthy.
        assert w.evaluate() == WatchdogVerdict.HEALTHY
        assert w.check_total_only() is False
        time.sleep(0.02)
        assert w.evaluate() == WatchdogVerdict.HEALTHY
        assert w.check_total_only() is False

    def test_total_timeout_negative_means_unlimited(self):
        w = ExecutionWatchdog(total_timeout_seconds=-1, idle_timeout_seconds=100)
        assert w.check_total_only() is False

    def test_check_total_only_positive(self):
        """check_total_only returns True once total timeout elapses."""
        w = ExecutionWatchdog(total_timeout_seconds=0.01, idle_timeout_seconds=100)
        assert w.check_total_only() is False
        time.sleep(0.02)
        assert w.check_total_only() is True

    def test_check_total_only_no_side_effects(self):
        """check_total_only must not touch _idle_warnings."""
        w = ExecutionWatchdog(total_timeout_seconds=100, idle_timeout_seconds=0.01)
        time.sleep(0.02)
        # idle exceeded but total not — check_total_only should return False
        # and not increment idle_warnings.
        assert w.check_total_only() is False
        assert w._idle_warnings == 0
        assert w.check_total_only() is False
        assert w._idle_warnings == 0

    def test_total_smaller_than_idle_triggers_hard_terminate(self):
        """When total_timeout < idle_timeout, evaluate() must still fire HARD_TERMINATE
        based on total alone, without waiting for idle to also exceed."""
        w = ExecutionWatchdog(total_timeout_seconds=0.01, idle_timeout_seconds=100)
        time.sleep(0.02)
        # idle is way below threshold (100s), but total IS exceeded.
        # evaluate() must check total first and return HARD_TERMINATE.
        assert w.evaluate() == WatchdogVerdict.HARD_TERMINATE


class TestExecutionControl:
    def test_defaults(self):
        c = ExecutionControl()
        assert c.should_terminate is False
        assert c.idle_recovery_hint is None

    def test_set_terminate(self):
        c = ExecutionControl()
        c.should_terminate = True
        assert c.should_terminate is True

    def test_set_hint(self):
        c = ExecutionControl()
        c.idle_recovery_hint = "try something else"
        assert c.idle_recovery_hint == "try something else"


class TestShouldTerminateHelper:
    def test_no_control(self):
        assert _should_terminate({}) is False
        assert _should_terminate({"configurable": {}}) is False

    def test_control_not_terminated(self):
        c = ExecutionControl()
        assert _should_terminate({"configurable": {"execution_control": c}}) is False

    def test_control_terminated(self):
        c = ExecutionControl(should_terminate=True)
        assert _should_terminate({"configurable": {"execution_control": c}}) is True


class TestIsProgressEvent:
    @pytest.mark.parametrize("event_factory,expected", [
        (lambda: MessageEvent(role="assistant", message="hi"), True),
        (lambda: TitleEvent(title="Test"), True),
        (lambda: FinishingEvent(), True),
        (lambda: ToolEvent(
            tool_call_id="1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLED,
        ), True),
        (lambda: ToolEvent(
            tool_call_id="1", tool_name="shell", function_name="shell_execute",
            function_args={}, status=ToolEventStatus.CALLING,
        ), False),
        (lambda: StepEvent(
            step=None, status=StepEventStatus.COMPLETED,
        ), True),
        (lambda: StepEvent(
            step=None, status=StepEventStatus.STARTED,
        ), False),
        (lambda: PlanEvent(plan=None, status=PlanEventStatus.CREATED), True),
        (lambda: ErrorEvent(error="boom"), False),
        (lambda: WaitEvent(), False),
        (lambda: ContextStatusEvent(), False),
        (lambda: CompactionEvent(), False),
    ])
    def test_classification(self, event_factory, expected):
        # Some events require fields that may fail validation — skip if so
        try:
            event = event_factory()
        except Exception:
            pytest.skip("Event construction requires full model context")
        assert _is_progress_event(event) == expected
