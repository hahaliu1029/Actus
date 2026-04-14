"""Tests for ExecutionMetrics."""

from app.domain.services.execution_metrics import ExecutionMetrics


class TestExecutionMetrics:
    def test_initial_state(self):
        m = ExecutionMetrics()
        assert m.tool_calls_total == 0
        assert m.tool_success_rate == 1.0
        assert m.avg_tool_latency_ms == 0.0
        assert m.avg_llm_latency_ms == 0.0

    def test_record_tool_call_success(self):
        m = ExecutionMetrics()
        m.record_tool_call(success=True, latency_ms=100.0)
        assert m.tool_calls_total == 1
        assert m.tool_calls_success == 1
        assert m.tool_calls_failed == 0
        assert m.tool_success_rate == 1.0
        assert m.avg_tool_latency_ms == 100.0

    def test_record_tool_call_failure(self):
        m = ExecutionMetrics()
        m.record_tool_call(success=False, latency_ms=50.0)
        assert m.tool_calls_total == 1
        assert m.tool_calls_failed == 1
        assert m.tool_success_rate == 0.0

    def test_mixed_tool_calls(self):
        m = ExecutionMetrics()
        m.record_tool_call(success=True, latency_ms=100.0)
        m.record_tool_call(success=True, latency_ms=200.0)
        m.record_tool_call(success=False, latency_ms=50.0)
        assert m.tool_calls_total == 3
        assert m.tool_success_rate == pytest.approx(2 / 3, rel=1e-3)
        assert m.avg_tool_latency_ms == pytest.approx(350 / 3, rel=1e-1)

    def test_record_llm_call(self):
        m = ExecutionMetrics()
        m.record_llm_call(latency_ms=500.0)
        m.record_llm_call(latency_ms=300.0)
        assert m.llm_calls_total == 2
        assert m.avg_llm_latency_ms == 400.0

    def test_to_dict(self):
        m = ExecutionMetrics()
        m.record_tool_call(success=True, latency_ms=100.0)
        m.record_llm_call(latency_ms=200.0)
        m.steps_completed = 3
        m.steps_failed = 1
        m.context_usage_ratio = 0.75
        m.compaction_count = 2

        d = m.to_dict()
        assert d["tool_success_rate"] == 1.0
        assert d["avg_tool_latency_ms"] == 100.0
        assert d["avg_llm_latency_ms"] == 200.0
        assert d["tool_calls_total"] == 1
        assert d["tool_calls_failed"] == 0
        assert d["llm_calls_total"] == 1
        assert d["steps_completed"] == 3
        assert d["steps_failed"] == 1
        assert d["context_usage_ratio"] == 0.75
        assert d["compaction_count"] == 2

    def test_to_dict_rounding(self):
        m = ExecutionMetrics()
        m.record_tool_call(success=True, latency_ms=100.123456)
        d = m.to_dict()
        assert d["avg_tool_latency_ms"] == 100.1  # rounded to 1 decimal


import pytest
