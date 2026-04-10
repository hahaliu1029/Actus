"""Execution metrics collector for a single task session.

Tracks tool/LLM call counts, latencies (running sum), and context usage.
Emitted with HealthEvent and DoneEvent for observability.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutionMetrics:
    """In-memory metrics for one task execution session.

    Lives on ``PlannerReActFlow`` to accumulate across invoke/resume.
    """

    tool_calls_total: int = 0
    tool_calls_success: int = 0
    tool_calls_failed: int = 0
    _tool_latency_sum_ms: float = 0.0
    llm_calls_total: int = 0
    _llm_latency_sum_ms: float = 0.0
    steps_completed: int = 0
    steps_failed: int = 0
    context_usage_ratio: float = 0.0
    compaction_count: int = 0

    def record_tool_call(self, success: bool, latency_ms: float) -> None:
        self.tool_calls_total += 1
        if success:
            self.tool_calls_success += 1
        else:
            self.tool_calls_failed += 1
        self._tool_latency_sum_ms += latency_ms

    def record_llm_call(self, latency_ms: float) -> None:
        self.llm_calls_total += 1
        self._llm_latency_sum_ms += latency_ms

    @property
    def tool_success_rate(self) -> float:
        if self.tool_calls_total == 0:
            return 1.0
        return self.tool_calls_success / self.tool_calls_total

    @property
    def avg_tool_latency_ms(self) -> float:
        if self.tool_calls_total == 0:
            return 0.0
        return self._tool_latency_sum_ms / self.tool_calls_total

    @property
    def avg_llm_latency_ms(self) -> float:
        if self.llm_calls_total == 0:
            return 0.0
        return self._llm_latency_sum_ms / self.llm_calls_total

    def to_dict(self) -> dict:
        return {
            "tool_success_rate": round(self.tool_success_rate, 3),
            "avg_tool_latency_ms": round(self.avg_tool_latency_ms, 1),
            "avg_llm_latency_ms": round(self.avg_llm_latency_ms, 1),
            "tool_calls_total": self.tool_calls_total,
            "tool_calls_failed": self.tool_calls_failed,
            "llm_calls_total": self.llm_calls_total,
            "steps_completed": self.steps_completed,
            "steps_failed": self.steps_failed,
            "context_usage_ratio": round(self.context_usage_ratio, 3),
            "compaction_count": self.compaction_count,
        }
