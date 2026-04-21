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
    # N1 AST validator counters
    ast_validations_total: int = 0        # denominator for parser_failure_rate
    ast_validations_parse_failed: int = 0  # Layer-0 parse_failed results only
    ast_validations_denied: int = 0        # non-ok, non-parse_failed denials
    ast_validator_crashes: int = 0         # Layer-2 crash events (distinct — see spec §6.5)

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

    def record_ast_validation(self, code: str) -> None:
        """N1 — record outcome of a shell AST validate() call.

        ``code`` is ``ValidationResult.code``: one of
        {ok, fs_destructive, process_control, network_exfil, system_admin,
         cwd_boundary, parse_failed, oversized_command}.

        Per spec §6.5: Layer-2 crash is NOT a parse_failed — use
        :meth:`record_ast_validator_crash` for those.
        """
        self.ast_validations_total += 1
        if code == "parse_failed":
            self.ast_validations_parse_failed += 1
        elif code != "ok":
            self.ast_validations_denied += 1

    def record_ast_validator_crash(self) -> None:
        """N1 — record a Layer-2 call-site crash (validate() raised despite I-N1.1).

        This is a distinct P0 signal (spec §6.5). It is NOT folded into
        ``parser_failure_rate``: crashes don't increment
        ``ast_validations_total`` or ``ast_validations_parse_failed``.
        CI/telemetry dashboards must alert on non-zero
        ``ast_validator_crashes``.
        """
        self.ast_validator_crashes += 1

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

    @property
    def parser_failure_rate(self) -> float:
        if self.ast_validations_total == 0:
            return 0.0
        return self.ast_validations_parse_failed / self.ast_validations_total

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
            # N1 AST validator counters — surface via DoneEvent/HealthEvent
            # per AgentTaskRunner._snapshot_metrics (agent_task_runner.py:2371)
            "ast_validations_total": self.ast_validations_total,
            "ast_validations_parse_failed": self.ast_validations_parse_failed,
            "ast_validations_denied": self.ast_validations_denied,
            "ast_validator_crashes": self.ast_validator_crashes,
            "parser_failure_rate": round(self.parser_failure_rate, 3),
        }
