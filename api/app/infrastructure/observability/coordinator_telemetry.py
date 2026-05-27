"""[C2 PR-6 §14.6] Coordinator OTel metrics.

Mirror of :class:`OtelLLMMetricsCallback` but scoped to the C2
coordinator run lifecycle. Each coordinator run can:

- accumulate USD cost (via budget enforcement callback)
- accumulate tool call counts (worker telemetry)
- record total wallclock duration on terminal
- emit a budget-exhaustion signal when ``StopReason.TOKEN_BUDGET`` or
  ``StopReason.WALLCLOCK_BUDGET`` fires

These instruments are stateless: recording happens at the call site
(orchestrator, supervisor, runner). This class just constructs the
instrument objects from a :class:`MeterPort` and exposes them as
typed attributes.

Wiring (``service_dependencies.py`` / PR-8) is out of scope for this
task — Task 6.7 only builds the instrument bundle.

Unit choice
-----------
``actus_coordinator_run_cost_usd`` uses ``unit="usd"`` rather than the
``usd_micro`` convention from :mod:`otel_llm_metrics` because the
coordinator aggregates already-converted budget figures (the budget
enforcement callback works in fractional USD), so emitting micro-USD
would require an upstream multiplier step that has no other use. Both
units are documented and the dashboard layer can convert if needed.
"""
from __future__ import annotations

from typing import Any

from app.domain.external.observability import MeterPort


class CoordinatorMetrics:
    """Coordinator-scoped OTel instrument bundle.

    Construct once per process (composition root) with the shared
    :class:`MeterPort`. Pass instruments to consumers (orchestrator /
    supervisor callbacks / runners) that need them.

    The four spec §14.6 instruments are:

    - ``actus_coordinator_run_cost_usd`` (Counter, ``usd``)
    - ``actus_coordinator_tool_calls`` (Counter, dimensionless)
    - ``actus_coordinator_duration_seconds`` (Histogram, ``s``)
    - ``actus_coordinator_budget_exhaustion_total`` (Counter,
      dimensionless)
    """

    def __init__(self, meter: MeterPort) -> None:
        self._run_cost_usd = meter.create_counter(
            "actus_coordinator_run_cost_usd",
            unit="usd",
            description=(
                "Cumulative USD cost across coordinator runs. Attributes: "
                "coordinator_run_id, user_id_hash, outcome."
            ),
        )
        self._tool_calls = meter.create_counter(
            "actus_coordinator_tool_calls",
            unit="1",
            description=(
                "Cumulative tool call count across coordinator children. "
                "Attributes: coordinator_run_id, work_unit_id, tool_name."
            ),
        )
        self._duration_seconds = meter.create_histogram(
            "actus_coordinator_duration_seconds",
            unit="s",
            description=(
                "Wallclock duration of a coordinator run from dispatch to "
                "terminal. Attributes: coordinator_run_id, group_outcome."
            ),
        )
        self._budget_exhaustion = meter.create_counter(
            "actus_coordinator_budget_exhaustion_total",
            unit="1",
            description=(
                "Coordinator child budget exhaustion events. Attributes: "
                "stop_reason (token_budget / wallclock_budget), "
                "coordinator_run_id, work_unit_id."
            ),
        )

    # ------------------------------------------------------------------
    # Accessors
    #
    # Typed as ``Any`` matching the :class:`MeterPort` contract; concrete
    # instrument objects are platform-specific — OTel ``Counter`` /
    # ``Histogram`` in production, no-op stubs in unit tests.
    # ------------------------------------------------------------------

    @property
    def run_cost_usd(self) -> Any:
        """Counter — cumulative USD cost across coordinator runs."""
        return self._run_cost_usd

    @property
    def tool_calls(self) -> Any:
        """Counter — cumulative tool call count across coordinator children."""
        return self._tool_calls

    @property
    def duration_seconds(self) -> Any:
        """Histogram — coordinator-run wallclock duration in seconds."""
        return self._duration_seconds

    @property
    def budget_exhaustion(self) -> Any:
        """Counter — coordinator child budget-exhaustion events."""
        return self._budget_exhaustion
