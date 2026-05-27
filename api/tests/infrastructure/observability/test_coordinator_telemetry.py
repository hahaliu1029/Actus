"""[C2 PR-6 §14.6] :class:`CoordinatorMetrics` — instrument bundle locks.

Pure unit tests on a synchronous constructor; no anyio / event loop
required. The :class:`MeterPort` Protocol is satisfied via
:class:`unittest.mock.MagicMock` so we can assert the exact
instrument-creation arguments without spinning up an OTel SDK.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from app.infrastructure.observability.coordinator_telemetry import (
    CoordinatorMetrics,
)


class TestCoordinatorMetrics:
    """Lock the four spec §14.6 instruments by name / unit / accessor."""

    def test_constructs_all_four_instruments(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        # Three counters + one histogram = four total instruments.
        assert meter.create_counter.call_count == 3
        assert meter.create_histogram.call_count == 1

    def test_run_cost_usd_instrument_name(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        names = [call.args[0] for call in meter.create_counter.call_args_list]
        assert "actus_coordinator_run_cost_usd" in names

    def test_run_cost_usd_unit_is_usd(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        for call in meter.create_counter.call_args_list:
            if call.args[0] == "actus_coordinator_run_cost_usd":
                assert call.kwargs.get("unit") == "usd"
                break
        else:  # pragma: no cover — defensive
            raise AssertionError("run_cost_usd counter not created")

    def test_tool_calls_instrument_name(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        names = [call.args[0] for call in meter.create_counter.call_args_list]
        assert "actus_coordinator_tool_calls" in names

    def test_budget_exhaustion_instrument_name(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        names = [call.args[0] for call in meter.create_counter.call_args_list]
        assert "actus_coordinator_budget_exhaustion_total" in names

    def test_duration_histogram_name_and_unit(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        meter.create_histogram.assert_called_once()
        call = meter.create_histogram.call_args
        assert call.args[0] == "actus_coordinator_duration_seconds"
        assert call.kwargs.get("unit") == "s"

    def test_all_counters_carry_description(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        for call in meter.create_counter.call_args_list:
            description = call.kwargs.get("description")
            assert description, (
                f"Counter {call.args[0]!r} missing description"
            )

    def test_histogram_carries_description(self) -> None:
        meter = MagicMock()
        CoordinatorMetrics(meter=meter)
        description = meter.create_histogram.call_args.kwargs.get("description")
        assert description, "duration_seconds histogram missing description"

    def test_accessors_return_instrument_objects(self) -> None:
        meter = MagicMock()
        counter_sentinel = MagicMock(name="counter")
        hist_sentinel = MagicMock(name="histogram")
        meter.create_counter.return_value = counter_sentinel
        meter.create_histogram.return_value = hist_sentinel
        cm = CoordinatorMetrics(meter=meter)
        # All three counter accessors return the same MagicMock return-value
        # (one mock shared across the three create_counter calls).
        assert cm.run_cost_usd is counter_sentinel
        assert cm.tool_calls is counter_sentinel
        assert cm.budget_exhaustion is counter_sentinel
        assert cm.duration_seconds is hist_sentinel
