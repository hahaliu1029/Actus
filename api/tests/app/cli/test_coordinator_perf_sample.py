"""[C2b rollout WS4] Unit tests for the coordinator perf-sampling CLI helpers."""
from __future__ import annotations

from app.cli.coordinator_perf_sample import (
    _parse_args,
    _percentile,
    _parse_metric_samples,
)


def test_parse_args_defaults() -> None:
    args = _parse_args(["--runs", "10"])
    assert args.runs == 10
    assert args.metrics_url.endswith("/api/v1/metrics")


def test_parse_args_metrics_token_flag() -> None:
    args = _parse_args(["--metrics-token", "tok"])
    assert args.metrics_token == "tok"


def test_percentile() -> None:
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert 5.0 <= _percentile(data, 50) <= 5.5
    assert _percentile(data, 95) >= 9.0


def test_parse_metric_samples_extracts_named_metric() -> None:
    text = (
        "# HELP actus_coordinator_run_cost_usd_total ...\n"
        "# TYPE actus_coordinator_run_cost_usd_total counter\n"
        'actus_coordinator_run_cost_usd_total{coordinator_run_id="r1",outcome="success"} 0.5\n'
        'actus_coordinator_run_cost_usd_total{coordinator_run_id="r2",outcome="success"} 0.7\n'
    )
    samples = _parse_metric_samples(text, "actus_coordinator_run_cost_usd_total")
    assert sorted(samples) == [0.5, 0.7]
