"""Coordinator perf-sampling CLI — scrape /api/v1/metrics and report p50/p95
duration + total cost + tool-call distribution for a coordinator perf session.

Usage:
    ACTUS_METRICS_ENDPOINT_TOKEN=<tok> uv run python -m app.cli.coordinator_perf_sample \
        --runs 10 --base-url http://localhost:8000

Thin CLI client — NO app/ behavior change. venv-only. Requires (on the target
API): ACTUS_C2_COORDINATOR_ENABLED=true, ACTUS_METRICS_ENDPOINT_TOKEN set (else
/api/v1/metrics 404s), and a real provider in api/config.yaml (the planner must
emit parallel_work_units — WS0 makes that possible). Driving the N sessions is
the operator's job per the runbook; this CLI focuses on the scrape + report.
See docs/runbooks/c2-coordinator-canary-rollback.md for thresholds.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys

import httpx

_METRIC_LINE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)\{[^}]*\}\s+(?P<value>[-+0-9.eE]+)\s*$')


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m app.cli.coordinator_perf_sample",
        description="Scrape /api/v1/metrics and report coordinator perf metrics.",
    )
    p.add_argument("--runs", type=int, default=10, help="number of coordinator runs (N>=10)")
    p.add_argument("--base-url", default="http://localhost:8000", help="target API base URL")
    p.add_argument("--metrics-url", default=None, help="override; defaults to <base-url>/api/v1/metrics")
    p.add_argument("--metrics-token", default=None, help="bearer token; defaults to $ACTUS_METRICS_ENDPOINT_TOKEN")
    args = p.parse_args(argv)
    if args.metrics_url is None:
        args.metrics_url = args.base_url.rstrip("/") + "/api/v1/metrics"
    if args.metrics_token is None:
        args.metrics_token = os.environ.get("ACTUS_METRICS_ENDPOINT_TOKEN", "")
    return args


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _parse_metric_samples(prom_text: str, metric_name: str) -> list[float]:
    """Extract all sample values for a named metric from Prometheus exposition text."""
    out: list[float] = []
    for line in prom_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = _METRIC_LINE.match(line)
        if m and m.group("name") == metric_name:
            out.append(float(m.group("value")))
    return out


async def _scrape(metrics_url: str, token: str) -> str:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(metrics_url, headers=headers)
        resp.raise_for_status()
        return resp.text


async def _run(args: argparse.Namespace) -> dict:
    prom_text = await _scrape(args.metrics_url, args.metrics_token)
    durations = _parse_metric_samples(prom_text, "actus_coordinator_duration_seconds_sum")
    costs = _parse_metric_samples(prom_text, "actus_coordinator_run_cost_usd_total")
    tool_calls = _parse_metric_samples(prom_text, "actus_coordinator_tool_calls_total")
    return {
        "runs_requested": args.runs,
        "duration_p50": _percentile(durations, 50),
        "duration_p95": _percentile(durations, 95),
        "total_cost_usd": sum(costs),
        "total_tool_calls": sum(tool_calls),
        "sample_counts": {
            "duration": len(durations),
            "cost": len(costs),
            "tool_calls": len(tool_calls),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = asyncio.run(_run(args))
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
