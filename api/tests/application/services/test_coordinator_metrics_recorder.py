"""[C2b rollout WS1b] Unit tests for CoordinatorMetricsRecorder + hash_user_id."""
from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from app.application.services.coordinator_metrics_recorder import (
    CoordinatorMetricsRecorder,
    hash_user_id,
)


def _fake_metrics() -> MagicMock:
    m = MagicMock()
    m.run_cost_usd = MagicMock()
    m.tool_calls = MagicMock()
    m.duration_seconds = MagicMock()
    return m


def test_hash_user_id_matches_convention() -> None:
    salt, uid = "s3cr3t", "user-123"
    expected = hashlib.sha256((salt + uid).encode("utf-8")).hexdigest()[:16]
    assert hash_user_id(uid, salt) == expected


def test_hash_user_id_empty_salt_returns_none() -> None:
    assert hash_user_id("user-123", "") is None


def test_hash_user_id_empty_user_returns_none() -> None:
    assert hash_user_id(None, "salt") is None
    assert hash_user_id("", "salt") is None


def test_record_run_terminal_authoritative_records_cost_and_duration() -> None:
    m = _fake_metrics()
    rec = CoordinatorMetricsRecorder(metrics=m, user_id_hash_salt="salt")
    rec.record_run_terminal(
        coordinator_run_id="run1", user_id="u1", cost_usd=1.25,
        outcome="success", cost_authoritative=True, duration_s=3.5,
    )
    m.run_cost_usd.add.assert_called_once()
    args, kwargs = m.run_cost_usd.add.call_args
    assert args[0] == 1.25
    attrs = kwargs["attributes"]
    assert attrs["coordinator_run_id"] == "run1"
    assert attrs["outcome"] == "success"
    assert attrs["user_id_hash"] == hash_user_id("u1", "salt")
    m.duration_seconds.record.assert_called_once()
    d_args, d_kwargs = m.duration_seconds.record.call_args
    assert d_args[0] == 3.5
    assert d_kwargs["attributes"]["group_outcome"] == "success"


def test_record_run_terminal_not_authoritative_skips_cost_keeps_duration() -> None:
    m = _fake_metrics()
    rec = CoordinatorMetricsRecorder(metrics=m, user_id_hash_salt="salt")
    rec.record_run_terminal(
        coordinator_run_id="run1", user_id="u1", cost_usd=0.0,
        outcome="incomplete", cost_authoritative=False, duration_s=2.0,
    )
    m.run_cost_usd.add.assert_not_called()  # no add(0) — masks "unknown" as "zero"
    m.duration_seconds.record.assert_called_once()


def test_record_run_terminal_empty_salt_omits_user_hash() -> None:
    m = _fake_metrics()
    rec = CoordinatorMetricsRecorder(metrics=m, user_id_hash_salt="")
    rec.record_run_terminal(
        coordinator_run_id="run1", user_id="u1", cost_usd=1.0,
        outcome="success", cost_authoritative=True, duration_s=None,
    )
    attrs = m.run_cost_usd.add.call_args.kwargs["attributes"]
    assert "user_id_hash" not in attrs
    m.duration_seconds.record.assert_not_called()  # duration_s=None


def test_record_tool_call_records_with_lineage() -> None:
    m = _fake_metrics()
    rec = CoordinatorMetricsRecorder(metrics=m, user_id_hash_salt="salt")
    rec.record_tool_call(coordinator_run_id="run1", work_unit_id="wu1", function_name="file_read")
    m.tool_calls.add.assert_called_once()
    args, kwargs = m.tool_calls.add.call_args
    assert args[0] == 1
    assert kwargs["attributes"] == {
        "coordinator_run_id": "run1", "work_unit_id": "wu1", "tool_name": "file_read",
    }


def test_instrument_exception_does_not_propagate() -> None:
    m = _fake_metrics()
    m.run_cost_usd.add.side_effect = RuntimeError("otel boom")
    m.duration_seconds.record.side_effect = RuntimeError("otel boom")
    m.tool_calls.add.side_effect = RuntimeError("otel boom")
    rec = CoordinatorMetricsRecorder(metrics=m, user_id_hash_salt="salt")
    # Must NOT raise (INV-B9 best-effort).
    rec.record_run_terminal(
        coordinator_run_id="r", user_id="u", cost_usd=1.0,
        outcome="success", cost_authoritative=True, duration_s=1.0,
    )
    rec.record_tool_call(coordinator_run_id="r", work_unit_id="w", function_name="t")


def test_none_metrics_is_noop() -> None:
    rec = CoordinatorMetricsRecorder(metrics=None, user_id_hash_salt="salt")
    rec.record_run_terminal(
        coordinator_run_id="r", user_id="u", cost_usd=1.0,
        outcome="success", cost_authoritative=True, duration_s=1.0,
    )
    rec.record_tool_call(coordinator_run_id="r", work_unit_id="w", function_name="t")
