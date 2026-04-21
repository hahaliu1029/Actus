"""N1 — ExecutionMetrics extension for AST validator aggregation.

Per spec §8.2: parser_failure_rate denominator MUST come from an
application-level counter (cannot be estimated from CALLED events
because Asked path returns before CALLED emission).

Per spec §6.5: Layer-2 validator crash is a distinct P0 signal —
MUST NOT be conflated with parse_failed (which is Layer-0 normal
fail-closed result). Separate counter + separate to_dict() output.
"""
from __future__ import annotations

import pytest

from app.domain.services.execution_metrics import ExecutionMetrics


def test_record_ast_validation_ok_increments_total_only():
    m = ExecutionMetrics()
    m.record_ast_validation("ok")
    assert m.ast_validations_total == 1
    assert m.ast_validations_parse_failed == 0
    assert m.ast_validations_denied == 0
    assert m.ast_validator_crashes == 0


def test_record_ast_validation_parse_failed():
    m = ExecutionMetrics()
    m.record_ast_validation("parse_failed")
    assert m.ast_validations_total == 1
    assert m.ast_validations_parse_failed == 1
    assert m.ast_validations_denied == 0
    assert m.ast_validator_crashes == 0


def test_record_ast_validation_denied():
    m = ExecutionMetrics()
    m.record_ast_validation("fs_destructive")
    assert m.ast_validations_total == 1
    assert m.ast_validations_parse_failed == 0
    assert m.ast_validations_denied == 1


def test_record_ast_validator_crash_is_separate_signal():
    """P1-b: Layer-2 crash (validate() raised despite I-N1.1) is
    a DISTINCT counter, NOT conflated with parse_failed.
    """
    m = ExecutionMetrics()
    m.record_ast_validator_crash()
    assert m.ast_validator_crashes == 1
    # Crash does NOT inflate parser_failure_rate
    assert m.ast_validations_parse_failed == 0
    assert m.ast_validations_total == 0
    assert m.parser_failure_rate == 0.0


def test_parser_failure_rate_property():
    m = ExecutionMetrics()
    for _ in range(9):
        m.record_ast_validation("ok")
    m.record_ast_validation("parse_failed")
    assert m.parser_failure_rate == pytest.approx(0.1, abs=0.01)


def test_to_dict_includes_ast_counters():
    """P1-a: new AST counters must surface via to_dict() so they reach
    DoneEvent / HealthEvent (AgentTaskRunner._snapshot_metrics() route).
    """
    m = ExecutionMetrics()
    m.record_ast_validation("ok")
    m.record_ast_validation("fs_destructive")
    m.record_ast_validation("parse_failed")
    m.record_ast_validator_crash()

    d = m.to_dict()
    assert d["ast_validations_total"] == 3
    assert d["ast_validations_denied"] == 1
    assert d["ast_validations_parse_failed"] == 1
    assert d["ast_validator_crashes"] == 1
    assert d["parser_failure_rate"] == pytest.approx(1.0 / 3, abs=0.01)
