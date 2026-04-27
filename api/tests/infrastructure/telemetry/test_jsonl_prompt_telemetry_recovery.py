"""PR-3 T59: JsonlPromptTelemetry persists RecoveryEvent to disk."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.domain.services.provider_profiles._base import ErrorClass
from app.domain.services.recovery._event import RecoveryEvent
from app.infrastructure.telemetry.prompt_telemetry import JsonlPromptTelemetry


def test_T59_jsonl_telemetry_writes_recovery_event_jsonl(tmp_path: Path):
    """Round 22 P1 #1: real production port writes recovery_event.jsonl.

    Locks the contract that `_TelemetryProbe`-only tests cannot reach —
    proves PromptTelemetryPort.emit_recovery_event actually persists to
    disk via the same _append(...) path used for assembly.jsonl /
    llm_invocation.jsonl.
    """
    telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

    event = RecoveryEvent(
        call_id="test-call-uuid",
        attempt_index=1,
        provider_id="dashscope_qwen",
        api_mode="chat_completions",
        model_name="qwen-max-2025",
        error_class=ErrorClass.COMPAT_QUIRK,
        fingerprint_code="json_mode_with_thinking",
        action_code="strip_response_format",
        rewrite_applied_keys=("response_format",),
        outcome="retry_sent",
        latency_ms=147,
    )

    telemetry.emit_recovery_event(event)

    log_file = tmp_path / "recovery_event.jsonl"
    assert log_file.exists(), (
        "Round 22 P1 #1 regression: emit_recovery_event must append to "
        "recovery_event.jsonl. If this file is missing, the JsonlPromptTelemetry "
        "implementation reverted to no-op and B2 telemetry is silently dropped "
        "in production."
    )

    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, f"expected exactly 1 line; got {len(lines)}: {lines!r}"

    record = json.loads(lines[0])
    assert record["call_id"] == "test-call-uuid"
    assert record["attempt_index"] == 1
    assert record["provider_id"] == "dashscope_qwen"
    assert record["api_mode"] == "chat_completions"
    assert record["model_name"] == "qwen-max-2025"
    assert record["error_class"] == "compat_quirk"  # ErrorClass.value
    assert record["fingerprint_code"] == "json_mode_with_thinking"
    assert record["action_code"] == "strip_response_format"
    assert record["rewrite_applied_keys"] == ["response_format"]
    assert record["outcome"] == "retry_sent"
    assert record["latency_ms"] == 147
    assert "ts" in record


def test_T59b_jsonl_telemetry_recovery_event_with_none_optional_fields(tmp_path: Path):
    """Round 23 P3 #3: cover both None patterns the dataclass allows.

    (a) success outcome — error_class=None, fingerprint_code=None,
        action_code is the LAST applied action.
    (b) rule_missed / budget_exhausted outcome — action_code=None.
    """
    telemetry = JsonlPromptTelemetry(log_dir=tmp_path)

    success_event = RecoveryEvent(
        call_id="success-call",
        attempt_index=1,
        provider_id="openai_official",
        api_mode="chat_completions",
        model_name="gpt-4o",
        error_class=None,
        fingerprint_code=None,
        action_code="trigger_recompact",
        rewrite_applied_keys=(),
        outcome="success",
        latency_ms=850,
    )
    telemetry.emit_recovery_event(success_event)

    rule_missed_event = RecoveryEvent(
        call_id="missed-call",
        attempt_index=0,
        provider_id="generic_openai",
        api_mode="chat_completions",
        model_name="gpt-4o",
        error_class=ErrorClass.UNKNOWN,
        fingerprint_code=None,
        action_code=None,
        rewrite_applied_keys=(),
        outcome="rule_missed",
        latency_ms=42,
    )
    telemetry.emit_recovery_event(rule_missed_event)

    lines = (tmp_path / "recovery_event.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2, f"expected 2 lines; got {len(lines)}"

    success_record = json.loads(lines[0])
    assert success_record["outcome"] == "success"
    assert success_record["error_class"] is None
    assert success_record["fingerprint_code"] is None
    assert success_record["action_code"] == "trigger_recompact"
    assert success_record["rewrite_applied_keys"] == []

    missed_record = json.loads(lines[1])
    assert missed_record["outcome"] == "rule_missed"
    assert missed_record["error_class"] == "unknown"
    assert missed_record["fingerprint_code"] is None
    assert missed_record["action_code"] is None, (
        f"rule_missed/budget_exhausted action_code must serialize as JSON "
        f"null; got {missed_record['action_code']!r}"
    )
    assert missed_record["rewrite_applied_keys"] == []
