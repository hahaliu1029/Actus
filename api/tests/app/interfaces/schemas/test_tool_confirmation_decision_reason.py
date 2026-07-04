"""B10 §3.2 — ToolConfirmationSSEEvent.from_event 的 decision_reason wire 投影."""
from __future__ import annotations

from app.domain.models.event import ToolConfirmationEvent
from app.domain.models.tool_result import DecisionReason
from app.interfaces.schemas.event import (
    DecisionReasonWire,
    ToolConfirmationSSEEvent,
)


def _domain_event(**overrides) -> ToolConfirmationEvent:
    base = dict(
        tool_call_id="t1",
        tool_name="shell_execute",
        tool_args={"command": "rm -rf /tmp/x"},
        risk_level="high",
        risk_reason="dangerous command",
        matched_patterns=["rm -rf"],
        suggested_alternative=None,
        timeout_seconds=300,
    )
    base.update(overrides)
    return ToolConfirmationEvent(**base)


def test_from_event_maps_decision_reason_three_fields():
    evt = _domain_event(
        decision_reason=DecisionReason(
            type="approval_policy", code="rule_hit", message="blocked by rule"
        )
    )
    sse = ToolConfirmationSSEEvent.from_event(evt)
    assert sse.data.decision_reason == DecisionReasonWire(
        type="approval_policy", code="rule_hit", message="blocked by rule"
    )


def test_from_event_none_maps_none_and_old_fields_unchanged():
    # INV-B10-8 前置: 老事件 (无 decision_reason) → wire null, 旧字段逐一不变
    sse = ToolConfirmationSSEEvent.from_event(_domain_event())
    assert sse.data.decision_reason is None
    assert sse.data.tool_call_id == "t1"
    assert sse.data.tool_name == "shell_execute"
    assert sse.data.tool_args == {"command": "rm -rf /tmp/x"}
    assert sse.data.risk_level == "high"
    assert sse.data.risk_reason == "dangerous command"
    assert sse.data.matched_patterns == ["rm -rf"]
    assert sse.data.suggested_alternative is None
    assert sse.data.approval_options == ["once", "session", "always", "deny"]
    assert sse.data.timeout_seconds == 300
