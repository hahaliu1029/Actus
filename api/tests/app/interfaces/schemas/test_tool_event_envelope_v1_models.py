import json

import pytest
from pydantic import TypeAdapter, ValidationError

from app.domain.models.event import ToolEvent, ToolEventStatus
from app.interfaces.schemas.event import (
    DecisionReasonWire,
    FunctionResultV1,
    ToolEventEnvelopeV1,
    ToolSSEEvent,
    ToolStatusV1,
)


def test_tool_status_v1_accepts_5_values() -> None:
    adapter = TypeAdapter(ToolStatusV1)
    for value in ("ok", "error", "denied", "timeout", "passthrough"):
        assert adapter.validate_python(value) == value


def test_tool_status_v1_rejects_asked() -> None:
    """Asked 变体走独立 ToolConfirmationEvent, 不在 envelope v1 的 status 值域."""
    adapter = TypeAdapter(ToolStatusV1)
    with pytest.raises(ValidationError):
        adapter.validate_python("asked")


def test_tool_status_v1_rejects_unknown() -> None:
    adapter = TypeAdapter(ToolStatusV1)
    with pytest.raises(ValidationError):
        adapter.validate_python("partial_success")


def test_decision_reason_wire_defaults() -> None:
    dr = DecisionReasonWire(type="exception")
    assert dr.type == "exception"
    assert dr.code == ""
    assert dr.message == ""


def test_decision_reason_wire_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        DecisionReasonWire(type="exception", extra_field="x")


def test_function_result_v1_defaults() -> None:
    fr = FunctionResultV1(status="ok")
    assert fr.message == ""
    assert fr.data is None
    assert fr.retryable is False
    assert fr.user_action_required is False
    assert fr.reason is None
    assert fr.result_blocks is None


def test_function_result_v1_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        FunctionResultV1(status="ok", extra_field="x")


def test_tool_event_envelope_v1_wire_uses_short_names() -> None:
    """F1 fix: wire 字段名必须是 name/function/args 短名, 保持与 legacy 前端兼容."""
    env = ToolEventEnvelopeV1(
        tool_call_id="c1",
        tool_name="shell",
        function_name="shell_execute",
        function_args={"command": "ls"},
        status="called",
    )
    wire = env.model_dump(mode="json", by_alias=True)
    assert "name" in wire
    assert "function" in wire
    assert "args" in wire
    assert wire["name"] == "shell"
    assert wire["function"] == "shell_execute"
    assert wire["args"] == {"command": "ls"}
    assert "tool_name" not in wire
    assert "function_name" not in wire
    assert "function_args" not in wire


def test_tool_event_envelope_v1_accepts_short_names_via_alias() -> None:
    """Pydantic populate_by_name=True: 允许 Python 构造用长名 or 短名."""
    env1 = ToolEventEnvelopeV1(
        tool_call_id="c1", tool_name="shell", function_name="x",
        function_args={}, status="called",
    )
    env2 = ToolEventEnvelopeV1.model_validate({
        "tool_call_id": "c1", "name": "shell", "function": "x",
        "args": {}, "status": "called",
    })
    assert env1.tool_name == env2.tool_name == "shell"


def test_envelope_version_default_is_1() -> None:
    env = ToolEventEnvelopeV1(
        tool_call_id="c1", tool_name="shell", function_name="x",
        function_args={}, status="called",
    )
    assert env.envelope_version == 1


def test_envelope_version_rejects_zero() -> None:
    """Round 2b F1C fix: int + validator (>=1), not Literal[1]."""
    with pytest.raises(ValidationError):
        ToolEventEnvelopeV1(
            envelope_version=0,
            tool_call_id="c1", tool_name="shell", function_name="x",
            function_args={}, status="called",
        )


def test_envelope_version_accepts_v2_forward_compat() -> None:
    """Round 2b F1C fix: 非 Literal, 允许 v2 事件流经老 server 时 Pydantic 不硬抛."""
    env = ToolEventEnvelopeV1(
        envelope_version=2,
        tool_call_id="c1", tool_name="shell", function_name="x",
        function_args={}, status="called",
    )
    assert env.envelope_version == 2


def test_envelope_version_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        ToolEventEnvelopeV1(
            envelope_version=-1,
            tool_call_id="c1", tool_name="shell", function_name="x",
            function_args={}, status="called",
        )


def test_tool_event_envelope_v1_rejects_extra() -> None:
    """extra=forbid is a CS3 contract invariant — unknown fields must not be silently swallowed."""
    with pytest.raises(ValidationError):
        ToolEventEnvelopeV1(
            tool_call_id="c1", tool_name="shell", function_name="x",
            function_args={}, status="called", rogue_field="x",
        )


class TestToolSSEEventIntegration:
    def test_from_event_calls_projector(self) -> None:
        """ToolSSEEvent.from_event must route through projector, not build ToolEventData directly."""
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
        )
        sse_evt = ToolSSEEvent.from_event(evt)
        assert isinstance(sse_evt.data, ToolEventEnvelopeV1)
        assert sse_evt.data.envelope_version == 1

    def test_to_sse_data_json_wire_uses_short_names(self) -> None:
        """I-R4.6: wire serialization forces by_alias=True to emit name/function/args."""
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
            status=ToolEventStatus.CALLED,
        )
        sse_evt = ToolSSEEvent.from_event(evt)
        wire_json = sse_evt.to_sse_data_json()
        wire = json.loads(wire_json)
        assert "name" in wire
        assert "function" in wire
        assert "args" in wire
        assert "tool_name" not in wire
        assert "function_name" not in wire
        assert "function_args" not in wire

    def test_from_event_propagates_seq_to_wire_envelope(self) -> None:
        """B3-core PR-1 §3.3: ToolEvent.seq propagates through the projector to
        the SSE wire envelope (was silently dropped before the projector was
        updated to forward seq into ToolEventEnvelopeV1)."""
        evt = ToolEvent(
            tool_call_id="c2",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "echo"},
            status=ToolEventStatus.CALLING,
            seq=42,
        )
        sse_evt = ToolSSEEvent.from_event(evt)
        assert sse_evt.data.seq == 42

        wire_json = sse_evt.to_sse_data_json()
        wire = json.loads(wire_json)
        assert wire.get("seq") == 42

    def test_from_event_legacy_event_without_seq_keeps_none(self) -> None:
        """B3-core PR-1 §3.3 backward compat: legacy ToolEvent (seq=None default)
        round-trips with `data.seq is None` and the wire payload includes
        `"seq": null` (consistent with other SSE event types)."""
        evt = ToolEvent(
            tool_call_id="c3",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "noop"},
            status=ToolEventStatus.CALLING,
        )
        sse_evt = ToolSSEEvent.from_event(evt)
        assert sse_evt.data.seq is None

        wire_json = sse_evt.to_sse_data_json()
        wire = json.loads(wire_json)
        assert wire.get("seq") is None
