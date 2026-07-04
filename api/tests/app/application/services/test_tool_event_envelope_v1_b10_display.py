"""B10 §7.3 — projector read_only/destructive 直通 + §7.5 wire null 语义."""
from __future__ import annotations

import json

from app.application.services.tool_event_envelope_v1 import (
    project_tool_event_to_envelope_v1,
)
from app.domain.models.event import ToolEvent, ToolEventStatus


def _make_calling_event(**overrides) -> ToolEvent:
    base = dict(
        tool_call_id="tc-b10-1",
        tool_name="file",
        function_name="file_read",
        function_args={"filepath": "/tmp/a.txt"},
        status=ToolEventStatus.CALLING,
    )
    base.update(overrides)
    return ToolEvent(**base)


class TestDisplayPolicyPassthrough:
    def test_bits_pass_through_verbatim(self):
        evt = _make_calling_event(
            read_only=True, destructive=False, display_icon="file"
        )
        env = project_tool_event_to_envelope_v1(evt)
        assert env.read_only is True
        assert env.destructive is False
        assert env.display_icon == "file"

    def test_default_none_bits_stay_none(self):
        env = project_tool_event_to_envelope_v1(_make_calling_event())
        assert env.read_only is None
        assert env.destructive is None


class TestWireNullSemantics:
    """F0.3: exclude_none=False → 新 key 以 null 出现; 旧字段不变 (INV-B10-0)."""

    def test_wire_json_carries_null_keys_when_unset(self):
        env = project_tool_event_to_envelope_v1(_make_calling_event())
        wire = json.loads(env.model_dump_json(by_alias=True, exclude_none=False))
        assert wire["read_only"] is None
        assert wire["destructive"] is None
        # 旧字段抽查: wire 短名保持 (I-R4.6)
        assert wire["name"] == "file"
        assert wire["function"] == "file_read"
        assert wire["args"] == {"filepath": "/tmp/a.txt"}
        assert wire["status"] == "calling"
        assert wire["envelope_version"] == 1

    def test_envelope_constructed_without_new_fields_defaults_none(self):
        # additive optional: 老构造代码不传新字段也能构造 (F0.2)
        env = project_tool_event_to_envelope_v1(_make_calling_event())
        assert env.model_dump()["read_only"] is None
