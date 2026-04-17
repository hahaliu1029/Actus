"""R2 CS2 day-0 preview for R4 CS3 envelope: backward-compat ``ToolEvent`` deserialization.

R4 will freeze a new ``ToolEventEnvelope`` that embeds ``ToolArtifact``
alongside the legacy ``function_result: Optional[ToolResult]`` field.
The transition must be backward-compatible: pre-R2 ``ToolEvent`` JSON
blobs produced before CS2 landed (no ``function_result``, no typed
``artifact``, no ``tool_source``) must still deserialize cleanly
through both ``ToolEvent.model_validate(...)`` and the polymorphic
``TypeAdapter[Event]`` dispatch.

These tests lock that invariant in place NOW (Chunk 5 / PR-B Commit
3) so any future model change to ``ToolEvent`` — required fields,
field renames, reducers — must explicitly grandfather the legacy
shape instead of silently breaking session history replay from the
event log.

Scope: only the shape of pre-R2 JSON that's actually in production
event logs. We don't try to reconstruct the typed ``ToolArtifact``
from a legacy blob (that's R4's job) — we just verify the legacy
blob still loads into a valid ``ToolEvent`` with sensible defaults.
"""
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from app.domain.models.event import (
    Event,
    ToolEvent,
    ToolEventStatus,
)
from app.domain.models.tool_result import AllowSuccess, ToolArtifact
from app.domain.services.tools.tool_source_resolver import ToolSource


class TestLegacyToolEventDirectValidation:
    """Direct ``ToolEvent.model_validate`` on pre-R2 JSON shapes."""

    def test_minimal_legacy_shape_without_artifact(self):
        """Pre-R2 event with only required core fields — no artifact,
        no function_result, no tool_source."""
        legacy_json = {
            "type": "tool",
            "id": "evt_legacy_1",
            "tool_call_id": "c1",
            "tool_name": "shell",  # category in legacy format
            "function_name": "shell_execute",
            "function_args": {"command": "ls"},
        }

        evt = ToolEvent.model_validate(legacy_json)

        assert evt.type == "tool"
        assert evt.tool_call_id == "c1"
        assert evt.function_name == "shell_execute"
        assert evt.function_args == {"command": "ls"}
        # Optional fields default to None / CALLING
        assert evt.function_result is None
        assert evt.status == ToolEventStatus.CALLING
        assert evt.tool_content is None

    def test_legacy_with_called_status(self):
        """Pre-R2 event in the terminal CALLED state."""
        legacy_json = {
            "type": "tool",
            "id": "evt_legacy_2",
            "tool_call_id": "c2",
            "tool_name": "search",
            "function_name": "search_web",
            "function_args": {"query": "AI news"},
            "status": "called",
        }

        evt = ToolEvent.model_validate(legacy_json)

        assert evt.status == ToolEventStatus.CALLED

    def test_legacy_with_legacy_function_result(self):
        """Pre-R2 event carrying a legacy ``ToolResult`` in ``function_result``.

        This is the R1 shape — ``ToolResult(success=..., message=...)``
        serialized as a nested dict. R2 introduces ``ToolArtifact`` as
        the typed replacement, but the legacy shape must still
        deserialize because it's in production event logs.
        """
        legacy_json = {
            "type": "tool",
            "id": "evt_legacy_3",
            "tool_call_id": "c3",
            "tool_name": "file",
            "function_name": "file_read",
            "function_args": {"path": "/tmp/x"},
            "status": "called",
            "function_result": {
                "success": True,
                "message": "file contents here",
                "data": None,
            },
        }

        evt = ToolEvent.model_validate(legacy_json)

        assert evt.function_result is not None
        assert evt.function_result.success is True
        assert evt.function_result.message == "file contents here"

    def test_legacy_failure_result(self):
        """Pre-R2 failure case: ``function_result.success=False``."""
        legacy_json = {
            "type": "tool",
            "id": "evt_legacy_4",
            "tool_call_id": "c4",
            "tool_name": "mcp",
            "function_name": "mcp_slack_post",
            "function_args": {"channel": "#foo"},
            "status": "called",
            "function_result": {
                "success": False,
                "message": "Connection refused",
                "data": None,
            },
        }

        evt = ToolEvent.model_validate(legacy_json)

        assert evt.function_result is not None
        assert evt.function_result.success is False
        assert "Connection refused" in evt.function_result.message


class TestLegacyToolEventViaEventTypeAdapter:
    """Polymorphic ``TypeAdapter[Event]`` dispatch on pre-R2 shapes."""

    def test_type_adapter_dispatches_legacy_to_tool_event(self):
        """``TypeAdapter(Event)`` must route a pre-R2 tool event to
        the ``ToolEvent`` variant of the tagged union."""
        legacy_json = {
            "type": "tool",
            "id": "evt_adapter_1",
            "tool_call_id": "c5",
            "tool_name": "shell",
            "function_name": "shell_execute",
            "function_args": {},
        }

        adapter: TypeAdapter[Event] = TypeAdapter(Event)
        evt = adapter.validate_python(legacy_json)

        assert isinstance(evt, ToolEvent)
        assert evt.tool_call_id == "c5"

    def test_type_adapter_validate_json_path(self):
        """Also cover the JSON-string validation path used by the
        event log reader — ``TypeAdapter(Event).validate_json(...)``."""
        import json

        legacy_json = {
            "type": "tool",
            "id": "evt_adapter_2",
            "tool_call_id": "c6",
            "tool_name": "search",
            "function_name": "search_web",
            "function_args": {"query": "x"},
            "status": "called",
        }

        adapter: TypeAdapter[Event] = TypeAdapter(Event)
        evt = adapter.validate_json(json.dumps(legacy_json))

        assert isinstance(evt, ToolEvent)
        assert evt.status == ToolEventStatus.CALLED


class TestLegacyToolEventRoundTrip:
    """Legacy → validate → dump must not synthesize fields that the
    original payload didn't carry.

    This locks in the soft-coexistence invariant that R2 doesn't
    retroactively add ``ToolArtifact`` to legacy events. R4 is where
    envelope migration happens; Chunk 5 only guards against silent
    drift via model changes.
    """

    def test_legacy_dump_does_not_add_artifact_field(self):
        legacy_json = {
            "type": "tool",
            "id": "evt_rt_1",
            "tool_call_id": "c7",
            "tool_name": "shell",
            "function_name": "shell_execute",
            "function_args": {"command": "pwd"},
            "status": "called",
        }

        evt = ToolEvent.model_validate(legacy_json)
        dumped = evt.model_dump(mode="json")

        # R4 post-F5: artifact key may be present (= None) or absent depending on
        # future serialization config (e.g. exclude_none). Either is acceptable —
        # legacy blobs never populate it; projector's legacy fallback path handles this.
        assert dumped.get("artifact") is None
        # function_result defaults to None and is still present in
        # the dump (not stripped) — consumers that read it must
        # handle None gracefully.
        assert "function_result" in dumped
        assert dumped["function_result"] is None


class TestR4ToolEventExtensions:
    """R4 新增 6 个 Optional 字段测试 (F2 fix: artifact 是 dict 而非 typed)."""

    def test_r4_tool_event_all_optional_fields_default(self) -> None:
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={"command": "ls"},
        )
        assert evt.artifact is None
        assert evt.tool_source is None
        assert evt.activity_description == ""
        assert evt.display_icon is None
        assert evt.render_style is None
        assert evt.media_type is None

    def test_r4_tool_event_artifact_accepts_dict(self) -> None:
        """F2: artifact 字段类型 Optional[Dict[str, Any]], 不是 typed ToolArtifact."""
        artifact_dict = {
            "tool_call_id": "c1",
            "tool_name": "shell_execute",
            "tool_source": {"source": "native", "category": "shell", "canonical_name": "shell_execute"},
            "outcome": {
                "variant": "allow_success",
                "content": "ok",
                "data": None,
            },
        }
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={},
            artifact=artifact_dict,
        )
        assert isinstance(evt.artifact, dict)
        assert evt.artifact["outcome"]["variant"] == "allow_success"

    def test_r4_tool_event_artifact_rejects_typed_object(self) -> None:
        """F2: Pydantic Optional[Dict[str, Any]] 会拒绝非 dict 输入到 artifact 字段.

        spec 锁 fail-fast 语义: react_graph 必须手动 .model_dump() 再赋给
        ToolEvent.artifact, 不能直接塞 typed ToolArtifact 实例.
        """
        typed_artifact = ToolArtifact(
            tool_call_id="c1",
            tool_name="shell_execute",
            tool_source=ToolSource(source="native", category="shell", canonical_name="shell_execute"),
            outcome=AllowSuccess(content="ok"),
        )
        with pytest.raises(ValidationError):
            ToolEvent(
                tool_call_id="c1",
                tool_name="shell",
                function_name="shell_execute",
                function_args={},
                artifact=typed_artifact,
            )

    def test_r4_tool_event_tool_source_typed(self) -> None:
        ts = ToolSource(source="native", category="shell", canonical_name="shell_execute")
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={},
            tool_source=ts,
        )
        assert evt.tool_source.source == "native"
        assert evt.tool_source.category == "shell"
        assert evt.tool_source.canonical_name == "shell_execute"

    def test_r4_tool_event_with_artifact_dict_deserializes(self) -> None:
        """R4 shape JSON (artifact as dict) validate 通过."""
        json_data = {
            "type": "tool",
            "id": "evt_r4_1",
            "tool_call_id": "c1",
            "tool_name": "shell",
            "function_name": "shell_execute",
            "function_args": {"command": "ls"},
            "status": "called",
            "artifact": {
                "tool_call_id": "c1",
                "tool_name": "shell_execute",
                "tool_source": {"source": "native", "category": "shell",
                                "canonical_name": "shell_execute"},
                "outcome": {"variant": "allow_success", "content": "ok", "data": None},
            },
        }
        evt = ToolEvent.model_validate(json_data)
        assert evt.artifact is not None
        assert evt.artifact["outcome"]["variant"] == "allow_success"

    def test_r4_pre_r4_json_validates_via_type_adapter(self) -> None:
        """R0/R1 shape (无 artifact) 仍 TypeAdapter(Event) 通过."""
        pre_r4_json = {
            "type": "tool",
            "id": "evt_pre_r4_1",
            "tool_call_id": "c1",
            "tool_name": "shell",
            "function_name": "shell_execute",
            "function_args": {},
        }
        adapter: TypeAdapter[Event] = TypeAdapter(Event)
        evt = adapter.validate_python(pre_r4_json)
        assert isinstance(evt, ToolEvent)
        assert evt.artifact is None
        assert evt.tool_source is None

    def test_r4_unknown_variant_artifact_no_type_adapter_error(self) -> None:
        """F2 关键: artifact 是 dict 而非 typed, 未知 variant 不在事件日志反序列化层炸."""
        future_variant_json = {
            "type": "tool",
            "id": "evt_future_variant",
            "tool_call_id": "c1",
            "tool_name": "shell",
            "function_name": "shell_execute",
            "function_args": {},
            "status": "called",
            "artifact": {
                "tool_call_id": "c1",
                "tool_name": "shell_execute",
                "tool_source": {"source": "native", "category": "shell",
                                "canonical_name": "shell_execute"},
                # 未来 R2 可能加 "partial_success" variant — 老 server 读这个 JSON
                # 必须 TypeAdapter 成功 (artifact 是 dict), projector 层才降级
                "outcome": {"variant": "partial_success", "content": "部分成功"},
            },
        }
        adapter: TypeAdapter[Event] = TypeAdapter(Event)
        evt = adapter.validate_python(future_variant_json)
        assert isinstance(evt, ToolEvent)
        assert evt.artifact["outcome"]["variant"] == "partial_success"

    def test_r4_dump_includes_envelope_version_when_projected(self) -> None:
        """Projector 输出必有 envelope_version=1 (所有路径)."""
        from app.application.services.tool_event_envelope_v1 import (
            project_tool_event_to_envelope_v1,
        )

        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={},
        )
        envelope = project_tool_event_to_envelope_v1(evt)
        wire = envelope.model_dump(mode="json", by_alias=True)
        assert wire["envelope_version"] == 1

    def test_r4_r2_legacy_test_assertions_still_valid(self) -> None:
        """R4 更新后, 现有 test_legacy_dump_does_not_add_artifact_field 变体仍绿.
        这里再加一条覆盖 artifact=None 时 wire roundtrip 不丢信息."""
        evt = ToolEvent(
            tool_call_id="c1",
            tool_name="shell",
            function_name="shell_execute",
            function_args={},
        )
        dumped = evt.model_dump(mode="json")
        # R4 字段 1/6: artifact — 存在且为 None
        assert "artifact" in dumped
        assert dumped["artifact"] is None
        # R4 字段 2-6/6: 其他 5 个 Optional key 都存在
        for key in ("tool_source", "activity_description", "display_icon",
                    "render_style", "media_type"):
            assert key in dumped
