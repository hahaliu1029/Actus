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

from pydantic import TypeAdapter

from app.domain.models.event import (
    Event,
    ToolEvent,
    ToolEventStatus,
)


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

        # ToolEvent has no `artifact` field — asserting absence
        # documents the R4 envelope boundary.
        assert "artifact" not in dumped
        # function_result defaults to None and is still present in
        # the dump (not stripped) — consumers that read it must
        # handle None gracefully.
        assert "function_result" in dumped
        assert dumped["function_result"] is None
