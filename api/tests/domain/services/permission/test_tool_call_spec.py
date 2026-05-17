import dataclasses

import pytest
from app.domain.services.permission.tool_call_spec import ToolCallSpec


def test_tool_call_spec_is_frozen():
    spec = ToolCallSpec(
        tool_name="file_write",
        tool_args={"path": "/workspace/notes.md", "content": "hi"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tool_name = "x"


def test_tool_call_spec_optional_fields_default():
    spec = ToolCallSpec(
        tool_name="shell_execute",
        tool_args={"command": "ls"},
        tool_source="native",
        user_id="u1",
        session_id="s1",
    )
    assert spec.primary_arg is None
    assert spec.dir_arg is None
    assert spec.arg_digest is None
    assert spec.risk_assessment is None


def test_tool_call_spec_tool_source_str_only():
    # str typed; PE-1+ will validate the union at construction time
    spec = ToolCallSpec(
        tool_name="x", tool_args={}, tool_source="native",
        user_id="u", session_id="s",
    )
    assert spec.tool_source == "native"
