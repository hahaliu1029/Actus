"""Tests for LangChain MCP adapter.

Also covers create_mcp_langchain_tools factory per-schema isolation
(#27 prerequisite): MCP factory must skip malformed schemas with a warning
log instead of crashing the whole factory, aligning with
create_dynamic_skill_langchain_tools behavior.
"""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.models.tool_result import ToolResult
from app.domain.services.tools import langchain_mcp as mcp_module
from app.domain.services.tools.langchain_mcp import create_mcp_langchain_tools

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class TestCreateMCPTools:
    async def test_creates_tools_from_mcp_tool(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.get_tools.return_value = [
            {
                "type": "function",
                "function": {
                    "name": "mcp_tool_1",
                    "description": "A test MCP tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        mock_mcp_tool.invoke = AsyncMock(return_value=ToolResult(
            success=True, message="ok", data={"result": "done"}
        ))

        tools = create_mcp_langchain_tools(mock_mcp_tool)
        assert len(tools) == 1
        assert tools[0].name == "mcp_tool_1"

    async def test_empty_tools_returns_empty_list(self):
        mock_mcp_tool = MagicMock()
        mock_mcp_tool.get_tools.return_value = []

        tools = create_mcp_langchain_tools(mock_mcp_tool)
        assert tools == []


class _FakeMCPTool:
    """Minimal MCP tool stand-in returning a list of function schemas."""

    def __init__(self, schemas: list[dict[str, Any]]):
        self._schemas = schemas

    def get_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas)


def _good_schema(name: str) -> dict[str, Any]:
    return {
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "q"},
                },
                "required": ["query"],
            },
        }
    }


def test_malformed_schema_is_skipped_with_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Regression: one malformed MCP schema must not take down the whole factory."""
    caplog.set_level(logging.WARNING)

    original_build_model = mcp_module._build_pydantic_model

    # "BadSchemaArgs" is derived by mcp_module._sanitize_model_name("bad_schema").
    # If _sanitize_model_name changes its suffix (e.g., "Args" → "Model"), this
    # guard silently stops firing — keep the tool name and the expected model
    # name in sync.
    def flaky_build_model(model_name: str, schema: dict):
        if model_name == "BadSchemaArgs":
            raise ValueError("simulated pydantic model failure")
        return original_build_model(model_name, schema)

    monkeypatch.setattr(
        mcp_module, "_build_pydantic_model", flaky_build_model
    )

    fake = _FakeMCPTool(
        [
            _good_schema("good_one"),
            _good_schema("bad_schema"),
            _good_schema("good_two"),
        ]
    )

    tools = create_mcp_langchain_tools(
        fake,
        tool_names=None,
        url_map_ref=lambda: {},
        sandbox_file_uploader=None,
    )

    names = [t.name for t in tools]
    assert "good_one" in names
    assert "good_two" in names
    assert "bad_schema" not in names
    assert any(
        "Skipping malformed MCP tool schema" in record.message
        for record in caplog.records
    )


def test_structured_tool_from_function_failure_is_skipped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Regression: per-schema failure at StructuredTool.from_function stage
    (not at _json_schema_to_pydantic stage) is also isolated."""
    caplog.set_level(logging.WARNING)

    original_from_function = mcp_module.StructuredTool.from_function

    def flaky_from_function(*args, **kwargs):
        if kwargs.get("name") == "bad_second":
            raise TypeError("simulated structured tool build failure")
        return original_from_function(*args, **kwargs)

    monkeypatch.setattr(
        mcp_module.StructuredTool, "from_function", flaky_from_function
    )

    fake = _FakeMCPTool(
        [
            _good_schema("good_first"),
            _good_schema("bad_second"),
            _good_schema("good_third"),
        ]
    )

    tools = create_mcp_langchain_tools(
        fake,
        tool_names=None,
        url_map_ref=lambda: {},
        sandbox_file_uploader=None,
    )

    names = [t.name for t in tools]
    assert "good_first" in names
    assert "good_third" in names
    assert "bad_second" not in names
    assert any(
        "Skipping malformed MCP tool schema" in record.message
        for record in caplog.records
    )


def test_tool_names_filter_combined_with_schema_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """Regression: tool_names filter and per-schema failure compose correctly.

    - Schemas filtered out by tool_names should NOT reach _build_pydantic_model
      (and therefore must not trigger warnings).
    - Schemas that pass the filter but fail conversion should still be skipped
      and logged.
    """
    caplog.set_level(logging.WARNING)

    patched_calls: list[str] = []
    original_build_model = mcp_module._build_pydantic_model

    def flaky_build_model(model_name: str, schema: dict):
        patched_calls.append(model_name)
        if model_name == "BadSchemaArgs":
            raise ValueError("simulated pydantic model failure")
        return original_build_model(model_name, schema)

    monkeypatch.setattr(
        mcp_module, "_build_pydantic_model", flaky_build_model
    )

    fake = _FakeMCPTool(
        [
            _good_schema("good_two"),     # passes filter
            _good_schema("bad_schema"),   # passes filter but raises
            _good_schema("filtered_out"), # filtered out — must not reach _build_pydantic_model
        ]
    )

    tools = create_mcp_langchain_tools(
        fake,
        tool_names={"good_two", "bad_schema"},
        url_map_ref=lambda: {},
        sandbox_file_uploader=None,
    )

    names = [t.name for t in tools]
    assert names == ["good_two"]

    # bad_schema was attempted and logged
    assert any(
        "Skipping malformed MCP tool schema" in record.message
        and "bad_schema" in record.message
        for record in caplog.records
    )

    # filtered_out was never even tried — no model construction for it
    assert "FilteredOutArgs" not in patched_calls
