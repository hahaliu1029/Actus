"""Tests for create_dynamic_skill_langchain_tools bridge."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.tools import StructuredTool

from app.domain.models.tool_result import AllowError, AllowSuccess, DecisionReason

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _openai_tool_schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
) -> dict:
    """Build an OpenAI function-calling format dict."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
        },
    }


def _make_skill_tool_mock(tools_schemas: list[dict]) -> MagicMock:
    """Return a MagicMock that behaves like SkillTool for our purposes."""
    mock = MagicMock()
    mock.get_tools.return_value = tools_schemas
    mock.invoke = AsyncMock()
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmptyToolList:
    def test_empty_tool_list_returns_empty(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        mock = _make_skill_tool_mock([])
        result = create_dynamic_skill_langchain_tools(mock)
        assert result == []


class TestSingleToolWrap:
    def test_single_tool_wraps_correctly(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        schema = _openai_tool_schema(
            name="skill_demo_hello",
            description="Say hello",
            properties={
                "name": {"type": "string", "description": "The name to greet"},
            },
            required=["name"],
        )
        mock = _make_skill_tool_mock([schema])
        tools = create_dynamic_skill_langchain_tools(mock)

        assert len(tools) == 1
        assert isinstance(tools[0], StructuredTool)
        assert tools[0].name == "skill_demo_hello"
        assert tools[0].description == "Say hello"


class TestMultipleToolsIndependentClosures:
    async def test_multiple_tools_have_independent_closures(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        schema_a = _openai_tool_schema(
            name="tool_a",
            description="Tool A",
            properties={"x": {"type": "string"}},
            required=["x"],
        )
        schema_b = _openai_tool_schema(
            name="tool_b",
            description="Tool B",
            properties={"y": {"type": "integer"}},
            required=["y"],
        )
        mock = _make_skill_tool_mock([schema_a, schema_b])
        mock.invoke.return_value = AllowSuccess(content="ok")

        tools = create_dynamic_skill_langchain_tools(mock)
        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"

        # Invoke both to verify closures bind to different names
        await tools[0].ainvoke({"x": "hello"})
        await tools[1].ainvoke({"y": 42})

        calls = mock.invoke.call_args_list
        assert calls[0].args[0] == "tool_a"
        assert calls[1].args[0] == "tool_b"


class TestInvokeDelegatesToSkillTool:
    async def test_invoke_delegates_to_skill_tool(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        schema = _openai_tool_schema(
            name="skill_test_run",
            description="Run test",
            properties={
                "code": {"type": "string", "description": "Code to run"},
            },
            required=["code"],
        )
        mock = _make_skill_tool_mock([schema])
        mock.invoke.return_value = AllowSuccess(
            content="executed", data={"output": "42"}
        )

        tools = create_dynamic_skill_langchain_tools(mock)
        result = await tools[0].ainvoke({"code": "print(42)"})

        mock.invoke.assert_awaited_once_with("skill_test_run", code="print(42)")
        assert result == "executed"

class TestFailedInvokeReturnsAllowErrorContent:
    async def test_failed_invoke_returns_error_content(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        schema = _openai_tool_schema(
            name="skill_failing",
            description="Will fail",
            properties={"q": {"type": "string"}},
            required=["q"],
        )
        mock = _make_skill_tool_mock([schema])
        mock.invoke.return_value = AllowError(
            content="Something went wrong",
            reason=DecisionReason(
                type="exception",
                code="skill_runtime_error",
                message="Something went wrong",
            ),
        )

        tools = create_dynamic_skill_langchain_tools(mock)
        result = await tools[0].ainvoke({"q": "test"})
        assert result == "Something went wrong"


class TestOptionalParamsDefaultNone:
    async def test_optional_params_have_default_none(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        schema = _openai_tool_schema(
            name="skill_optional",
            description="Tool with optional param",
            properties={
                "query": {"type": "string", "description": "The query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            required=["query"],  # 'limit' is optional
        )
        mock = _make_skill_tool_mock([schema])
        mock.invoke.return_value = AllowSuccess(content="ok")

        tools = create_dynamic_skill_langchain_tools(mock)

        # Invoke without providing the optional param
        await tools[0].ainvoke({"query": "hello"})

        mock.invoke.assert_awaited_once()
        call_kwargs = mock.invoke.call_args
        assert call_kwargs.args[0] == "skill_optional"
        assert call_kwargs.kwargs["query"] == "hello"
        # 'limit' should either not be in kwargs or be None
        assert call_kwargs.kwargs.get("limit") is None


class TestMalformedSchemaSkipped:
    def test_malformed_schema_is_skipped(self):
        from app.domain.services.tools.langchain_dynamic_skill_tools import (
            create_dynamic_skill_langchain_tools,
        )

        good_schema = _openai_tool_schema(
            name="good_tool",
            description="A good tool",
            properties={"x": {"type": "string"}},
            required=["x"],
        )
        bad_schema = {"type": "function"}  # missing "function" key

        mock = _make_skill_tool_mock([bad_schema, good_schema])
        tools = create_dynamic_skill_langchain_tools(mock)

        # The bad schema should be skipped, only the good one remains
        assert len(tools) == 1
        assert tools[0].name == "good_tool"
