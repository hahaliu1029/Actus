"""Tests for ActusResponsesModel -- BaseChatModel wrapping OpenAI Responses API.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult

from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers -- mock Responses API response objects
# ---------------------------------------------------------------------------


def _make_responses_api_response(
    text: str | None = "Hello!",
    function_calls: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a mock Responses API response as a dict (simulating model_dump()).

    Responses API output contains items of type "message" (with content parts)
    and/or type "function_call".
    """
    output: list[dict[str, Any]] = []

    if text is not None:
        output.append({
            "type": "message",
            "content": [
                {"type": "output_text", "text": text},
            ],
        })

    if function_calls:
        for fc in function_calls:
            output.append({
                "type": "function_call",
                "call_id": fc.get("call_id", ""),
                "name": fc.get("name", ""),
                "arguments": fc.get("arguments", "{}"),
            })

    return {"status": "completed", "output": output}


class _MockResponseObj:
    """Simulates an OpenAI SDK response object with model_dump()."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def model_dump(self) -> dict[str, Any]:
        return self._data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ActusResponsesModel:
    """Create an ActusResponsesModel with test configuration."""
    return ActusResponsesModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="gpt-5.4-pro",
        temperature=0.5,
        max_tokens=1024,
    )


# ---------------------------------------------------------------------------
# Tests: _llm_type
# ---------------------------------------------------------------------------


class TestLLMType:
    """Test _llm_type property."""

    def test_llm_type_value(self, model: ActusResponsesModel) -> None:
        assert model._llm_type == "actus-responses"


# ---------------------------------------------------------------------------
# Tests: _generate (sync) should raise
# ---------------------------------------------------------------------------


class TestSyncGenerate:
    """_generate should raise NotImplementedError (async-only project)."""

    def test_generate_raises(self, model: ActusResponsesModel) -> None:
        with pytest.raises(NotImplementedError):
            model._generate([HumanMessage(content="hi")])


# ---------------------------------------------------------------------------
# Tests: _sanitize_json_schema
# ---------------------------------------------------------------------------


class TestSanitizeJsonSchema:
    """Test _sanitize_json_schema adds missing items for array types."""

    def test_adds_items_for_array_type(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array"},  # missing "items"
            },
        }
        result = ActusResponsesModel._sanitize_json_schema(schema)
        assert result["properties"]["tags"]["items"] == {}

    def test_keeps_existing_items(self) -> None:
        schema = {
            "type": "array",
            "items": {"type": "string"},
        }
        result = ActusResponsesModel._sanitize_json_schema(schema)
        assert result["items"] == {"type": "string"}

    def test_handles_nested_array(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {
                        "type": "array",  # nested array, missing items
                    },
                },
            },
        }
        result = ActusResponsesModel._sanitize_json_schema(schema)
        assert result["properties"]["matrix"]["items"]["items"] == {}

    def test_handles_list_type_with_array(self) -> None:
        """When type is a list like ["array", "null"], should still add items."""
        schema = {"type": ["array", "null"]}
        result = ActusResponsesModel._sanitize_json_schema(schema)
        assert result["items"] == {}

    def test_non_array_type_untouched(self) -> None:
        schema = {"type": "string"}
        result = ActusResponsesModel._sanitize_json_schema(schema)
        assert "items" not in result

    def test_handles_list_input(self) -> None:
        """Schema can be a list; each element should be sanitized."""
        schema_list = [
            {"type": "array"},
            {"type": "string"},
        ]
        result = ActusResponsesModel._sanitize_json_schema(schema_list)
        assert result[0]["items"] == {}
        assert "items" not in result[1]

    def test_handles_non_dict_non_list(self) -> None:
        """Primitives should pass through unchanged."""
        assert ActusResponsesModel._sanitize_json_schema("hello") == "hello"
        assert ActusResponsesModel._sanitize_json_schema(42) == 42
        assert ActusResponsesModel._sanitize_json_schema(None) is None


# ---------------------------------------------------------------------------
# Tests: _convert_tools
# ---------------------------------------------------------------------------


class TestConvertTools:
    """Test Chat Completions tool format -> Responses API tool format."""

    def test_basic_conversion(self) -> None:
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                },
            }
        ]
        result = ActusResponsesModel._convert_tools(chat_tools)
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather for a city"
        assert tool["strict"] is False
        assert tool["parameters"]["type"] == "object"

    def test_sanitizes_array_params(self) -> None:
        """Array parameters without items should be sanitized."""
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "process",
                    "description": "Process items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items_list": {"type": "array"},
                        },
                    },
                },
            }
        ]
        result = ActusResponsesModel._convert_tools(chat_tools)
        props = result[0]["parameters"]["properties"]
        assert props["items_list"]["items"] == {}

    def test_empty_tools(self) -> None:
        assert ActusResponsesModel._convert_tools([]) == []

    def test_missing_function_fields_default(self) -> None:
        """Missing function fields should default gracefully."""
        chat_tools = [{"type": "function", "function": {}}]
        result = ActusResponsesModel._convert_tools(chat_tools)
        assert result[0]["name"] == ""
        assert result[0]["description"] == ""
        assert result[0]["parameters"] == {}


# ---------------------------------------------------------------------------
# Tests: _convert_input_messages
# ---------------------------------------------------------------------------


class TestConvertInputMessages:
    """Test conversion of LangChain BaseMessage -> Responses API input items."""

    def test_system_message(self, model: ActusResponsesModel) -> None:
        msgs = [SystemMessage(content="You are helpful")]
        result = model._convert_input_messages(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful"

    def test_human_message(self, model: ActusResponsesModel) -> None:
        msgs = [HumanMessage(content="Hello")]
        result = model._convert_input_messages(msgs)
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_tool_message_becomes_function_call_output(self, model: ActusResponsesModel) -> None:
        msgs = [ToolMessage(content="result data", tool_call_id="call_123")]
        result = model._convert_input_messages(msgs)
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call_123"
        assert result[0]["output"] == "result data"

    def test_ai_message_with_tool_calls(self, model: ActusResponsesModel) -> None:
        """AI message with tool_calls should produce function_call items."""
        msgs = [
            AIMessage(
                content="Let me check",
                tool_calls=[
                    {"id": "call_abc", "name": "search", "args": {"q": "test"}},
                ],
            ),
        ]
        result = model._convert_input_messages(msgs)

        # Should produce: assistant content message + function_call item
        assert len(result) == 2
        # First: assistant content
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Let me check"
        # Second: function_call
        assert result[1]["type"] == "function_call"
        assert result[1]["call_id"] == "call_abc"
        assert result[1]["name"] == "search"
        # arguments should be JSON string
        args = json.loads(result[1]["arguments"])
        assert args == {"q": "test"}

    def test_ai_message_with_tool_calls_no_content(self, model: ActusResponsesModel) -> None:
        """AI message with tool_calls but empty content should skip content entry."""
        msgs = [
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_abc", "name": "search", "args": {"q": "test"}},
                ],
            ),
        ]
        result = model._convert_input_messages(msgs)
        # Should produce only function_call items (no assistant content since empty)
        assert len(result) == 1
        assert result[0]["type"] == "function_call"

    def test_ai_message_without_tool_calls(self, model: ActusResponsesModel) -> None:
        """AI message without tool_calls should pass through as normal."""
        msgs = [AIMessage(content="Just a response")]
        result = model._convert_input_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Just a response"

    def test_full_conversation(self, model: ActusResponsesModel) -> None:
        """A full turn: system + human + AI(tool_call) + tool_result."""
        msgs = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Search for X"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search", "args": {"q": "X"}},
                ],
            ),
            ToolMessage(content="Found X", tool_call_id="call_1"),
        ]
        result = model._convert_input_messages(msgs)
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[2]["type"] == "function_call"
        assert result[3]["type"] == "function_call_output"


# ---------------------------------------------------------------------------
# Tests: _normalize_response
# ---------------------------------------------------------------------------


class TestNormalizeResponse:
    """Test Responses API output -> AIMessage conversion."""

    def test_text_only_response(self) -> None:
        response = _MockResponseObj(_make_responses_api_response(text="Hello world"))
        msg = ActusResponsesModel._normalize_response(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Hello world"
        assert "tool_calls" not in msg

    def test_tool_calls_response(self) -> None:
        response = _MockResponseObj(_make_responses_api_response(
            text=None,
            function_calls=[
                {
                    "call_id": "call_abc",
                    "name": "get_weather",
                    "arguments": '{"city": "Paris"}',
                },
            ],
        ))
        msg = ActusResponsesModel._normalize_response(response)
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == '{"city": "Paris"}'

    def test_mixed_text_and_tool_calls(self) -> None:
        response = _MockResponseObj(_make_responses_api_response(
            text="I'll search for that",
            function_calls=[
                {"call_id": "call_1", "name": "search", "arguments": '{"q": "test"}'},
            ],
        ))
        msg = ActusResponsesModel._normalize_response(response)
        assert msg["content"] == "I'll search for that"
        assert len(msg["tool_calls"]) == 1

    def test_empty_output(self) -> None:
        response = _MockResponseObj({"output": []})
        msg = ActusResponsesModel._normalize_response(response)
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert "tool_calls" not in msg

    def test_dict_input(self) -> None:
        """Should handle dict directly (no model_dump needed)."""
        raw = _make_responses_api_response(text="test")
        msg = ActusResponsesModel._normalize_response(raw)
        assert msg["content"] == "test"


# ---------------------------------------------------------------------------
# Tests: _agenerate
# ---------------------------------------------------------------------------


class TestAGenerate:
    """Test _agenerate -- async Responses API calls."""

    async def test_basic_text_response(self, model: ActusResponsesModel) -> None:
        """Mock AsyncOpenAI responses.create to return a simple text response."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="Hello from LLM!"))

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Hi")])

        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello from LLM!"
        assert msg.tool_calls == []

    async def test_tool_calls_response(self, model: ActusResponsesModel) -> None:
        """Mock Responses API to return a response with tool calls."""
        mock_resp = _MockResponseObj(_make_responses_api_response(
            text=None,
            function_calls=[
                {
                    "call_id": "call_abc123",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            ],
        ))

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Weather?")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"location": "Paris"}

    async def test_max_tokens_mapped_to_max_output_tokens(self, model: ActusResponsesModel) -> None:
        """max_tokens should be sent as max_output_tokens to Responses API."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate([HumanMessage(content="hi")])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "max_output_tokens" in call_kwargs
        assert call_kwargs["max_output_tokens"] == 1024
        assert "max_tokens" not in call_kwargs

    async def test_input_param_instead_of_messages(self, model: ActusResponsesModel) -> None:
        """Responses API uses 'input' param instead of 'messages'."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate([HumanMessage(content="hi")])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "input" in call_kwargs
        assert "messages" not in call_kwargs

    async def test_response_format_mapped_to_text_format(self, model: ActusResponsesModel) -> None:
        """response_format should be mapped to text.format for Responses API."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text='{"answer": 42}'))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="answer")],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "response_format" not in call_kwargs
        assert call_kwargs["text"] == {"format": {"type": "json_object"}}

    async def test_tools_converted_and_sent(self, model: ActusResponsesModel) -> None:
        """Tools should be converted from Chat Completions format to Responses API format."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A tool",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tools=tools,
            )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        assert len(sent_tools) == 1
        assert sent_tools[0]["name"] == "test_tool"
        assert sent_tools[0]["strict"] is False
        # Should be in Responses API format, not Chat Completions format
        assert "function" not in sent_tools[0]

    async def test_tool_choice_passed(self, model: ActusResponsesModel) -> None:
        """tool_choice from kwargs should be passed to the API."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="auto",
            )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"

    async def test_tool_choice_any_normalized_to_required(self, model: ActusResponsesModel) -> None:
        """LangChain's 'any' tool_choice should be normalized to 'required' for the API."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="any",
            )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"

    async def test_messages_converted_to_input(self, model: ActusResponsesModel) -> None:
        """LangChain messages should be converted to Responses API input format."""
        mock_resp = _MockResponseObj(_make_responses_api_response(text="Done"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        messages = [
            SystemMessage(content="You are helpful"),
            HumanMessage(content="Hello"),
            AIMessage(
                content="Let me check",
                tool_calls=[{"id": "tc1", "name": "search", "args": {"q": "test"}}],
            ),
            ToolMessage(content="result data", tool_call_id="tc1"),
        ]

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(messages)

        call_kwargs = mock_client.responses.create.call_args.kwargs
        input_items = call_kwargs["input"]

        # system message
        assert input_items[0]["role"] == "system"
        # user message
        assert input_items[1]["role"] == "user"
        # AI with tool_calls -> assistant content + function_call
        assert input_items[2]["role"] == "assistant"
        assert input_items[2]["content"] == "Let me check"
        assert input_items[3]["type"] == "function_call"
        assert input_items[3]["name"] == "search"
        # tool result -> function_call_output
        assert input_items[4]["type"] == "function_call_output"
        assert input_items[4]["call_id"] == "tc1"


# ---------------------------------------------------------------------------
# Tests: _astream (Responses SSE)
# ---------------------------------------------------------------------------


class TestAStream:
    """Test _astream consumes Responses terminal events."""

    async def test_stream_returns_result(self, model: ActusResponsesModel) -> None:
        """_astream should yield at least one chunk with the response content."""
        from langchain_core.messages import AIMessageChunk

        mock_resp = _MockResponseObj(_make_responses_api_response(text="Streamed response"))
        async def events():
            yield {"type": "response.completed", "response": mock_resp}

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=events())

        with patch.object(model, "_get_client", return_value=mock_client):
            collected = []
            async for gen_chunk in model._astream([HumanMessage(content="hello")]):
                collected.append(gen_chunk)

        assert len(collected) >= 1
        # Concatenate all content
        full_content = "".join(
            c.message.content for c in collected if c.message.content
        )
        assert "Streamed response" in full_content
        assert mock_client.responses.create.call_args.kwargs["stream"] is True


# ---------------------------------------------------------------------------
# Tests: bind_tools
# ---------------------------------------------------------------------------


class TestBindTools:
    """Test bind_tools returns new instance with tools bound."""

    def test_bind_tools_returns_new_instance(self, model: ActusResponsesModel) -> None:
        """bind_tools should return a new ActusResponsesModel, not mutate the original."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        bound = model.bind_tools([dummy_tool])
        assert isinstance(bound, ActusResponsesModel)
        assert bound is not model

    def test_bind_tools_stores_converted_tools(self, model: ActusResponsesModel) -> None:
        """bind_tools should convert tools to Responses API format and store them."""
        from langchain_core.tools import tool

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"

        bound = model.bind_tools([get_weather])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
        # Should be in Responses API format (name at top level, not nested in "function")
        assert bound._bound_tools[0]["type"] == "function"
        assert bound._bound_tools[0]["name"] == "get_weather"
        assert "function" not in bound._bound_tools[0]

    def test_original_has_no_tools(self, model: ActusResponsesModel) -> None:
        """Original model should remain unchanged after bind_tools."""
        from langchain_core.tools import tool

        @tool
        def dummy(x: str) -> str:
            """Dummy."""
            return x

        model.bind_tools([dummy])
        assert model._bound_tools is None or model._bound_tools == []

    async def test_bound_tools_sent_to_api(self, model: ActusResponsesModel) -> None:
        """After bind_tools, the bound tools should be sent in _agenerate calls."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        bound = model.bind_tools([search])

        mock_resp = _MockResponseObj(_make_responses_api_response(text="found it"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate([HumanMessage(content="search something")])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "search"

    async def test_bound_tools_merged_with_kwargs_tools(
        self, model: ActusResponsesModel
    ) -> None:
        """Per-call tools from kwargs should be merged with bound tools."""
        from langchain_core.tools import tool

        @tool
        def bound_tool(x: str) -> str:
            """A bound tool."""
            return x

        bound = model.bind_tools([bound_tool])

        extra_tool = {
            "type": "function",
            "function": {
                "name": "extra_tool",
                "description": "Extra",
                "parameters": {},
            },
        }

        mock_resp = _MockResponseObj(_make_responses_api_response(text="ok"))
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_resp)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate(
                [HumanMessage(content="test")],
                tools=[extra_tool],
            )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        tool_names = [t["name"] for t in sent_tools]
        assert "bound_tool" in tool_names
        assert "extra_tool" in tool_names


# ---------------------------------------------------------------------------
# Tests: Model properties
# ---------------------------------------------------------------------------


class TestModelProperties:
    """Test that model configuration fields are accessible."""

    def test_model_name_accessible(self, model: ActusResponsesModel) -> None:
        assert model.model_name == "gpt-5.4-pro"

    def test_temperature_accessible(self, model: ActusResponsesModel) -> None:
        assert model.temperature == 0.5

    def test_max_tokens_accessible(self, model: ActusResponsesModel) -> None:
        assert model.max_tokens == 1024

    def test_base_url_accessible(self, model: ActusResponsesModel) -> None:
        assert model.base_url == "https://api.test.com/v1"
