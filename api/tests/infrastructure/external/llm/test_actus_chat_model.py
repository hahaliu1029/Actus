"""Tests for ActusChatModel — BaseChatModel wrapping OpenAI Chat Completions API.

TDD: these tests are written BEFORE the implementation.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult

from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers — mock OpenAI response objects
# ---------------------------------------------------------------------------


def _make_chat_completion(
    content: str | None = "Hello!",
    tool_calls: list[dict[str, Any]] | None = None,
) -> SimpleNamespace:
    """Build a mock ChatCompletion response matching OpenAI SDK structure."""
    message_fields: dict[str, Any] = {
        "role": "assistant",
        "content": content,
        "tool_calls": None,
    }
    if tool_calls:
        message_fields["tool_calls"] = [
            SimpleNamespace(
                id=tc["id"],
                type="function",
                function=SimpleNamespace(
                    name=tc["name"],
                    arguments=tc["arguments"],
                ),
            )
            for tc in tool_calls
        ]
    message = SimpleNamespace(**message_fields)
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
    return SimpleNamespace(
        id="chatcmpl-test",
        choices=[choice],
        model="test-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _make_stream_chunks(
    content: str = "Hello!",
) -> list[SimpleNamespace]:
    """Build mock streaming chunks for a simple text response."""
    chunks = []
    for char in content:
        delta = SimpleNamespace(content=char, role=None, tool_calls=None)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))
    # Final chunk with finish_reason
    final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


def _make_stream_tool_call_chunks(
    tool_id: str = "call_123",
    tool_name: str = "get_weather",
    arguments: str = '{"location": "Paris"}',
) -> list[SimpleNamespace]:
    """Build mock streaming chunks for a tool call response."""
    chunks = []

    # First chunk: tool call start with name
    tc = SimpleNamespace(
        index=0,
        id=tool_id,
        type="function",
        function=SimpleNamespace(name=tool_name, arguments=""),
    )
    delta = SimpleNamespace(content=None, role="assistant", tool_calls=[tc])
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
    chunks.append(SimpleNamespace(choices=[choice]))

    # Argument chunks
    for char in arguments:
        tc = SimpleNamespace(
            index=0,
            id=None,
            type=None,
            function=SimpleNamespace(name=None, arguments=char),
        )
        delta = SimpleNamespace(content=None, role=None, tool_calls=[tc])
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice]))

    # Final chunk
    final_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    final_choice = SimpleNamespace(index=0, delta=final_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[final_choice]))
    return chunks


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model() -> ActusChatModel:
    """Create an ActusChatModel with test configuration."""
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        supports_response_format=True,
    )


@pytest.fixture
def model_no_response_format() -> ActusChatModel:
    """Create an ActusChatModel with supports_response_format=False."""
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        supports_response_format=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMType:
    """Test _llm_type property."""

    def test_llm_type_value(self, model: ActusChatModel) -> None:
        assert model._llm_type == "actus-chat"


class TestSyncGenerate:
    """_generate should raise NotImplementedError (async-only project)."""

    def test_generate_raises(self, model: ActusChatModel) -> None:
        with pytest.raises(NotImplementedError):
            model._generate([HumanMessage(content="hi")])


class TestAGenerate:
    """Test _agenerate — async OpenAI Chat Completions calls."""

    async def test_basic_text_response(self, model: ActusChatModel) -> None:
        """Mock AsyncOpenAI to return a simple text response."""
        mock_response = _make_chat_completion(content="Hello from LLM!")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Hi")])

        assert isinstance(result, ChatResult)
        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.content == "Hello from LLM!"
        assert msg.tool_calls == []

    async def test_tool_calls_response(self, model: ActusChatModel) -> None:
        """Mock AsyncOpenAI to return a response with tool calls."""
        mock_response = _make_chat_completion(
            content="",
            tool_calls=[
                {
                    "id": "call_abc123",
                    "name": "get_weather",
                    "arguments": '{"location": "Paris"}',
                },
            ],
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="Weather?")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert len(msg.tool_calls) == 1
        tc = msg.tool_calls[0]
        assert tc["id"] == "call_abc123"
        assert tc["name"] == "get_weather"
        assert tc["args"] == {"location": "Paris"}

    async def test_multiple_message_types(self, model: ActusChatModel) -> None:
        """Ensure SystemMessage, HumanMessage, AIMessage, ToolMessage all convert."""
        mock_response = _make_chat_completion(content="Done")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

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
            result = await model._agenerate(messages)

        # Verify the call was made with properly converted messages
        call_kwargs = mock_client.chat.completions.create.call_args
        sent_messages = call_kwargs.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"
        assert sent_messages[1]["role"] == "user"
        assert sent_messages[2]["role"] == "assistant"
        assert sent_messages[2]["tool_calls"][0]["function"]["name"] == "search"
        assert sent_messages[3]["role"] == "tool"
        assert sent_messages[3]["tool_call_id"] == "tc1"

    async def test_response_format_passed_when_supported(
        self, model: ActusChatModel
    ) -> None:
        """response_format should be passed when supports_response_format=True."""
        mock_response = _make_chat_completion(content='{"answer": 42}')
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="answer")],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    async def test_response_format_not_passed_when_unsupported(
        self, model_no_response_format: ActusChatModel
    ) -> None:
        """response_format should NOT be passed when supports_response_format=False."""
        mock_response = _make_chat_completion(content="text only")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(
            model_no_response_format, "_get_client", return_value=mock_client
        ):
            await model_no_response_format._agenerate(
                [HumanMessage(content="answer")],
                response_format={"type": "json_object"},
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "response_format" not in call_kwargs

    async def test_tools_from_kwargs(self, model: ActusChatModel) -> None:
        """Per-call tools passed via kwargs should be sent to OpenAI."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "test_tool"}}]

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tools=tools,
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"] == tools

    async def test_tool_choice_from_kwargs(self, model: ActusChatModel) -> None:
        """tool_choice from kwargs should be passed to OpenAI."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="auto",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "auto"

    async def test_tool_choice_any_normalized_to_required(self, model: ActusChatModel) -> None:
        """LangChain's 'any' tool_choice should be normalized to 'required' for OpenAI API."""
        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            await model._agenerate(
                [HumanMessage(content="hi")],
                tool_choice="any",
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tool_choice"] == "required"


class TestBindTools:
    """Test bind_tools returns new instance with tools bound."""

    def test_bind_tools_returns_new_instance(self, model: ActusChatModel) -> None:
        """bind_tools should return a new ActusChatModel, not mutate the original."""
        from langchain_core.tools import tool

        @tool
        def dummy_tool(x: str) -> str:
            """A dummy tool."""
            return x

        bound = model.bind_tools([dummy_tool])
        assert isinstance(bound, ActusChatModel)
        assert bound is not model

    def test_bind_tools_stores_converted_tools(self, model: ActusChatModel) -> None:
        """bind_tools should convert tools to OpenAI format and store them."""
        from langchain_core.tools import tool

        @tool
        def get_weather(location: str) -> str:
            """Get weather for a location."""
            return f"Sunny in {location}"

        bound = model.bind_tools([get_weather])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
        assert bound._bound_tools[0]["type"] == "function"
        assert bound._bound_tools[0]["function"]["name"] == "get_weather"

    def test_original_has_no_tools(self, model: ActusChatModel) -> None:
        """Original model should remain unchanged after bind_tools."""
        from langchain_core.tools import tool

        @tool
        def dummy(x: str) -> str:
            """Dummy."""
            return x

        model.bind_tools([dummy])
        assert model._bound_tools is None or model._bound_tools == []

    async def test_bound_tools_sent_to_openai(self, model: ActusChatModel) -> None:
        """After bind_tools, the bound tools should be sent in _agenerate calls."""
        from langchain_core.tools import tool

        @tool
        def search(query: str) -> str:
            """Search the web."""
            return "results"

        bound = model.bind_tools([search])

        mock_response = _make_chat_completion(content="found it")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate([HumanMessage(content="search something")])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "search"

    async def test_bound_tools_merged_with_kwargs_tools(
        self, model: ActusChatModel
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
            "function": {"name": "extra_tool", "parameters": {}},
        }

        mock_response = _make_chat_completion(content="ok")
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(bound, "_get_client", return_value=mock_client):
            await bound._agenerate(
                [HumanMessage(content="test")],
                tools=[extra_tool],
            )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        sent_tools = call_kwargs["tools"]
        tool_names = [t["function"]["name"] for t in sent_tools]
        assert "bound_tool" in tool_names
        assert "extra_tool" in tool_names


class TestAStream:
    """Test _astream — streaming from OpenAI Chat Completions."""

    async def test_stream_text_response(self, model: ActusChatModel) -> None:
        """Streaming a simple text response should yield AIMessageChunk objects."""
        from langchain_core.messages import AIMessageChunk

        chunks = _make_stream_chunks("Hi!")

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = mock_stream

        with patch.object(model, "_get_client", return_value=mock_client):
            collected = []
            async for gen_chunk in model._astream([HumanMessage(content="hello")]):
                collected.append(gen_chunk)

        # Should have yielded chunks with content
        contents = [
            c.message.content for c in collected if c.message.content
        ]
        assert "".join(contents) == "Hi!"

    async def test_stream_tool_call_response(self, model: ActusChatModel) -> None:
        """Streaming a tool call response should yield chunks that aggregate."""
        from langchain_core.messages import AIMessageChunk

        chunks = _make_stream_tool_call_chunks(
            tool_id="call_456",
            tool_name="get_weather",
            arguments='{"city": "NYC"}',
        )

        async def mock_stream(*args, **kwargs):
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.chat.completions.create = mock_stream

        with patch.object(model, "_get_client", return_value=mock_client):
            collected = []
            async for gen_chunk in model._astream([HumanMessage(content="weather")]):
                collected.append(gen_chunk)

        # At least one chunk should have tool_call_chunks
        has_tool_chunks = any(
            c.message.tool_call_chunks for c in collected
            if hasattr(c.message, "tool_call_chunks") and c.message.tool_call_chunks
        )
        assert has_tool_chunks


class TestModelProperties:
    """Test that model configuration fields are accessible."""

    def test_model_name_accessible(self, model: ActusChatModel) -> None:
        assert model.model_name == "test-model"

    def test_temperature_accessible(self, model: ActusChatModel) -> None:
        assert model.temperature == 0.5

    def test_max_tokens_accessible(self, model: ActusChatModel) -> None:
        assert model.max_tokens == 1024

    def test_supports_response_format_accessible(self, model: ActusChatModel) -> None:
        assert model.supports_response_format is True


# ---------------------------------------------------------------------------
# Tests — Fallback tool call extraction from content
# ---------------------------------------------------------------------------

def _make_model_with_tools(tool_names: list[str]) -> ActusChatModel:
    """Create an ActusChatModel with _bound_tools and _bound_tool_names set."""
    model = ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key",
        model_name="test-model",
    )
    model._bound_tools = [
        {"type": "function", "function": {"name": name, "parameters": {}}}
        for name in tool_names
    ]
    model._bound_tool_names = frozenset(tool_names)
    return model


class TestContentToolEnvelope:
    @pytest.mark.parametrize("profile", ["generic_openai", "openai_official", "minimax"])
    @pytest.mark.parametrize("content", [
        '{"name":"file_read","arguments":{"filepath":"/etc/hosts"}}',
        'Example only; do not execute:\n```json\n{"name":"file_read","arguments":{}}\n```',
        '<invoke name="file_read"><filepath>/tmp/a</filepath></invoke>',
        'Example:\nminimax:tool_call\n<invoke name="file_read"><filepath>/tmp/a</filepath></invoke>',
        '```xml\n<minimax:tool_call><invoke name="file_read"/></minimax:tool_call>\n```',
        'Normal response with no calls.',
    ])
    def test_text_examples_never_become_calls(self, profile, content):
        from app.domain.services.provider_profiles import get_profile
        model = _make_model_with_tools(["file_read"])
        model.profile = get_profile(profile)
        assert model._extract_tool_calls_from_content(content) == ([], content)

    @pytest.mark.parametrize("envelope", [
        'minimax:tool_call\n{body}', '<minimax:tool_call>{body}</minimax:tool_call>',
    ])
    def test_complete_opted_in_envelope(self, envelope):
        from app.domain.services.provider_profiles import get_profile
        model = _make_model_with_tools(["file_read", "file_write"])
        model.profile = get_profile("minimax")
        body = ('<invoke name="file_read"><parameter name="filepath">/tmp/a</parameter></invoke>'
                '<invoke name="file_write"><append>true</append><count>5</count></invoke>')
        calls, content = model._extract_tool_calls_from_content(envelope.format(body=body))
        assert content == ""
        assert [call["name"] for call in calls] == ["file_read", "file_write"]
        assert [call["args"] for call in calls] == [{"filepath": "/tmp/a"}, {"append": True, "count": 5}]
        assert len({call["id"] for call in calls}) == 2

    def test_marker_requires_profile_opt_in(self):
        model = _make_model_with_tools(["file_read"])
        content = 'minimax:tool_call\n<invoke name="file_read"/>'
        assert model._extract_tool_calls_from_content(content) == ([], content)

    @pytest.mark.parametrize("body", [
        '<invoke name="file_read"><filepath>/tmp/a</filepath>',
        '<invoke name="file_read"/><invoke name="unknown"/>',
        '<invoke name="file_read"><x><nested/></x></invoke>',
        '<invoke name="file_read"><x>1</x><x>2</x></invoke>',
        '<!DOCTYPE test [<!ENTITY a "x">]><invoke name="file_read"/>',
        '<invoke name="file_read"/> explanation',
        'x' * 32001,
    ])
    def test_ambiguous_or_invalid_envelope_kept_as_text(self, body):
        from app.domain.services.provider_profiles import get_profile
        model = _make_model_with_tools(["file_read"])
        model.profile = get_profile("minimax")
        content = f"minimax:tool_call\n{body}"
        assert model._extract_tool_calls_from_content(content) == ([], content)


class TestFallbackIntegrationInAgenerate:
    """Test that _agenerate uses fallback extraction when API returns no tool_calls."""

    async def test_xml_tool_call_in_content_extracted(self) -> None:
        """When API returns no tool_calls but content has XML, fallback extracts it."""
        model = _make_model_with_tools(["mcp_MiniMax_understand_image"])

        from app.domain.services.provider_profiles import get_profile
        model.profile = get_profile("minimax")
        xml_content = (
            "minimax:tool_call\n"
            '<invoke name="mcp_MiniMax_understand_image">'
            "<image_url>http://example.com/img.png</image_url>"
            "</invoke>"
        )
        mock_response = _make_chat_completion(content=xml_content, tool_calls=None)
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="分析图片")])

        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "mcp_MiniMax_understand_image"
        assert msg.tool_calls[0]["args"]["image_url"] == "http://example.com/img.png"
        # Content should be cleaned
        assert "<invoke" not in msg.content

    async def test_structured_tool_calls_take_precedence(self) -> None:
        """When API returns structured tool_calls, fallback is NOT invoked."""
        model = _make_model_with_tools(["get_weather"])

        # Content also has XML, but structured tool_calls are present
        content_with_xml = '<invoke name="get_weather"><location>NYC</location></invoke>'
        mock_response = _make_chat_completion(
            content=content_with_xml,
            tool_calls=[{
                "id": "call_real",
                "name": "get_weather",
                "arguments": '{"location": "Paris"}',
            }],
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="weather?")])

        msg = result.generations[0].message
        # Should use the structured tool call, not the XML one
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["id"] == "call_real"
        assert msg.tool_calls[0]["args"]["location"] == "Paris"
        # Content should still contain the XML (fallback not invoked)
        assert "<invoke" in msg.content
