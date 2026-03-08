"""OpenAIResponsesLLM 单元测试。"""

from __future__ import annotations

import pytest

from app.infrastructure.external.llm.openai_responses_llm import OpenAIResponsesLLM


class TestConvertTools:
    """Chat Completions → Responses API 工具格式转换测试。"""

    def test_converts_single_tool(self) -> None:
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "搜索互联网",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "web_search"
        assert result[0]["description"] == "搜索互联网"
        assert result[0]["parameters"]["properties"]["query"]["type"] == "string"
        assert result[0]["strict"] is False

    def test_converts_multiple_tools(self) -> None:
        chat_tools = [
            {"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {}}},
        ]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"

    def test_handles_missing_fields_gracefully(self) -> None:
        chat_tools = [{"type": "function", "function": {}}]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert result[0]["name"] == ""
        assert result[0]["parameters"] == {}

    def test_adds_items_for_array_schema_when_missing(self) -> None:
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "publish",
                    "description": "发布内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title_suggestions": {
                                "type": "array",
                            }
                        },
                    },
                },
            }
        ]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert result[0]["parameters"]["properties"]["title_suggestions"] == {
            "type": "array",
            "items": {},
        }

    def test_adds_items_for_nested_array_schema_when_missing(self) -> None:
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "publish",
                    "description": "发布内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "payload": {
                                "type": "object",
                                "properties": {
                                    "title_suggestions": {
                                        "type": "array",
                                    }
                                },
                            }
                        },
                    },
                },
            }
        ]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert result[0]["parameters"]["properties"]["payload"]["properties"][
            "title_suggestions"
        ] == {
            "type": "array",
            "items": {},
        }

    def test_preserves_existing_array_items_schema(self) -> None:
        chat_tools = [
            {
                "type": "function",
                "function": {
                    "name": "publish",
                    "description": "发布内容",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title_suggestions": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                    },
                },
            }
        ]

        result = OpenAIResponsesLLM._convert_tools(chat_tools)

        assert result[0]["parameters"]["properties"]["title_suggestions"] == {
            "type": "array",
            "items": {"type": "string"},
        }


class TestConvertInputMessages:
    """Chat 风格消息 → Responses API 输入项转换测试。"""

    def test_converts_assistant_tool_calls_to_function_call_items(self) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_456",
                        "type": "function",
                        "function": {
                            "name": "browser_navigate",
                            "arguments": '{"url": "https://example.com"}',
                        },
                    }
                ],
            }
        ]

        result = OpenAIResponsesLLM._convert_input_messages(messages)

        assert result == [
            {
                "type": "function_call",
                "call_id": "call_456",
                "name": "browser_navigate",
                "arguments": '{"url": "https://example.com"}',
            }
        ]

    def test_converts_tool_message_to_function_call_output(self) -> None:
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "function_name": "browser_navigate",
                "content": '{"success": true}',
            }
        ]

        result = OpenAIResponsesLLM._convert_input_messages(messages)

        assert result == [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"success": true}',
            }
        ]


class TestNormalizeResponse:
    """Responses API 响应归一化测试。"""

    def test_normalizes_text_message(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "你好，"},
                        {"type": "output_text", "text": "世界！"},
                    ],
                }
            ]
        }

        result = OpenAIResponsesLLM._normalize_response(response)

        assert result["role"] == "assistant"
        assert result["content"] == "你好，世界！"
        assert "tool_calls" not in result

    def test_normalizes_function_call(self) -> None:
        response = {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_abc123",
                    "name": "web_search",
                    "arguments": '{"query": "python"}',
                }
            ]
        }

        result = OpenAIResponsesLLM._normalize_response(response)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["id"] == "call_abc123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        assert tc["function"]["arguments"] == '{"query": "python"}'

    def test_normalizes_mixed_output(self) -> None:
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "我来搜索一下"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "search",
                    "arguments": "{}",
                },
            ]
        }

        result = OpenAIResponsesLLM._normalize_response(response)

        assert result["content"] == "我来搜索一下"
        assert len(result["tool_calls"]) == 1

    def test_normalizes_empty_output(self) -> None:
        response = {"output": []}

        result = OpenAIResponsesLLM._normalize_response(response)

        assert result["role"] == "assistant"
        assert result["content"] is None
        assert "tool_calls" not in result
