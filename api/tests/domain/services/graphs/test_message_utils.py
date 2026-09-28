"""Tests for message_utils helper functions."""

import json
from copy import deepcopy
from unittest.mock import patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai import AsyncOpenAI

from app.domain.models.memory import Memory
from app.domain.services.graphs.message_utils import dicts_to_messages, messages_to_dicts


@pytest.fixture
def anyio_backend():
    return "asyncio"


class TestProtocolHistoryPersistence:
    def test_only_protocol_fields_are_persisted_and_restored(self):
        msg = AIMessage(
            content="answer",
            additional_kwargs={
                "reasoning_content": "provider reasoning",
                "responses_output_items": [],
                "refusal": "provider refusal",
                "unrelated_metadata": {"debug": "do not persist"},
            },
            response_metadata={"model": "not persisted"},
        )
        stored = messages_to_dicts([msg])
        assert stored == [{
            "role": "assistant", "content": "answer",
            "additional_kwargs": {
                "reasoning_content": "provider reasoning",
                "responses_output_items": [],
                "refusal": "provider refusal",
            },
        }]
        stored[0]["additional_kwargs"]["unexpected"] = "ignore on read too"
        restored = dicts_to_messages(stored)[0]
        assert set(restored.additional_kwargs) == {
            "reasoning_content", "responses_output_items", "refusal",
        }
        assert restored.response_metadata == {}

    def test_native_items_are_deep_copied_in_both_directions(self):
        items = [{
            "type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
            "summary": [{"type": "summary_text", "text": "original"}],
        }]
        msg = AIMessage(content="answer", additional_kwargs={"responses_output_items": items})
        stored = messages_to_dicts([msg])
        stored_items = stored[0]["additional_kwargs"]["responses_output_items"]
        stored_items[0]["summary"][0]["text"] = "stored edit"
        assert msg.additional_kwargs["responses_output_items"][0]["summary"][0]["text"] == "original"
        restored = dicts_to_messages(stored)[0]
        restored.additional_kwargs["responses_output_items"][0]["summary"][0]["text"] = "restored edit"
        assert stored_items[0]["summary"][0]["text"] == "stored edit"

    @pytest.mark.parametrize("kwargs", [None, [], "legacy"])
    def test_legacy_history_without_protocol_fields_remains_valid(self, kwargs):
        stored = [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer", "additional_kwargs": kwargs},
        ]
        restored = dicts_to_messages(stored)
        assert restored[1].additional_kwargs == {}
        assert messages_to_dicts(restored) == [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]

    @pytest.mark.anyio
    @pytest.mark.parametrize("profile_id,preserved", [("glm_5_2_coding", True), ("glm_5_2", False)])
    async def test_glm_history_roundtrip_reaches_wire_with_profile_policy(self, profile_id, preserved):
        from app.domain.services.provider_profiles import get_profile
        from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

        history = [
            HumanMessage("first question"),
            AIMessage(
                content="Checking", additional_kwargs={"reasoning_content": "provider reasoning"},
                tool_calls=[{"id": "call_1", "name": "lookup", "args": {"query": "demo"}}],
            ),
            ToolMessage(content="found", tool_call_id="call_1"),
            HumanMessage("next question"),
        ]
        # The graph persists Memory as JSON and restores it for the next user turn.
        stored = Memory(messages=messages_to_dicts(history)).model_dump_json()
        restored = dicts_to_messages(Memory.model_validate_json(stored).get_messages())
        assert restored[1].additional_kwargs["reasoning_content"] == "provider reasoning"
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, request=request, json={
                "id": "chatcmpl_test", "object": "chat.completion", "created": 1,
                "model": "glm-5.2", "choices": [{
                    "index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "done"},
                }],
            })

        model = ActusChatModel(api_key="test-only", model_name="glm-5.2", profile=get_profile(profile_id))
        async with AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch.object(model, "_get_client", return_value=client):
                await model.ainvoke(restored)
        assistant = requests[0]["messages"][1]
        assert assistant.get("reasoning_content") == ("provider reasoning" if preserved else None)
        assert assistant["tool_calls"][0]["id"] == "call_1"
        assert "additional_kwargs" not in assistant

    @pytest.mark.anyio
    async def test_responses_native_history_survives_json_persistence_and_wire(self):
        from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

        model = ActusResponsesModel(api_key="test-only", model_name="test-model")
        native = [
            {"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque", "summary": [{"type": "summary_text", "text": "summary"}]},
            {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1", "name": "lookup", "arguments": '{"query":"demo"}'},
        ]
        msg = model._response_to_message({"status": "completed", "output": native})
        history = [HumanMessage("question"), msg, ToolMessage(content="found", tool_call_id="call_1"), HumanMessage("next question")]
        stored = Memory(messages=messages_to_dicts(history)).model_dump_json()
        restored = dicts_to_messages(Memory.model_validate_json(stored).get_messages())
        original = deepcopy(restored[1].additional_kwargs)
        requests = []

        async def handler(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, request=request, json={
                "id": "resp_2", "object": "response", "created_at": 1, "model": "test-model", "status": "completed",
                "output": [{"id": "msg_2", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "done", "annotations": []}]}],
            })

        async with AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
            with patch.object(model, "_get_client", return_value=client):
                await model.ainvoke(restored)
        wire = requests[0]["input"]
        assert wire[1:3] == native
        assert sum(item.get("type") == "function_call" for item in wire) == 1
        assert wire[3] == {"type": "function_call_output", "call_id": "call_1", "output": "found"}
        assert restored[1].additional_kwargs == original


class TestDedupMessages:
    def test_same_id_replaced_by_later_message(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        msg1 = HumanMessage(content="first", id="msg-1")
        msg2 = HumanMessage(content="updated", id="msg-1")
        result = dedup_messages([msg1, msg2])

        assert len(result) == 1
        assert result[0].content == "updated"

    def test_no_id_messages_appended(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        sys = SystemMessage(content="system")
        human = HumanMessage(content="hello")
        result = dedup_messages([sys, human])

        assert len(result) == 2
        assert result[0].content == "system"
        assert result[1].content == "hello"

    def test_empty_list(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        result = dedup_messages([])
        assert result == []

    def test_none_id_messages_always_appended(self):
        """Messages with id=None are never deduped (LangChain auto-generates UUIDs,
        so id=None must be set explicitly to trigger this path)."""
        from app.domain.services.graphs.message_utils import dedup_messages

        sys = SystemMessage(content="sys", id=None)
        h1 = HumanMessage(content="v1", id="h-1")
        ai = AIMessage(content="response", id="ai-1")
        h2 = HumanMessage(content="v2", id="h-1")  # replaces h1
        result = dedup_messages([sys, h1, ai, h2])

        assert len(result) == 3
        assert result[0].content == "sys"
        assert result[1].content == "v2"  # replaced
        assert result[2].content == "response"

    def test_preserves_order(self):
        from app.domain.services.graphs.message_utils import dedup_messages

        msgs = [
            HumanMessage(content="a", id="1"),
            AIMessage(content="b", id="2"),
            HumanMessage(content="c", id="3"),
        ]
        result = dedup_messages(msgs)

        assert len(result) == 3
        assert [m.content for m in result] == ["a", "b", "c"]


class TestTruncateToolContent:
    def test_no_truncation_within_limit(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "a" * 8000
        result = truncate_tool_content(content, max_chars=8000)
        assert result == content

    def test_truncation_head_tail(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "H" * 5000 + "M" * 2000 + "T" * 5000  # 12000 chars
        result = truncate_tool_content(content, max_chars=8000)
        assert result.startswith("H")
        assert result.endswith("T")
        assert "已截断" in result
        assert len(result) <= 8000

    def test_boundary_exact_limit(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "x" * 8000
        result = truncate_tool_content(content, max_chars=8000)
        assert result == content

    def test_truncation_with_small_threshold(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "A" * 1000 + "B" * 2000 + "C" * 1000  # 4000 chars
        result = truncate_tool_content(content, max_chars=2000)
        assert result.startswith("A")
        assert result.endswith("C")
        assert len(result) <= 2000

    def test_truncation_marker_contains_char_count(self):
        from app.domain.services.graphs.message_utils import truncate_tool_content

        content = "x" * 10000
        result = truncate_tool_content(content, max_chars=8000)
        assert "已截断" in result
        assert len(result) <= 8000

    def test_result_never_exceeds_max_chars(self):
        """Strict guarantee: result length <= max_chars for all inputs."""
        from app.domain.services.graphs.message_utils import truncate_tool_content

        for max_chars in [50, 100, 200, 500, 2000, 8000]:
            for input_len in [max_chars + 1, max_chars * 2, max_chars * 10]:
                content = "x" * input_len
                result = truncate_tool_content(content, max_chars=max_chars)
                assert len(result) <= max_chars, (
                    f"max_chars={max_chars}, input={input_len}, result={len(result)}"
                )


class TestToolResultMaxCharsConfig:
    """Verify tool_result_max_chars config field defaults and projection."""

    def test_llm_config_default(self):
        from app.domain.models.app_config import LLMConfig
        config = LLMConfig()
        assert config.tool_result_max_chars == 8000

    def test_overflow_config_default(self):
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        config = ContextOverflowConfig()
        assert config.tool_result_max_chars == 8000

    def test_from_llm_config_projection(self):
        from app.domain.models.app_config import LLMConfig
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        llm = LLMConfig(tool_result_max_chars=5000)
        overflow = ContextOverflowConfig.from_llm_config(llm)
        assert overflow.tool_result_max_chars == 5000

    def test_from_llm_config_default_projection(self):
        from app.domain.models.app_config import LLMConfig
        from app.domain.models.context_overflow_config import ContextOverflowConfig
        llm = LLMConfig()
        overflow = ContextOverflowConfig.from_llm_config(llm)
        assert overflow.tool_result_max_chars == 8000
