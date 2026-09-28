"""Responses wire contracts and terminal-state safety, without network or keys."""
from __future__ import annotations

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import openai
import pytest
from langchain_core.messages import HumanMessage, ToolMessage, message_chunk_to_message
from openai import AsyncOpenAI

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def model():
    return ActusResponsesModel(api_key="test-only-key", model_name="test-model")


def text_item(text="hello", *, refusal=False):
    part = {"type": "refusal", "refusal": text} if refusal else {
        "type": "output_text", "text": text, "annotations": [],
    }
    return {"id": "msg_1", "type": "message", "role": "assistant", "status": "completed", "content": [part]}


def call_item(arguments='{"query":"demo"}'):
    return {"id": "fc_1", "type": "function_call", "status": "completed", "call_id": "call_1", "name": "lookup", "arguments": arguments}


def response(output=None, **overrides):
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "test-model", "status": "completed", "output": [text_item()] if output is None else output, **overrides}


async def invoke(model, payload):
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=payload)))
    with patch.object(model, "_get_client", return_value=client):
        return await model.ainvoke([HumanMessage("question")])


class EventStream:
    def __init__(self, events):
        self.events = events
        self.closed = False

    async def __aiter__(self):
        for event in self.events:
            if isinstance(event, Exception):
                raise event
            yield event

    async def close(self):
        self.closed = True


def stream_client(events):
    stream = EventStream(events)
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=stream)))
    return client, stream


@pytest.mark.parametrize("status", [None, "incomplete", "failed", "cancelled", "in_progress", "queued", "unknown"])
async def test_non_completed_response_never_admits_partial_text_or_tools(model, status):
    with pytest.raises(ServerRequestsError, match="non-completed"):
        await invoke(model, response([text_item("partial"), call_item()], status=status))


@pytest.mark.parametrize("field", ["error", "incomplete_details"])
@pytest.mark.parametrize("details", [{}, {"reason": "max_output_tokens"}])
async def test_error_details_reject_even_completed_header(model, field, details):
    with pytest.raises(ServerRequestsError):
        await invoke(model, response(**{field: details}))


async def test_missing_response_status_rejected(model):
    payload = response([call_item()])
    del payload["status"]
    with pytest.raises(ServerRequestsError, match="non-completed"):
        await invoke(model, payload)


async def test_incomplete_output_item_rejected(model):
    item = call_item()
    item["status"] = "incomplete"
    with pytest.raises(ServerRequestsError, match="incomplete output item"):
        await invoke(model, response([item]))


@pytest.mark.parametrize("content", [None, "text", {}, ["text"], [{"type": "output_text", "text": None}], [{"type": "refusal", "refusal": []}]])
async def test_malformed_nested_message_is_a_protocol_error(model, content):
    with pytest.raises(ServerRequestsError):
        await invoke(model, response([{**text_item(), "content": content}]))


async def test_stop_sequences_rejected_before_sdk_call(model):
    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock()))
    with patch.object(model, "_get_client", return_value=client):
        with pytest.raises(ValueError, match="does not support stop"):
            await model.ainvoke([HumanMessage("q")], stop=["END"])
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("arguments", ['{"query":"demo"', "[]", "null", "NaN", '{"x":NaN}', '{"x":Infinity}', {"x": float("nan")}, ""])
async def test_tool_arguments_must_be_complete_json_object(model, arguments):
    with pytest.raises(ServerRequestsError):
        await invoke(model, response([call_item(arguments)]))


@pytest.mark.parametrize("field", ["call_id", "arguments", "name"])
async def test_tool_identity_and_arguments_cannot_be_invented(model, field):
    item = call_item()
    del item[field]
    with pytest.raises(ServerRequestsError):
        await invoke(model, response([item]))


async def test_duplicate_tool_call_id_rejected(model):
    with pytest.raises(ServerRequestsError, match="duplicate"):
        await invoke(model, response([call_item(), {**call_item(), "id": "fc_2"}]))


@pytest.mark.parametrize("call_id", ["", " ", 123, None])
async def test_invalid_tool_call_id_rejected(model, call_id):
    with pytest.raises(ServerRequestsError):
        await invoke(model, response([{**call_item(), "call_id": call_id}]))


async def test_refusal_is_a_normal_response(model):
    msg = await invoke(model, response([text_item("I cannot do that.", refusal=True)]))
    assert msg.content == "I cannot do that."
    assert msg.additional_kwargs["refusal"] == "I cannot do that."
    assert msg.tool_calls == []
    assert msg.response_metadata["status"] == "completed"


async def test_reasoning_round_trip_preserves_native_order_and_encrypted_content(model):
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "summary"}], "encrypted_content": "opaque-test-payload"}
    original = [reasoning, text_item("Checking"), call_item()]
    msg = await invoke(model, response(original))
    history = model._convert_input_messages([HumanMessage("question"), msg, ToolMessage(content="found", tool_call_id="call_1")])
    assert history[1:4] == original
    assert sum(item.get("type") == "function_call" for item in history) == 1
    assert history[-1] == {"type": "function_call_output", "call_id": "call_1", "output": "found"}
    history[1]["encrypted_content"] = "mutated"
    assert msg.additional_kwargs["responses_output_items"][0]["encrypted_content"] == "opaque-test-payload"


async def test_edited_message_does_not_restore_stale_native_content(model):
    msg = await invoke(model, response([text_item("old response")]))
    msg.content = "edited response"
    assert model._convert_input_messages([msg]) == [{"role": "assistant", "content": "edited response"}]


async def test_edited_tools_do_not_restore_stale_native_calls(model):
    reasoning = {"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    msg = await invoke(model, response([reasoning, text_item(), call_item()]))
    msg.tool_calls = []
    history = model._convert_input_messages([msg])
    assert history == [reasoning, {"role": "assistant", "content": "hello"}]


@pytest.mark.parametrize("choice", ["lookup", {"type": "function", "name": "lookup"}])
async def test_named_tool_choice_variants(model, choice):
    params = model._build_request_params([HumanMessage("q")], tool_choice=choice)
    assert params["tool_choice"] == {"type": "function", "name": "lookup"}


async def test_json_schema_and_named_tool_choice_use_responses_wire_shape(model):
    captured = []

    async def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response(), request=request)

    schema = {"name": "Answer", "strict": True, "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}}
    original = deepcopy(schema)
    async with AsyncOpenAI(api_key="test-only-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
        with patch.object(model, "_get_client", return_value=client):
            await model._agenerate(
                [HumanMessage("question")],
                response_format={"type": "json_schema", "json_schema": schema},
                text={"verbosity": "low"},
                tool_choice={"type": "function", "function": {"name": "lookup"}},
            )
    assert captured[0]["text"] == {"verbosity": "low", "format": {"type": "json_schema", **schema}}
    assert captured[0]["tool_choice"] == {"type": "function", "name": "lookup"}
    assert schema == original


async def test_stateless_requests_include_encrypted_reasoning_once(model):
    requested = ["message.output_text.logprobs"]
    params = model._build_request_params([HumanMessage("q")], store=False, include=requested)
    assert params["include"] == [*requested, "reasoning.encrypted_content"]
    assert requested == ["message.output_text.logprobs"]
    params = model._build_request_params([HumanMessage("q")], store=False, include=params["include"])
    assert params["include"].count("reasoning.encrypted_content") == 1


async def test_stream_yields_text_before_completion_and_usage_only_once(model):
    release = asyncio.Event()
    closed = False
    final_response = response([text_item("hello world"), call_item()], usage={"input_tokens": 10, "output_tokens": 4, "total_tokens": 14})

    async def events():
        nonlocal closed
        try:
            yield {"type": "response.output_text.delta", "delta": "hello "}
            await release.wait()
            yield {"type": "response.output_text.delta", "delta": "world"}
            yield {"type": "response.completed", "response": final_response}
        finally:
            closed = True

    client = SimpleNamespace(responses=SimpleNamespace(create=AsyncMock(return_value=events())))
    with patch.object(model, "_get_client", return_value=client):
        stream = model.astream([HumanMessage("q")])
        first = await asyncio.wait_for(anext(stream), timeout=1)
        assert first.content == "hello "
        assert not first.tool_calls
        release.set()
        chunks = [first, *[chunk async for chunk in stream]]
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    msg = message_chunk_to_message(aggregate)
    assert msg.content == "hello world"
    assert msg.tool_calls[0]["args"] == {"query": "demo"}
    assert msg.usage_metadata["total_tokens"] == 14
    assert sum(chunk.usage_metadata is not None for chunk in chunks) == 1
    assert msg.response_metadata["status"] == "completed"
    assert closed
    assert client.responses.create.call_args.kwargs["stream"] is True


@pytest.mark.parametrize("terminal", ["response.incomplete", "response.failed", "error", None])
async def test_failed_or_unterminated_stream_never_exposes_tool_chunks(model, terminal):
    events = [
        {"type": "response.output_text.delta", "delta": "partial"},
        {"type": "response.output_item.added", "output_index": 0, "item": call_item('')},
        {"type": "response.function_call_arguments.delta", "delta": '{"query":"demo"', "output_index": 0},
    ]
    if terminal:
        events.append({"type": terminal, "response": response([call_item()], status="incomplete"), "code": "test_error"})
    client, stream = stream_client(events)
    chunks = []
    with patch.object(model, "_get_client", return_value=client):
        with pytest.raises(ServerRequestsError):
            async for chunk in model._astream([HumanMessage("q")]):
                chunks.append(chunk.message)
    assert all(not chunk.tool_call_chunks and not chunk.tool_calls for chunk in chunks)
    assert stream.closed


async def test_completed_stream_still_requires_strict_tool_arguments(model):
    client, stream = stream_client([{"type": "response.completed", "response": response([call_item('{"query":"demo"')])}])
    with patch.object(model, "_get_client", return_value=client):
        with pytest.raises(ServerRequestsError, match="invalid JSON"):
            await anext(model._astream([HumanMessage("q")]))
    assert stream.closed


async def test_stream_refusal_preserves_content_without_duplicate_text(model):
    final = response([text_item("Cannot help", refusal=True)])
    client, stream = stream_client([
        {"type": "response.refusal.delta", "delta": "Cannot "},
        {"type": "response.refusal.delta", "delta": "help"},
        {"type": "response.completed", "response": final},
    ])
    with patch.object(model, "_get_client", return_value=client):
        chunks = [chunk async for chunk in model.astream([HumanMessage("q")])]
    aggregate = chunks[0]
    for chunk in chunks[1:]:
        aggregate += chunk
    assert aggregate.content == "Cannot help"
    assert aggregate.additional_kwargs["refusal"] == "Cannot help"
    assert stream.closed


async def test_mid_stream_sdk_error_translated_and_stream_closed(model):
    error = openai.APIError("test SSE error", request=httpx.Request("POST", "https://example.invalid/v1/responses"), body=None)
    client, stream = stream_client([{"type": "response.output_text.delta", "delta": "partial"}, error])
    with patch.object(model, "_get_client", return_value=client):
        with pytest.raises(ServerRequestsError, match="transient transport"):
            _ = [chunk async for chunk in model._astream([HumanMessage("q")])]
    assert stream.closed


async def test_actual_sdk_sse_keeps_reasoning_and_tool_call_round_trip(model):
    native = [{"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "opaque-test-value"}, call_item()]
    events = [
        {"type": "response.output_item.added", "output_index": 1, "sequence_number": 0, "item": {**call_item(), "arguments": "", "status": "in_progress"}},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "sequence_number": 1, "item_id": "fc_1", "delta": '{"query":'},
        {"type": "response.function_call_arguments.delta", "output_index": 1, "sequence_number": 2, "item_id": "fc_1", "delta": '"demo"}'},
        {"type": "response.completed", "sequence_number": 3, "response": response(native)},
    ]
    received = []

    async def handler(request):
        received.append(json.loads(request.content))
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    async with AsyncOpenAI(api_key="test-only-key", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))) as client:
        with patch.object(model, "_get_client", return_value=client):
            chunks = [chunk async for chunk in model.astream([HumanMessage("q")], store=False)]
    assert received[0]["stream"] is True
    final = chunks[0]
    for chunk in chunks[1:]:
        final += chunk
    history = model._convert_input_messages([message_chunk_to_message(final), ToolMessage("found", tool_call_id="call_1")])
    assert history[0]["encrypted_content"] == "opaque-test-value"
    assert [item["type"] for item in history] == ["reasoning", "function_call", "function_call_output"]
