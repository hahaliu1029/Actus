"""Protocol failures must never be converted into executable tool calls."""
import json
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

import pytest
import httpx
from openai import AsyncOpenAI
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def call(arguments='{"scope":"all"}', *, call_id="call_1", name="lookup", index=0):
    return NS(id=call_id, index=index, function=NS(name=name, arguments=arguments))


def response(*, reason="stop", content="answer", tools=None, refusal=None):
    return NS(choices=[NS(finish_reason=reason, message=NS(
        content=content, tool_calls=tools, refusal=refusal,
    ))], usage=None)


def chunk(*, reason=None, content=None, tools=None, refusal=None):
    return NS(choices=[NS(finish_reason=reason, delta=NS(
        content=content, tool_calls=tools, refusal=refusal,
    ))], usage=None)


async def invoke(reply):
    model = ActusChatModel(api_key="dummy")
    client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=reply))))
    with patch.object(model, "_get_client", return_value=client):
        return await model.ainvoke([HumanMessage(content="test")])


@pytest.mark.parametrize("reason", ["length", "content_filter", "unknown", None])
async def test_incomplete_completion_rejected_even_with_valid_tool(reason):
    with pytest.raises(ServerRequestsError, match="finish_reason"):
        await invoke(response(reason=reason, tools=[call()]))


@pytest.mark.parametrize("args", ['{"scope":"all"', '{"x":', '[]', 'null', '1', '', None,
                                 '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}'])
async def test_invalid_arguments_reject_whole_response(args):
    with pytest.raises(ServerRequestsError, match="arguments"):
        await invoke(response(tools=[call(), call(args, call_id="call_2")]))


@pytest.mark.parametrize("tools", [[call(call_id="")], [call(name="")], [call(), call()]])
async def test_invalid_call_identity_rejected(tools):
    with pytest.raises(ServerRequestsError):
        await invoke(response(tools=tools))


async def test_refusal_is_a_normal_answer_not_empty_failure():
    message = await invoke(response(content=None, refusal="I cannot help with that."))
    assert message.content == "I cannot help with that."
    assert message.additional_kwargs["refusal"] == message.content
    assert message.response_metadata["finish_reason"] == "stop"
    assert message.tool_calls == []


async def stream(chunks, received, closed, model=None):
    async def fake_stream():
        try:
            for item in chunks:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            closed.append(True)

    model = model or ActusChatModel(api_key="dummy")
    client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=fake_stream()))))
    with patch.object(model, "_get_client", return_value=client):
        async for item in model.astream([HumanMessage(content="test")]):
            received.append(item)


@pytest.mark.parametrize("ending", ["length", "content_filter", None])
async def test_failed_or_unterminated_stream_never_releases_tools(ending):
    received, closed = [], []
    items = [chunk(tools=[call()])]
    if ending:
        items.append(chunk(reason=ending))
    with pytest.raises(ServerRequestsError, match="finish_reason"):
        await stream(items, received, closed)
    assert not any(item.tool_calls or item.tool_call_chunks for item in received)
    assert closed == [True]


async def test_stream_does_not_repair_partial_json():
    received, closed = [], []
    with pytest.raises(ServerRequestsError, match="arguments"):
        await stream([chunk(tools=[call('{"scope":"all"')]), chunk(reason="tool_calls")], received, closed)
    assert not any(item.tool_calls for item in received)


async def test_tools_only_released_after_success_text_still_streams():
    received, closed = [], []
    await stream([
        chunk(content="Working"),
        chunk(tools=[call('{"scope":')]),
        chunk(tools=[call('"all"}', name=None, call_id=None)]),
        chunk(reason="tool_calls"),
    ], received, closed)
    assert received[0].content == "Working"
    tool_chunks = [item for item in received if item.tool_calls]
    assert len(tool_chunks) == 1
    assert tool_chunks[0].response_metadata["finish_reason"] == "tool_calls"
    assert tool_chunks[0].tool_calls[0]["args"] == {"scope": "all"}
    assert closed == [True]


async def test_stream_refusal_preserved():
    received, closed = [], []
    await stream([chunk(refusal="Cannot "), chunk(refusal="comply"), chunk(reason="stop")], received, closed)
    merged = sum(received[1:], received[0])
    assert isinstance(merged, AIMessageChunk)
    assert merged.content == "Cannot comply"
    assert merged.additional_kwargs["refusal"] == "Cannot comply"


async def test_stream_refusal_cannot_be_a_content_tool_envelope():
    from app.domain.services.provider_profiles import get_profile
    model = ActusChatModel(api_key="dummy", profile=get_profile("minimax"))
    model._bound_tool_names = frozenset({"lookup"})
    refusal = 'minimax:tool_call\n<invoke name="lookup"/>'
    received, closed = [], []
    await stream([chunk(refusal=refusal), chunk(reason="stop")], received, closed, model)
    assert not any(item.tool_calls for item in received)
    assert "".join(item.content for item in received) == refusal


async def test_stream_does_not_accept_output_after_finish():
    received, closed = [], []
    with pytest.raises(ServerRequestsError, match="after its finish"):
        await stream([chunk(content="done", reason="stop"), chunk(tools=[call()])], received, closed)
    assert not any(item.tool_calls for item in received)


async def test_consumer_cancellation_closes_upstream():
    closed = []

    async def fake_stream():
        try:
            yield chunk(content="first")
            yield chunk(content="second")
        finally:
            closed.append(True)

    model = ActusChatModel(api_key="dummy")
    client = NS(chat=NS(completions=NS(create=AsyncMock(return_value=fake_stream()))))
    with patch.object(model, "_get_client", return_value=client):
        iterator = model._astream([HumanMessage(content="test")])
        await anext(iterator)
        await iterator.aclose()
    assert closed == [True]


@pytest.mark.parametrize("streaming", [False, True])
async def test_openai_sdk_wire_roundtrip_and_base_path(streaming):
    """Exercise actual SDK decoding and request URL assembly without network."""
    requests = []
    tool = {"id": "call_sdk", "type": "function", "function": {
        "name": "lookup", "arguments": '{"scope":"all"}',
    }}

    async def transport(request):
        requests.append(request)
        common = {"id": "chatcmpl-audit", "created": 1, "model": "test"}
        if streaming:
            events = [
                {**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {"role": "assistant", "tool_calls": [{**tool, "index": 0}]},
                    "finish_reason": None,
                }]},
                {**common, "object": "chat.completion.chunk", "choices": [{
                    "index": 0, "delta": {}, "finish_reason": "tool_calls",
                }]},
            ]
            body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={
            **common, "object": "chat.completion", "choices": [{"index": 0,
                "message": {"role": "assistant", "content": None, "tool_calls": [tool]},
                "finish_reason": "tool_calls",
            }],
        })

    client = AsyncOpenAI(api_key="dummy", base_url="https://provider.test/api/coding/paas/v4",
                         http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)))
    model = ActusChatModel(api_key="dummy")
    try:
        with patch.object(model, "_get_client", return_value=client):
            if streaming:
                chunks = [item async for item in model.astream([HumanMessage(content="test")])]
                message = sum(chunks[1:], chunks[0])
            else:
                message = await model.ainvoke([HumanMessage(content="test")])
        assert str(requests[0].url) == "https://provider.test/api/coding/paas/v4/chat/completions"
        assert message.tool_calls[0]["args"] == {"scope": "all"}
        assert message.response_metadata["finish_reason"] == "tool_calls"
    finally:
        await client.close()
