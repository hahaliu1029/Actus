"""B4 M0 Phase B1: ActusChatModel._astream propagates the final usage block.

OpenAI SSE streaming with ``stream_options={"include_usage": true}`` sends a
trailing chunk with ``choices=[]`` and a populated ``usage`` block. Without
the fix, the adapter drops that chunk at the ``if not chunk.choices: continue``
guard and B4's ``CostCallbackHandler`` never sees ``usage_metadata`` for any
streamed response — which is 100% of react-loop LLM calls.

These tests lock the contract that the terminal usage chunk becomes an
``AIMessageChunk`` with ``usage_metadata`` populated, so the aggregated
``AIMessage`` at the callback reads provider-reported counters.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, AsyncIterator, List
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from app.infrastructure.external.llm.actus_chat_model import ActusChatModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def model() -> ActusChatModel:
    return ActusChatModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
        supports_response_format=True,
    )


def _make_content_chunks(text: str) -> List[SimpleNamespace]:
    """Build per-character stream chunks culminating in a finish_reason='stop' delta."""
    chunks: List[SimpleNamespace] = []
    for char in text:
        delta = SimpleNamespace(content=char, role=None, tool_calls=None)
        choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
        chunks.append(SimpleNamespace(choices=[choice], usage=None))
    stop_delta = SimpleNamespace(content=None, role=None, tool_calls=None)
    stop_choice = SimpleNamespace(index=0, delta=stop_delta, finish_reason="stop")
    chunks.append(SimpleNamespace(choices=[stop_choice], usage=None))
    return chunks


def _make_usage_only_chunk(
    prompt_tokens: int = 42,
    completion_tokens: int = 8,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> SimpleNamespace:
    """Build OpenAI's trailing usage-only chunk (choices=[] + usage block)."""
    usage_kwargs: dict[str, Any] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }
    if cached_tokens is not None:
        usage_kwargs["prompt_tokens_details"] = SimpleNamespace(
            cached_tokens=cached_tokens
        )
    if reasoning_tokens is not None:
        usage_kwargs["completion_tokens_details"] = SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        )
    return SimpleNamespace(choices=[], usage=SimpleNamespace(**usage_kwargs))


def _make_fake_stream(chunks: List[SimpleNamespace]) -> Any:
    """Wrap a chunk list in an async-iterable matching AsyncOpenAI's stream shape."""

    async def _gen() -> AsyncIterator[SimpleNamespace]:
        for c in chunks:
            yield c

    return _gen()


class TestAstreamUsageMetadata:
    """_astream must surface provider usage on a terminal AIMessageChunk."""

    async def test_final_usage_chunk_produces_usage_metadata_on_aggregate(
        self, model: ActusChatModel
    ) -> None:
        """Aggregating yielded chunks yields an AIMessageChunk with usage_metadata."""
        all_chunks = _make_content_chunks("hi") + [
            _make_usage_only_chunk(prompt_tokens=42, completion_tokens=8)
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_fake_stream(all_chunks)
        )

        with patch.object(model, "_get_client", return_value=mock_client):
            yielded = [
                gen.message async for gen in model._astream(
                    [HumanMessage(content="hi")]
                )
            ]

        assert yielded, "astream yielded nothing"
        aggregated: AIMessageChunk = yielded[0]
        for chunk in yielded[1:]:
            aggregated = aggregated + chunk

        assert aggregated.usage_metadata is not None, (
            "Aggregated AIMessageChunk.usage_metadata must be populated "
            "from the trailing usage chunk; currently the no-choices chunk "
            "is dropped at the `continue` guard in actus_chat_model.py:949."
        )
        assert aggregated.usage_metadata["input_tokens"] == 42
        assert aggregated.usage_metadata["output_tokens"] == 8
        assert aggregated.usage_metadata["total_tokens"] == 50

    async def test_cached_and_reasoning_tokens_surface_on_stream_aggregate(
        self, model: ActusChatModel
    ) -> None:
        """Stream aggregate must preserve cache_read + reasoning detail fields."""
        all_chunks = _make_content_chunks("ok") + [
            _make_usage_only_chunk(
                prompt_tokens=100,
                completion_tokens=50,
                cached_tokens=60,
                reasoning_tokens=30,
            )
        ]

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_fake_stream(all_chunks)
        )

        with patch.object(model, "_get_client", return_value=mock_client):
            yielded = [
                gen.message async for gen in model._astream(
                    [HumanMessage(content="hi")]
                )
            ]

        aggregated: AIMessageChunk = yielded[0]
        for chunk in yielded[1:]:
            aggregated = aggregated + chunk

        assert aggregated.usage_metadata is not None
        input_details = aggregated.usage_metadata.get("input_token_details") or {}
        output_details = aggregated.usage_metadata.get("output_token_details") or {}
        assert input_details.get("cache_read") == 60
        assert output_details.get("reasoning") == 30

    async def test_stream_without_usage_leaves_usage_metadata_none(
        self, model: ActusChatModel
    ) -> None:
        """If the provider doesn't emit a usage chunk, usage_metadata stays None."""
        all_chunks = _make_content_chunks("bye")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_fake_stream(all_chunks)
        )

        with patch.object(model, "_get_client", return_value=mock_client):
            yielded = [
                gen.message async for gen in model._astream(
                    [HumanMessage(content="hi")]
                )
            ]

        aggregated: AIMessageChunk = yielded[0]
        for chunk in yielded[1:]:
            aggregated = aggregated + chunk

        assert aggregated.usage_metadata is None, (
            "When upstream omits the usage block, aggregated usage_metadata "
            "must stay None so CostCallbackHandler flags the row as estimated."
        )

    async def test_astream_requests_include_usage_from_upstream(
        self, model: ActusChatModel
    ) -> None:
        """Adapter must request ``stream_options.include_usage=true`` so upstream actually sends usage.

        Without this, the adapter's ability to parse the final usage chunk is
        academic — OpenAI simply won't send one.
        """
        all_chunks = _make_content_chunks("x")

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_make_fake_stream(all_chunks)
        )

        with patch.object(model, "_get_client", return_value=mock_client):
            async for _ in model._astream([HumanMessage(content="hi")]):
                pass

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        stream_options = call_kwargs.get("stream_options") or {}
        assert stream_options.get("include_usage") is True, (
            f"Expected stream_options.include_usage=true in request, got "
            f"{call_kwargs.get('stream_options')!r}. Without this flag OpenAI "
            "streams never include the usage block and B4 cost tracking goes blind."
        )
