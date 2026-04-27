"""B4 M0 Phase B2: ActusResponsesModel surfaces OpenAI Responses API `usage`.

The Responses API usage shape differs from Chat Completions:
    usage.input_tokens                          (vs prompt_tokens)
    usage.output_tokens                         (vs completion_tokens)
    usage.input_tokens_details.cached_tokens    (vs prompt_tokens_details.cached_tokens)
    usage.output_tokens_details.reasoning_tokens (vs completion_tokens_details.reasoning_tokens)

Current gap: ``actus_responses_model.py:705`` constructs the AIMessage without
``usage_metadata``. The ``_astream`` path (line 710+) wraps ``_agenerate`` so it
inherits the same gap. These tests lock the contract.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def model() -> ActusResponsesModel:
    return ActusResponsesModel(
        base_url="https://api.test.com/v1",
        api_key="test-key-123",
        model_name="test-model",
        temperature=0.5,
        max_tokens=1024,
    )


def _make_responses_api_response(
    text: str = "hi from responses",
    input_tokens: int = 42,
    output_tokens: int = 8,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> Any:
    """Build a fake OpenAI Responses API response with .model_dump() + .usage."""
    dumped = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": text}],
            }
        ]
    }

    usage_kwargs: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    if cached_tokens is not None:
        usage_kwargs["input_tokens_details"] = SimpleNamespace(
            cached_tokens=cached_tokens
        )
    if reasoning_tokens is not None:
        usage_kwargs["output_tokens_details"] = SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        )

    response = SimpleNamespace(
        model_dump=lambda: dumped,
        usage=SimpleNamespace(**usage_kwargs),
    )
    return response


class TestResponsesModelUsageMetadata:
    """ActusResponsesModel._agenerate must populate AIMessage.usage_metadata."""

    async def test_agenerate_populates_usage_metadata(
        self, model: ActusResponsesModel
    ) -> None:
        mock_response = _make_responses_api_response(
            input_tokens=42, output_tokens=8
        )
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.usage_metadata is not None, (
            "AIMessage.usage_metadata must be populated from Responses API "
            "usage block; currently None — see actus_responses_model.py:705."
        )
        assert msg.usage_metadata["input_tokens"] == 42
        assert msg.usage_metadata["output_tokens"] == 8
        assert msg.usage_metadata["total_tokens"] == 50

    async def test_agenerate_captures_cached_and_reasoning_details(
        self, model: ActusResponsesModel
    ) -> None:
        mock_response = _make_responses_api_response(
            input_tokens=100,
            output_tokens=50,
            cached_tokens=60,
            reasoning_tokens=30,
        )
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.usage_metadata is not None
        input_details = msg.usage_metadata.get("input_token_details") or {}
        output_details = msg.usage_metadata.get("output_token_details") or {}
        assert input_details.get("cache_read") == 60
        assert output_details.get("reasoning") == 30

    async def test_astream_preserves_usage_metadata(
        self, model: ActusResponsesModel
    ) -> None:
        """_astream wraps _agenerate — the single yielded chunk must carry usage_metadata."""
        mock_response = _make_responses_api_response(
            input_tokens=10, output_tokens=3
        )
        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            yielded = [
                gen.message async for gen in model._astream(
                    [HumanMessage(content="hi")]
                )
            ]

        assert yielded, "_astream yielded nothing"
        first: AIMessageChunk = yielded[0]
        assert first.usage_metadata is not None, (
            "_astream (wrapping _agenerate) must preserve usage_metadata on "
            "the yielded AIMessageChunk so CostCallbackHandler sees it."
        )
        assert first.usage_metadata["input_tokens"] == 10
        assert first.usage_metadata["output_tokens"] == 3

    async def test_no_usage_yields_none_not_zeros(
        self, model: ActusResponsesModel
    ) -> None:
        """If provider omits usage, stays None so CostCallbackHandler marks row estimated."""
        dumped = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "no usage"}],
                }
            ]
        }
        response = SimpleNamespace(model_dump=lambda: dumped, usage=None)

        mock_client = AsyncMock()
        mock_client.responses.create = AsyncMock(return_value=response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        assert result.generations[0].message.usage_metadata is None
