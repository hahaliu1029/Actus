"""B4 M0 smoke gate: ActusChatModel surfaces OpenAI `usage` as AIMessage.usage_metadata.

Design assumption (load-bearing): when the OpenAI Chat Completions SDK returns a
response with a ``usage`` block, ``ActusChatModel`` must translate those counters
into a LangChain-standard ``UsageMetadata`` dict on the returned ``AIMessage``.

The per-session cost ledger (B4) has no other signal for the non-fallback path:
``CostCallbackHandler.on_llm_end`` reads ``response.generations[0].message.usage_metadata``.
If that field is ``None``, every non-streaming LLM call records either zeros
(silent under-counting) or an estimated row (user-visible partial badge).

Current gap: ``actus_chat_model.py:725-729`` constructs the ``AIMessage`` with
``content`` / ``tool_calls`` / ``additional_kwargs`` only; ``usage_metadata`` is
never set. These tests must FAIL until that gap is closed.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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


def _make_chat_completion_with_usage(
    prompt_tokens: int = 42,
    completion_tokens: int = 17,
    cached_tokens: int | None = None,
    reasoning_tokens: int | None = None,
) -> SimpleNamespace:
    """Build a ChatCompletion-shaped mock mirroring OpenAI SDK usage fields.

    ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens`` follow the OpenAI Python
    SDK's ``CompletionUsage`` surface.
    """
    message = SimpleNamespace(
        role="assistant",
        content="hello with usage",
        tool_calls=None,
    )
    choice = SimpleNamespace(index=0, message=message, finish_reason="stop")

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

    return SimpleNamespace(
        id="chatcmpl-test-usage",
        choices=[choice],
        model="test-model",
        usage=SimpleNamespace(**usage_kwargs),
    )


class TestUsageMetadataPopulation:
    """B4 contract: usage_metadata must mirror provider-reported counters."""

    async def test_basic_input_output_total(self, model: ActusChatModel) -> None:
        """Happy path: usage block on response → populated usage_metadata."""
        mock_response = _make_chat_completion_with_usage(
            prompt_tokens=42, completion_tokens=17
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert isinstance(msg, AIMessage)
        assert msg.usage_metadata is not None, (
            "AIMessage.usage_metadata must be populated so CostCallbackHandler "
            "can persist a CostRecord; currently None — see actus_chat_model.py:725-729."
        )
        assert msg.usage_metadata["input_tokens"] == 42
        assert msg.usage_metadata["output_tokens"] == 17
        assert msg.usage_metadata["total_tokens"] == 59

    async def test_cached_tokens_surface_as_cache_read(
        self, model: ActusChatModel
    ) -> None:
        """``prompt_tokens_details.cached_tokens`` → ``input_token_details.cache_read``.

        Required for the ``cache_read`` dim of the 5-dim B4 cost model.
        """
        mock_response = _make_chat_completion_with_usage(
            prompt_tokens=100, completion_tokens=20, cached_tokens=60
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.usage_metadata is not None
        details = msg.usage_metadata.get("input_token_details") or {}
        assert details.get("cache_read") == 60, (
            f"Expected cache_read=60 in input_token_details, got {details!r}. "
            "B4 needs this to avoid over-billing cached prefix."
        )

    async def test_reasoning_tokens_surface_in_output_details(
        self, model: ActusChatModel
    ) -> None:
        """``completion_tokens_details.reasoning_tokens`` → ``output_token_details.reasoning``.

        Required for the ``reasoning`` dim of the 5-dim B4 cost model (o1-style models).
        """
        mock_response = _make_chat_completion_with_usage(
            prompt_tokens=10, completion_tokens=50, reasoning_tokens=30
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.usage_metadata is not None
        details = msg.usage_metadata.get("output_token_details") or {}
        assert details.get("reasoning") == 30, (
            f"Expected reasoning=30 in output_token_details, got {details!r}. "
            "B4 needs this to price o1-style reasoning separately."
        )

    async def test_no_usage_block_yields_none_not_zeros(
        self, model: ActusChatModel
    ) -> None:
        """If the provider returns no usage block, usage_metadata stays ``None``.

        Enforces the ``estimated`` vs ``actual`` distinction from the B4 design:
        a CostRecord lacking provider usage MUST be flagged as ``estimated``, not
        silently persisted as zeros (which would pass through as ``actual`` and
        under-report cost forever).
        """
        message = SimpleNamespace(
            role="assistant", content="no usage block", tool_calls=None
        )
        choice = SimpleNamespace(index=0, message=message, finish_reason="stop")
        response = SimpleNamespace(
            id="chatcmpl-no-usage",
            choices=[choice],
            model="test-model",
            usage=None,
        )

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        with patch.object(model, "_get_client", return_value=mock_client):
            result = await model._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.usage_metadata is None, (
            "When provider omits usage, usage_metadata must be None so "
            "CostCallbackHandler can correctly mark the row as estimated "
            "(not silently zero-billed)."
        )
