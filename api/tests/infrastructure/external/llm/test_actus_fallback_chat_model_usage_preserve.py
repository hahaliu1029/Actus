"""B4 M0 Phase B3: ActusFallbackChatModel preserves usage_metadata across delegation.

The fallback model is transparent today — ``_agenerate`` returns whatever the
delegate ``_agenerate`` returned; ``_astream`` yields whatever the delegate
yielded. This test locks that contract: a future wrapping refactor must not
silently strip ``usage_metadata`` from the AIMessage(Chunk), because that would
invisibly break B4 cost tracking for every fallback-engaged call.

Covers both routes:
    - primary succeeds → usage_metadata from primary survives
    - primary raises a fallback-trigger exception → fallback path wins and
      its usage_metadata survives (with different counts, verifying we didn't
      accidentally cache primary data)
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List
from unittest.mock import MagicMock

import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    UsageMetadata,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_usage(input_tokens: int, output_tokens: int) -> UsageMetadata:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class _StubAdapter(BaseChatModel):
    """Minimal BaseChatModel whose _agenerate/_astream return preset usage_metadata."""

    return_usage: UsageMetadata | None = None
    return_content: str = "ok"
    raise_on_call: BaseException | None = None

    @property
    def _llm_type(self) -> str:
        return "stub-adapter"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        msg = AIMessage(content=self.return_content, usage_metadata=self.return_usage)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        chunk = AIMessageChunk(
            content=self.return_content, usage_metadata=self.return_usage
        )
        yield ChatGenerationChunk(message=chunk)

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError


class TestFallbackUsagePreservation:
    async def test_primary_success_preserves_usage_metadata(self) -> None:
        primary = _StubAdapter(return_usage=_make_usage(42, 8), return_content="primary-ok")
        fallback = _StubAdapter(return_usage=_make_usage(999, 999), return_content="should-not-run")

        fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
        result = await fb._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.content == "primary-ok"
        assert msg.usage_metadata == _make_usage(42, 8), (
            "Fallback model must pass through the primary's usage_metadata unchanged."
        )

    async def test_fallback_path_preserves_its_usage_metadata(self) -> None:
        bad_req = openai.BadRequestError(
            message="payload mismatch",
            response=MagicMock(status_code=400, request=MagicMock()),
            body=None,
        )
        primary = _StubAdapter(
            return_usage=_make_usage(42, 8),
            return_content="primary",
            raise_on_call=bad_req,
        )
        fallback = _StubAdapter(
            return_usage=_make_usage(100, 25),
            return_content="fallback-reply",
        )

        fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
        result = await fb._agenerate([HumanMessage(content="hi")])

        msg = result.generations[0].message
        assert msg.content == "fallback-reply"
        assert msg.usage_metadata == _make_usage(100, 25), (
            "Fallback path must pass through the fallback adapter's usage_metadata, "
            "not leak primary's counts or strip the field."
        )

    async def test_astream_primary_success_preserves_usage_metadata(self) -> None:
        primary = _StubAdapter(return_usage=_make_usage(10, 3), return_content="stream-ok")
        fallback = _StubAdapter(return_usage=_make_usage(999, 999))

        fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
        chunks = [gen.message async for gen in fb._astream([HumanMessage(content="hi")])]
        assert chunks, "astream yielded nothing"
        assert chunks[0].usage_metadata == _make_usage(10, 3)
