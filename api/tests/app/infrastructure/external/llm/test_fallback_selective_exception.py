"""Selective-exception fallback tests for ``ActusFallbackChatModel``.

The fallback wrapper is designed for **cross-protocol escalation** — chat
completions → responses — not for same-protocol retry. So it should only
fall through on errors that actually mean "this protocol/payload is not
supported by primary":

- ``openai.BadRequestError`` (400)
- ``openai.UnprocessableEntityError`` (422)

``openai.NotFoundError`` is deliberately excluded: a generic 404 also
fires on wrong model name or wrong base_url path, which are permanent
config errors that should not be silently papered over by escalation.

Everything else (transient timeouts, empty responses, 5xx, rate limits,
auth failures, or arbitrary ``Exception`` subclasses) must propagate so
LangGraph node ``RetryPolicy`` or upper layers can decide what to do.
Funneling them through fallback hides the real retry path and, on
providers that don't implement the Responses API (e.g. Zhipu
``glm-*`` on ``/api/paas/v4``), guarantees a 404 on the second hop.
"""
from __future__ import annotations

from typing import Any, Type
from unittest.mock import AsyncMock, patch

import openai
import pytest
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages import AIMessage, AIMessageChunk

from app.application.errors.exceptions import ServerRequestsError
from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)
from app.infrastructure.external.llm.actus_responses_model import ActusResponsesModel

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_openai_error(cls: Type[Exception], message: str = "x") -> Exception:
    """Create an openai SDK error instance without needing a real httpx.Response.

    openai's APIStatusError.__init__ requires ``response`` + ``body`` kwargs.
    For fallback routing we only care about the class identity (``isinstance``
    matching), so bypass ``__init__`` via ``__new__`` and set args manually.
    """
    err = cls.__new__(cls)
    Exception.__init__(err, message)
    return err


def _build_fallback_pair() -> tuple[ActusChatModel, ActusResponsesModel, ActusFallbackChatModel]:
    primary = ActusChatModel(
        base_url="https://x.test/v1",
        api_key="k",
        model_name="primary-model",
        timeout_seconds=0,  # disable wait_for wrap; we'll mock exceptions directly
    )
    fallback = ActusResponsesModel(
        base_url="https://x.test/v1",
        api_key="k",
        model_name="fallback-model",
        timeout_seconds=0,
    )
    wrapper = ActusFallbackChatModel(primary=primary, fallback=fallback)
    return primary, fallback, wrapper


def _ok_chat_result(text: str = "from-fallback") -> ChatResult:
    msg = AIMessage(content=text)
    return ChatResult(generations=[ChatGeneration(message=msg)])


async def _ok_chunk_stream(text: str = "from-fallback"):
    yield ChatGenerationChunk(message=AIMessageChunk(content=text))


# ---------------------------------------------------------------------------
# _agenerate: protocol-incompatibility errors → fallback
# ---------------------------------------------------------------------------


class TestAgenerateTriggersFallback:
    """These exception types mean "primary does not accept this protocol/payload".

    The whole point of ``ActusFallbackChatModel`` is to escalate to the
    Responses API when ``chat.completions`` rejects the request. Keep pinning
    the specific types so future refactors don't silently broaden the catch.
    """

    @pytest.mark.parametrize(
        "exc_cls",
        [
            openai.BadRequestError,
            openai.UnprocessableEntityError,
        ],
    )
    async def test_protocol_error_triggers_fallback(
        self, exc_cls: Type[Exception],
    ) -> None:
        primary, fallback, wrapper = _build_fallback_pair()

        primary_mock = AsyncMock(side_effect=_make_openai_error(exc_cls, "proto"))
        fallback_mock = AsyncMock(return_value=_ok_chat_result())

        with patch.object(primary, "_agenerate", primary_mock):
            with patch.object(fallback, "_agenerate", fallback_mock):
                result = await wrapper._agenerate([HumanMessage(content="hi")])

        assert result.generations[0].message.content == "from-fallback"
        assert primary_mock.await_count == 1
        assert fallback_mock.await_count == 1


# ---------------------------------------------------------------------------
# _agenerate: transient/permanent errors → propagate (no fallback)
# ---------------------------------------------------------------------------


class TestAgenerateDoesNotTriggerFallback:
    """These must propagate so LangGraph RetryPolicy / upper layers decide.

    Critically, ``ServerRequestsError`` (empty response, timeout wrap, etc.)
    must NOT trigger fallback — that was the bug that caused 404 cascades on
    providers without a Responses API endpoint.
    """

    async def test_server_requests_error_propagates(self) -> None:
        """Empty-response / unexpected-response from ChatModel must not fall through."""
        primary, fallback, wrapper = _build_fallback_pair()

        primary_mock = AsyncMock(
            side_effect=ServerRequestsError("LLM (primary-model) returned empty response")
        )
        fallback_mock = AsyncMock(return_value=_ok_chat_result())

        with patch.object(primary, "_agenerate", primary_mock):
            with patch.object(fallback, "_agenerate", fallback_mock):
                with pytest.raises(ServerRequestsError, match="empty response"):
                    await wrapper._agenerate([HumanMessage(content="hi")])

        assert fallback_mock.await_count == 0, (
            "ServerRequestsError is the D5.1 timeout / empty-response wrapper — "
            "it must propagate so LangGraph RetryPolicy can retry same endpoint, "
            "not trigger cross-protocol fallback."
        )

    @pytest.mark.parametrize(
        "exc_cls",
        [
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,  # 5xx
            openai.RateLimitError,       # 429
            openai.AuthenticationError,  # 401
            openai.PermissionDeniedError,  # 403
        ],
    )
    async def test_other_openai_errors_propagate(
        self, exc_cls: Type[Exception],
    ) -> None:
        primary, fallback, wrapper = _build_fallback_pair()

        # APITimeoutError / APIConnectionError take a single ``request`` arg
        # rather than response/body; go through __new__ for consistency.
        exc = _make_openai_error(exc_cls, "transient-or-permanent")

        primary_mock = AsyncMock(side_effect=exc)
        fallback_mock = AsyncMock(return_value=_ok_chat_result())

        with patch.object(primary, "_agenerate", primary_mock):
            with patch.object(fallback, "_agenerate", fallback_mock):
                with pytest.raises(exc_cls):
                    await wrapper._agenerate([HumanMessage(content="hi")])

        assert fallback_mock.await_count == 0

    async def test_generic_exception_propagates(self) -> None:
        """Unknown exception types default to propagate — never silently fall back."""
        primary, fallback, wrapper = _build_fallback_pair()

        primary_mock = AsyncMock(side_effect=RuntimeError("unexpected"))
        fallback_mock = AsyncMock(return_value=_ok_chat_result())

        with patch.object(primary, "_agenerate", primary_mock):
            with patch.object(fallback, "_agenerate", fallback_mock):
                with pytest.raises(RuntimeError, match="unexpected"):
                    await wrapper._agenerate([HumanMessage(content="hi")])

        assert fallback_mock.await_count == 0


# ---------------------------------------------------------------------------
# _astream: same semantics as _agenerate
# ---------------------------------------------------------------------------


class TestAstreamSelectiveFallback:
    async def test_protocol_error_triggers_fallback_in_stream(self) -> None:
        primary, fallback, wrapper = _build_fallback_pair()

        async def primary_stream(*_a: Any, **_kw: Any):
            raise _make_openai_error(openai.BadRequestError, "proto")
            yield  # make this an async generator

        with patch.object(primary, "_astream", primary_stream):
            with patch.object(fallback, "_astream", lambda *a, **kw: _ok_chunk_stream()):
                chunks = [
                    c async for c in wrapper._astream([HumanMessage(content="hi")])
                ]

        assert len(chunks) == 1
        assert chunks[0].message.content == "from-fallback"

    async def test_server_requests_error_propagates_in_stream(self) -> None:
        primary, fallback, wrapper = _build_fallback_pair()
        fallback_called = False

        async def primary_stream(*_a: Any, **_kw: Any):
            raise ServerRequestsError("stream empty")
            yield

        async def fallback_stream(*_a: Any, **_kw: Any):
            nonlocal fallback_called
            fallback_called = True
            yield ChatGenerationChunk(message=AIMessageChunk(content="x"))

        with patch.object(primary, "_astream", primary_stream):
            with patch.object(fallback, "_astream", fallback_stream):
                with pytest.raises(ServerRequestsError, match="stream empty"):
                    async for _ in wrapper._astream([HumanMessage(content="hi")]):
                        pass

        assert fallback_called is False
