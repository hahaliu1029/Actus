"""Tests for ActusFallbackChatModel — api_mode_fallback_enabled gate (A7 P1 fix).

These tests lock in the Task 1.8 P1 fix: ActusFallbackChatModel._agenerate must
consult ``profile.api_mode_fallback_enabled`` before performing the cross-
protocol escalation from Chat Completions to Responses. Providers without a
Responses API (e.g. Kimi) declare ``api_mode_fallback_enabled=False`` in their
profile, and for those the gate must re-raise the original primary exception
instead of invoking the fallback adapter.

Spec anchors:
- A7 spec section 4.2 ProviderProfile.api_mode_fallback_enabled
- Task 1.8 plan lines 2112-2119 (generic_openai api_mode_fallback_enabled=True)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.domain.services.provider_profiles._base import ProviderProfile
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_kimi_like_profile() -> ProviderProfile:
    """Profile without a Responses API — api_mode_fallback_enabled=False."""
    return ProviderProfile(
        provider_id="kimi_k2",
        human_name="Kimi K2",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=False,
    )


def _make_generic_profile() -> ProviderProfile:
    """Profile with Responses API — api_mode_fallback_enabled=True."""
    return ProviderProfile(
        provider_id="generic_openai",
        human_name="Generic OpenAI-compatible",
        default_api_mode="chat_completions",
        api_mode_fallback_enabled=True,
    )


def _bad_request_error() -> openai.BadRequestError:
    """Construct an openai.BadRequestError mirroring real SDK behavior."""
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.test/v1/chat/completions"),
        content=b'{"error": {"message": "test"}}',
    )
    return openai.BadRequestError(
        message="bad request",
        response=response,
        body={"error": {"message": "test"}},
    )


class _StubBaseModel:
    """Minimal stand-in mimicking BaseChatModel's ``_llm_type`` attr + _agenerate."""

    def __init__(self, llm_type: str = "stub") -> None:
        self._llm_type = llm_type
        self._agenerate = AsyncMock()


@pytest.fixture
def kimi_wrapper() -> ActusFallbackChatModel:
    """Wrapper with Kimi-like profile → fallback must NOT fire on 400."""
    # Pydantic validates primary/fallback as BaseChatModel; use model_construct
    # to bypass validation for lightweight stubs.
    return ActusFallbackChatModel.model_construct(
        primary=_StubBaseModel("primary-stub"),
        fallback=_StubBaseModel("fallback-stub"),
        provider_name="openai",
        profile=_make_kimi_like_profile(),
    )


@pytest.fixture
def generic_wrapper() -> ActusFallbackChatModel:
    """Wrapper with generic profile → default behavior (fallback runs on 400)."""
    return ActusFallbackChatModel.model_construct(
        primary=_StubBaseModel("primary-stub"),
        fallback=_StubBaseModel("fallback-stub"),
        provider_name="openai",
        profile=_make_generic_profile(),
    )


def _ok_chat_result(content: str = "ok") -> ChatResult:
    return ChatResult(
        generations=[ChatGeneration(message=AIMessage(content=content))]
    )


class TestApiModeFallbackGate:
    """A7 P1 fix: ActusFallbackChatModel must consult profile.api_mode_fallback_enabled."""

    async def test_disabled_profile_reraises_primary_bad_request(
        self, kimi_wrapper: ActusFallbackChatModel
    ) -> None:
        """400 from primary + api_mode_fallback_enabled=False → re-raise, NO fallback."""
        exc = _bad_request_error()
        kimi_wrapper.primary._agenerate = AsyncMock(side_effect=exc)
        kimi_wrapper.fallback._agenerate = AsyncMock(
            return_value=_ok_chat_result("fallback")
        )

        with pytest.raises(openai.BadRequestError) as excinfo:
            await kimi_wrapper._agenerate([HumanMessage(content="hi")])

        # The original primary exception propagates — no wrapping.
        assert excinfo.value is exc
        # Fallback must NOT have been invoked.
        kimi_wrapper.fallback._agenerate.assert_not_called()

    async def test_disabled_profile_reraises_primary_unprocessable(
        self, kimi_wrapper: ActusFallbackChatModel
    ) -> None:
        """422 from primary + api_mode_fallback_enabled=False → re-raise, NO fallback."""
        response = httpx.Response(
            status_code=422,
            request=httpx.Request("POST", "https://api.test/v1/chat/completions"),
            content=b'{}',
        )
        exc = openai.UnprocessableEntityError(
            message="unprocessable",
            response=response,
            body={},
        )
        kimi_wrapper.primary._agenerate = AsyncMock(side_effect=exc)
        kimi_wrapper.fallback._agenerate = AsyncMock(return_value=_ok_chat_result())

        with pytest.raises(openai.UnprocessableEntityError) as excinfo:
            await kimi_wrapper._agenerate([HumanMessage(content="hi")])

        assert excinfo.value is exc
        kimi_wrapper.fallback._agenerate.assert_not_called()

    async def test_enabled_profile_invokes_fallback_on_bad_request(
        self, generic_wrapper: ActusFallbackChatModel
    ) -> None:
        """400 from primary + api_mode_fallback_enabled=True → fallback runs."""
        exc = _bad_request_error()
        generic_wrapper.primary._agenerate = AsyncMock(side_effect=exc)
        generic_wrapper.fallback._agenerate = AsyncMock(
            return_value=_ok_chat_result("fallback-ok")
        )

        result = await generic_wrapper._agenerate([HumanMessage(content="hi")])

        assert isinstance(result, ChatResult)
        assert result.generations[0].message.content == "fallback-ok"
        generic_wrapper.fallback._agenerate.assert_awaited_once()

    async def test_disabled_profile_logs_skip(
        self,
        kimi_wrapper: ActusFallbackChatModel,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Guard must emit an INFO log naming provider_id + the disabled flag."""
        import logging

        exc = _bad_request_error()
        kimi_wrapper.primary._agenerate = AsyncMock(side_effect=exc)
        kimi_wrapper.fallback._agenerate = AsyncMock(return_value=_ok_chat_result())

        with caplog.at_level(
            logging.INFO,
            logger="app.infrastructure.external.llm.actus_fallback_chat_model",
        ):
            with pytest.raises(openai.BadRequestError):
                await kimi_wrapper._agenerate([HumanMessage(content="hi")])

        # The log line must reference the provider_id and the disabled flag.
        record_texts = [r.getMessage() for r in caplog.records]
        assert any("kimi_k2" in msg for msg in record_texts)
        assert any(
            "api_mode_fallback_enabled" in msg and "False" in msg
            for msg in record_texts
        )
