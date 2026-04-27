"""PR-2 T38/T39: Fallback T12 gate extended to CONTEXT_OVERFLOW.

Uses Helper A (_bad_request) + Helper B (_StubBaseModel / _make_fallback)
from the plan's "Common Test Helpers" section.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)


pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


def _bad_request(body_str: str, *, status: int = 400) -> openai.BadRequestError:
    req = httpx.Request("POST", "https://api.test/v1/chat/completions")
    resp = httpx.Response(status_code=status, request=req, text=body_str)
    return openai.BadRequestError(message=body_str, response=resp, body={"message": body_str})


class _StubBaseModel(BaseChatModel):
    """Real BaseChatModel subclass so wrap_with_recovery / bind_tools
    internal ctor validation accepts it. See plan header Helper B rationale
    (Audit Round 3 P1 #3)."""

    def __init__(self, llm_type: str = "stub", **data):
        super().__init__(**data)
        object.__setattr__(self, "_llm_type_override", llm_type)
        self._agenerate = AsyncMock()
        self._astream = None

    @property
    def _llm_type(self) -> str:
        return getattr(self, "_llm_type_override", "stub")

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return "stub-model"

    def _generate(self, *a, **kw):
        raise NotImplementedError("async-only stub")

    def bind_tools(self, tools, **kw):
        return self


def _make_fallback(primary, fallback, *, profile=None):
    return ActusFallbackChatModel.model_construct(
        primary=primary, fallback=fallback, provider_name="openai", profile=profile,
    )


async def test_T38_fallback_gate_blocks_context_overflow_escalation():
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE

    primary = _StubBaseModel("primary")
    primary._agenerate = AsyncMock(side_effect=_bad_request("context_length_exceeded"))

    fallback = _StubBaseModel("fallback")
    fallback._agenerate = AsyncMock(
        return_value=ChatResult(generations=[ChatGeneration(message=HumanMessage(content="x"))]),
    )

    fb = _make_fallback(primary, fallback, profile=GENERIC_OPENAI_PROFILE)
    with pytest.raises(openai.BadRequestError):
        await fb._agenerate([HumanMessage(content="big")])
    fallback._agenerate.assert_not_called()


async def test_T39_fallback_gate_still_blocks_compat_quirk_escalation():
    """Regression: COMPAT_QUIRK branch remains (v4 N4 defense-in-depth)."""
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE

    primary = _StubBaseModel("primary")
    primary._agenerate = AsyncMock(side_effect=_bad_request(
        "Json mode response is not supported when enable_thinking is true",
    ))

    fallback = _StubBaseModel("fallback")
    fallback._agenerate = AsyncMock()

    fb = _make_fallback(primary, fallback, profile=DASHSCOPE_QWEN_PROFILE)
    with pytest.raises(openai.BadRequestError):
        await fb._agenerate([HumanMessage(content="x")])
    fallback._agenerate.assert_not_called()
