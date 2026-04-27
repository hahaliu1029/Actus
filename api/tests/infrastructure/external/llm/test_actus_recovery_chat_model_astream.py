"""PR-2 streaming tests T45-T48 (Fallback peek-first-chunk + gate streaming).

Uses Helper A (_bad_request) + Helper B (_StubBaseModel / _make_fallback /
_make_recovery_wrapper) + Helper C (_recovery_rules_snapshot) from the plan
header.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGenerationChunk

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


def _make_recovery_wrapper(inner, profile, **kw):
    from app.infrastructure.external.llm.actus_recovery_chat_model import (
        ActusRecoveryChatModel,
    )
    return ActusRecoveryChatModel.model_construct(
        inner=inner, profile=profile,
        api_mode=kw.get("api_mode", "chat_completions"),
        on_context_overflow=kw.get("on_context_overflow"),
        max_rewrite_attempts=kw.get("max_rewrite_attempts", 2),
    )


@pytest.fixture(autouse=True)
def _recovery_rules_snapshot():
    from app.domain.services.recovery._registry import RECOVERY_RULES
    saved = dict(RECOVERY_RULES)
    try:
        yield
    finally:
        RECOVERY_RULES.clear()
        RECOVERY_RULES.update(saved)


async def test_T45_fallback_astream_does_not_cross_protocol_after_first_chunk():
    """Audit Round 2 P1 #1 fix: primary MUST throw a _FALLBACK_TRIGGER_EXCEPTIONS
    member (BadRequestError / UnprocessableEntityError) — that's the ONLY
    exception class today's `_astream` wraps in its try/except. Using a plain
    RuntimeError would propagate even WITHOUT the peek-first-chunk fix, so the
    test wouldn't actually lock the spec §4.1 contract.
    """
    primary = _StubBaseModel("primary")

    async def primary_astream(*a, **kw):
        yield ChatGenerationChunk(message=AIMessageChunk(content="chunk1"))
        # After yielding 1st chunk, raise a trigger exception. Pre-fix code
        # would catch this in the outer try and escalate to fallback. Post-fix
        # (peek-first-chunk) the try/except only wraps the initial anext(),
        # so this exception now propagates.
        raise _bad_request("protocol error after first chunk")

    primary._astream = primary_astream

    fallback = _StubBaseModel("fallback")
    fallback._astream = AsyncMock()

    fb = _make_fallback(primary, fallback)
    chunks = []
    with pytest.raises(openai.BadRequestError):
        async for chunk in fb._astream([HumanMessage(content="x")]):
            chunks.append(chunk)
    assert len(chunks) == 1
    fallback._astream.assert_not_called()


async def test_T46_fallback_astream_first_chunk_error_still_escalates():
    """Audit Round 2 P1 #1 fix: use _bad_request(...) — openai.BadRequestError
    requires response + body.

    Audit Round 16 P1 #1 + Round 17/18 corrections:
    Assert the fallback chunk carries response_metadata stamped by
    `_stamp_fallback_escalation` (`actus_fallback_chat_model.py:74-111`).
    Without it, streaming-fallback cost attribution silently regresses.

    `_identifying_params` is a read-only @property on BaseChatModel —
    the only way to override is via subclassing.
    """

    class _FallbackStub(_StubBaseModel):
        @property
        def _identifying_params(self) -> dict:
            return {
                "model": "fallback-model-x",
                "provider_id": "fallback-provider-x",
            }

    primary = _StubBaseModel("primary")

    async def primary_astream(*a, **kw):
        raise _bad_request("protocol error before first chunk")
        yield  # make generator

    primary._astream = primary_astream

    fallback = _FallbackStub("fallback")

    async def fallback_astream(*a, **kw):
        yield ChatGenerationChunk(message=AIMessageChunk(content="from_fallback"))

    fallback._astream = fallback_astream

    # No profile → gate off → escalation path runs.
    fb = _make_fallback(primary, fallback)
    chunks = []
    async for c in fb._astream([HumanMessage(content="x")]):
        chunks.append(c)
    assert len(chunks) == 1
    assert chunks[0].message.content == "from_fallback"

    msg = chunks[0].message
    metadata = getattr(msg, "response_metadata", {}) or {}
    assert metadata.get("actus_fallback_attempt_ix") is not None, (
        "fallback chunk missing actus_fallback_attempt_ix — "
        "_stamp_fallback_escalation was likely dropped from _astream. "
        "See actus_fallback_chat_model.py:367-381 for the canonical "
        "pattern; the peek-first rewrite must preserve it."
    )
    assert metadata.get("actus_fallback_model") == "fallback-model-x", (
        f"fallback chunk metadata.actus_fallback_model regressed; "
        f"expected 'fallback-model-x' from stub _identifying_params, "
        f"got {metadata.get('actus_fallback_model')!r}."
    )
    assert metadata.get("actus_fallback_provider") == "fallback-provider-x", (
        f"fallback chunk metadata.actus_fallback_provider regressed; "
        f"expected 'fallback-provider-x' from stub _identifying_params, "
        f"got {metadata.get('actus_fallback_provider')!r}."
    )


async def test_T47_fallback_astream_gate_blocks_context_overflow_at_first_chunk():
    from app.domain.services.provider_profiles.generic_openai import GENERIC_OPENAI_PROFILE

    primary = _StubBaseModel("primary")

    async def primary_astream(*a, **kw):
        raise _bad_request("context_length_exceeded")
        yield

    primary._astream = primary_astream

    fallback = _StubBaseModel("fallback")
    fallback._astream = AsyncMock()

    fb = _make_fallback(primary, fallback, profile=GENERIC_OPENAI_PROFILE)
    with pytest.raises(openai.BadRequestError):
        async for _ in fb._astream([HumanMessage(content="x")]):
            pass
    fallback._astream.assert_not_called()


async def test_T48_fallback_astream_gate_blocks_compat_quirk_at_first_chunk():
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE

    primary = _StubBaseModel("primary")

    async def primary_astream(*a, **kw):
        raise _bad_request("Json mode response is not supported when enable_thinking is true")
        yield

    primary._astream = primary_astream

    fallback = _StubBaseModel("fallback")
    fallback._astream = AsyncMock()

    fb = _make_fallback(primary, fallback, profile=DASHSCOPE_QWEN_PROFILE)
    with pytest.raises(openai.BadRequestError):
        async for _ in fb._astream([HumanMessage(content="x")]):
            pass
    fallback._astream.assert_not_called()


async def test_T40_astream_400_before_first_chunk_retries_with_rewrite():
    """ST2: pre-first-chunk failure with a matching rule routes through the
    Recovery loop; post-rewrite the second peek succeeds and streams normally.
    R1 (DashScope JSON-mode + thinking) is the test vehicle — its rule is
    registered at recovery package import."""
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE
    import app.domain.services.recovery  # noqa: F401  (trigger register_all)

    inner = _StubBaseModel()
    call = {"n": 0}

    async def inner_astream(messages, **kw):
        call["n"] += 1
        if call["n"] == 1:
            raise _bad_request(
                "Json mode response is not supported when enable_thinking is true"
            )
        yield ChatGenerationChunk(message=AIMessageChunk(content="retry_ok"))

    inner._astream = inner_astream

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    chunks = []
    async for c in wrapper._astream([HumanMessage(content="x")]):
        chunks.append(c)
    assert len(chunks) == 1
    assert chunks[0].message.content == "retry_ok"
    assert call["n"] == 2


async def test_T41_astream_400_after_first_chunk_propagates():
    """I12: post-first-chunk errors raise; no retry, no escalation. The
    recovery loop's try/except scope is bounded to the initial peek, so once
    a chunk is yielded downstream any later exception bubbles up unchanged.
    """
    from app.domain.services.provider_profiles.dashscope_qwen import DASHSCOPE_QWEN_PROFILE

    inner = _StubBaseModel()

    async def inner_astream(*a, **kw):
        yield ChatGenerationChunk(message=AIMessageChunk(content="first"))
        raise RuntimeError("mid-stream")

    inner._astream = inner_astream

    wrapper = _make_recovery_wrapper(inner, DASHSCOPE_QWEN_PROFILE)
    chunks = []
    with pytest.raises(RuntimeError):
        async for c in wrapper._astream([HumanMessage(content="x")]):
            chunks.append(c)
    assert len(chunks) == 1
