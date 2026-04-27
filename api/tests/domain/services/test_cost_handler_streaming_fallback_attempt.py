"""B4 M0 post-audit: streaming fallback also records attempt_ix=1.

LangChain's ``BaseChatModel.astream`` doesn't forward ``run_manager`` into
subclass ``_astream``, so the old ``_notify_fallback_escalation`` hook can't
reach the cost handler on the streaming path. Fix: fallback wrapper stamps
the first fallback chunk's ``response_metadata`` with
``actus_fallback_attempt_ix`` / ``actus_fallback_model`` /
``actus_fallback_provider``; handler reads those at ``on_llm_end`` and
applies them to the pending entry.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Optional
from unittest.mock import MagicMock

import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from app.domain.models.cost_record import CostRecord, CostStatus
from app.domain.services.cost_callback_handler import CostCallbackHandler
from app.infrastructure.external.llm.actus_fallback_chat_model import (
    ActusFallbackChatModel,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _StubAdapter(BaseChatModel):
    return_usage: dict | None = None
    return_content: str = "ok"
    raise_on_call: BaseException | None = None
    model_name_val: str = "stub-model"
    provider_id_val: str = "stub_provider"

    @property
    def _llm_type(self) -> str:
        return "stub-adapter"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model_name_val, "provider_id": self.provider_id_val}

    async def _agenerate(self, *a: Any, **kw: Any) -> ChatResult:
        raise NotImplementedError

    def _generate(self, *a: Any, **kw: Any) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if self.raise_on_call is not None:
            raise self.raise_on_call
        # Split into two chunks so we can prove stamping only touches the
        # first and still makes it through merge to the final message.
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=self.return_content[:1])
        )
        final_usage = self.return_usage
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content=self.return_content[1:],
                usage_metadata=final_usage,
            )
        )


async def test_streaming_fallback_tags_cost_record_with_attempt_ix_1() -> None:
    captured: list[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="sess-fb-stream", user_id="user-fb", persister=persist
    )

    bad_req = openai.BadRequestError(
        message="primary payload mismatch",
        response=MagicMock(status_code=400, request=MagicMock()),
        body=None,
    )
    primary = _StubAdapter(
        return_content="primary",
        raise_on_call=bad_req,
        model_name_val="primary-model",
        provider_id_val="primary_provider",
    )
    fallback = _StubAdapter(
        return_content="hi",
        return_usage={
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
        },
        model_name_val="gpt-4o",
        provider_id_val="openai_official",
    )

    fb = ActusFallbackChatModel(primary=primary, fallback=fallback)

    async for _ in fb.astream(
        [HumanMessage(content="hi")],
        config={"callbacks": [handler]},
    ):
        pass
    await handler.flush_pending()

    assert len(captured) == 1, f"expected 1 CostRecord, got {len(captured)}"
    rec = captured[0]
    assert rec.attempt_ix == 1, (
        f"Streaming fallback must tag attempt_ix=1; got {rec.attempt_ix}. "
        "If this is 0, _stamp_fallback_escalation → "
        "_apply_stream_fallback_tags plumbing regressed."
    )
    assert rec.model == "gpt-4o", (
        "Streaming fallback must swap model to the fallback adapter's "
        f"model; got {rec.model!r}"
    )
    assert rec.provider == "openai_official", (
        "Streaming fallback must swap provider to the fallback adapter's "
        f"provider_id; got {rec.provider!r}"
    )
    assert rec.cost_status == CostStatus.ACTUAL


async def test_streaming_primary_success_keeps_attempt_ix_0() -> None:
    """Control: no escalation → no actus_fallback_* tags → attempt_ix stays 0."""
    captured: list[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="sess-ok-stream", user_id="u", persister=persist
    )

    primary = _StubAdapter(
        return_content="hi",
        return_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        model_name_val="gpt-4o",
        provider_id_val="openai_official",
    )
    fallback = _StubAdapter(
        return_content="nope",
        model_name_val="other",
        provider_id_val="other_provider",
    )

    fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
    async for _ in fb.astream(
        [HumanMessage(content="hi")],
        config={"callbacks": [handler]},
    ):
        pass
    await handler.flush_pending()

    assert len(captured) == 1
    assert captured[0].attempt_ix == 0
