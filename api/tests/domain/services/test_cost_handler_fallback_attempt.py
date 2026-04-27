"""B4 M0 post-audit: CostRecord.attempt_ix reflects fallback escalation.

When ``ActusFallbackChatModel``'s primary raises a _FALLBACK_TRIGGER_EXCEPTIONS
and fallback successfully runs, the resulting CostRecord must carry
``attempt_ix=1``. Without this, fallback-successful calls are silently
attributed to the primary and the ledger lies about which adapter billed.

The wiring: fallback wrapper calls ``_notify_fallback_escalation(run_manager)``
on escalation, which iterates ``run_manager.handlers`` and invokes
``mark_fallback_escalation(run_id)`` on any handler that exposes it —
duck-typed to avoid infrastructure→domain import.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List, Optional
from unittest.mock import MagicMock

import openai
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

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
    """Minimal BaseChatModel that returns preset usage / raises a preset error."""

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

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
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
        stop: Optional[List[str]] = None,
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


async def test_fallback_success_tags_cost_record_with_attempt_ix_1() -> None:
    captured: list[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="sess-fb", user_id="user-fb", persister=persist
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
    )
    fallback = _StubAdapter(
        return_content="from-fallback",
        return_usage={"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500},
        model_name_val="gpt-4o",  # known model so compute_cost returns actual
        provider_id_val="openai_official",
    )

    fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
    await fb.ainvoke(
        [HumanMessage(content="hi")],
        config={"callbacks": [handler]},
    )
    await handler.flush_pending()

    assert len(captured) == 1, f"expected 1 CostRecord, got {len(captured)}"
    rec = captured[0]
    assert rec.attempt_ix == 1, (
        f"Fallback-engaged run must be tagged attempt_ix=1; got {rec.attempt_ix}. "
        "If this is 0, _notify_fallback_escalation → mark_fallback_escalation "
        "plumbing regressed."
    )
    assert rec.cost_status == CostStatus.ACTUAL
    assert rec.input_tokens == 1000


async def test_primary_success_keeps_attempt_ix_0() -> None:
    """Control: when primary succeeds, no escalation notification fires."""
    captured: list[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="sess-ok", user_id="u", persister=persist
    )

    primary = _StubAdapter(
        return_content="fine",
        return_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        model_name_val="gpt-4o",
        provider_id_val="openai_official",
    )
    fallback = _StubAdapter(
        return_content="should-not-run",
        return_usage={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
    )

    fb = ActusFallbackChatModel(primary=primary, fallback=fallback)
    await fb.ainvoke(
        [HumanMessage(content="hi")],
        config={"callbacks": [handler]},
    )
    await handler.flush_pending()

    assert len(captured) == 1
    assert captured[0].attempt_ix == 0
