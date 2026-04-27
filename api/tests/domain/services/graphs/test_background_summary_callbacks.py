"""B4 M0 post-audit: run_background_summary threads callbacks into the LLM astream.

The audit flagged that ``background_summary`` is graph-external — LangGraph's
metadata/callback propagation does NOT apply — so the cost handler has to be
passed in explicitly. This test locks the contract.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, List

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult

from app.domain.services.graphs.background_summary import run_background_summary

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _CaptureConfigLLM(BaseChatModel):
    """astream records the ``config`` kwarg it was called with for assertion."""

    captured_config: dict | None = None

    @property
    def _llm_type(self) -> str:
        return "capture-config"

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Any = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content='{"message": "ok", "attachments": []}'
            )
        )

    async def astream(
        self,
        input: Any,
        config: Any = None,
        **kwargs: Any,
    ):
        _CaptureConfigLLM.captured_config = config
        async for c in super().astream(input, config=config, **kwargs):
            yield c


async def test_callbacks_and_metadata_are_threaded() -> None:
    _CaptureConfigLLM.captured_config = None
    llm = _CaptureConfigLLM()

    class _Marker(AsyncCallbackHandler):
        pass

    marker_handler = _Marker()

    events: list = []

    async def _on_event(evt: Any) -> None:
        events.append(evt)

    summary = await run_background_summary(
        messages=[HumanMessage(content="hi")],
        summary_llm=llm,
        on_event=_on_event,
        lang="zh",
        callbacks=[marker_handler],
    )

    cfg = _CaptureConfigLLM.captured_config
    assert cfg is not None, "astream must receive a config dict when callbacks are provided"
    assert cfg.get("callbacks") == [marker_handler], (
        f"astream.config.callbacks must be the caller's list; got {cfg.get('callbacks')!r}"
    )
    metadata = cfg.get("metadata") or {}
    assert metadata.get("langgraph_node") == "background_summary", (
        "metadata.langgraph_node must be 'background_summary' so the aggregate's "
        "by_node breakdown attributes out-of-graph summary cost correctly."
    )
    assert summary == "ok"


async def test_no_callbacks_passes_no_config() -> None:
    """Backward compat: without callbacks, don't inject a config (keeps legacy callers)."""
    _CaptureConfigLLM.captured_config = None
    llm = _CaptureConfigLLM()

    async def _on_event(evt: Any) -> None:
        pass

    await run_background_summary(
        messages=[HumanMessage(content="hi")],
        summary_llm=llm,
        on_event=_on_event,
        lang="zh",
    )
    assert _CaptureConfigLLM.captured_config is None
