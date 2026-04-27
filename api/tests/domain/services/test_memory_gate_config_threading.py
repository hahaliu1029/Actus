"""B4 M0 post-audit: MemoryGateClassifier threads config into structured ainvoke.

Graph-external LLM calls (memory gate, conversation summary, background
summary) don't inherit LangGraph's callback/metadata propagation — the
plumbing has to be explicit. This test locks the contract that passing
``config={...}`` to ``classify`` reaches the underlying structured
``ainvoke``.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable

from app.domain.services.memory_gate import (
    MemoryGateClassifier,
    MemoryGateInput,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _SpyStructured(Runnable):
    """Captures the ``config`` kwarg on ainvoke + returns a fixed BatchDecision."""

    captured_config: Any = None

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        _SpyStructured.captured_config = config

        from app.domain.services.memory_gate import (
            _MemoryGateBatchDecision,
            _MemoryGateDecisionWire,
        )

        return _MemoryGateBatchDecision(
            decisions=[
                _MemoryGateDecisionWire(
                    chunk_index=0,
                    verdict="keep",
                    category="fact",
                    confidence=0.8,
                )
            ]
        )


class _SpyLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "spy"

    async def _agenerate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable:
        return _SpyStructured()


async def test_classify_threads_config_to_structured_ainvoke() -> None:
    _SpyStructured.captured_config = None
    classifier = MemoryGateClassifier(_SpyLLM())
    cfg = {"callbacks": ["marker"], "metadata": {"langgraph_node": "memory_gate"}}
    await classifier.classify(
        [MemoryGateInput(chunk_index=0, text="hello")],
        config=cfg,
    )
    assert _SpyStructured.captured_config is cfg, (
        f"classify must forward its config to structured.ainvoke; "
        f"got {_SpyStructured.captured_config!r}"
    )


async def test_classify_without_config_passes_no_config() -> None:
    """Backward-compat: default call path stays unchanged."""
    _SpyStructured.captured_config = None
    classifier = MemoryGateClassifier(_SpyLLM())
    await classifier.classify(
        [MemoryGateInput(chunk_index=0, text="hello")]
    )
    assert _SpyStructured.captured_config is None
