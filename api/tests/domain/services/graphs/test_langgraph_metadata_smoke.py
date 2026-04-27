"""B4 M0 smoke gate: LangGraph injects langgraph_node + langgraph_step into callback metadata.

Design assumption (load-bearing): when a chat model is invoked inside a LangGraph node,
the framework injects ``langgraph_node`` and ``langgraph_step`` into the run metadata
so that downstream callbacks (e.g. ``CostCallbackHandler``) can attribute LLM spend to
a specific graph node.

Upstream evidence: ``langgraph/pregel/_algo.py`` lines ~632-633, 781-782, 930-931.

If this test regresses, the entire per-session cost ledger (B4) design collapses —
we would lose the only reliable source of ``node_name`` attribution.
"""

from __future__ import annotations

from typing import Any, List, Optional
from uuid import UUID

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _FakeChatModel(BaseChatModel):
    """Minimal BaseChatModel that returns a fixed AIMessage without hitting any network."""

    @property
    def _llm_type(self) -> str:
        return "fake-smoke"

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="ok"))]
        )

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("async only")


class _MetadataCapturingHandler(AsyncCallbackHandler):
    """Records metadata dicts passed to on_chat_model_start for later assertion."""

    def __init__(self) -> None:
        self.captured: list[dict[str, Any]] = []

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self.captured.append(dict(metadata or {}))


class _State(TypedDict):
    messages: list[BaseMessage]


async def test_langgraph_injects_node_and_step_into_callback_metadata() -> None:
    """Within a graph node, on_chat_model_start metadata must carry langgraph_node + langgraph_step."""
    fake_llm = _FakeChatModel()
    handler = _MetadataCapturingHandler()

    async def planner_node(state: _State, config: RunnableConfig) -> _State:
        response = await fake_llm.ainvoke(state["messages"], config=config)
        return {"messages": state["messages"] + [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("planner", planner_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", END)
    graph = builder.compile()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )

    assert handler.captured, "on_chat_model_start was never invoked"
    metadata = handler.captured[0]

    assert metadata.get("langgraph_node") == "planner", (
        f"expected langgraph_node='planner', got {metadata.get('langgraph_node')!r}. "
        "B4 CostCallbackHandler depends on this to set CostRecord.node_name."
    )
    assert "langgraph_step" in metadata, (
        f"langgraph_step missing from metadata; keys present: {sorted(metadata.keys())}. "
        "B4 CostCallbackHandler uses this to deduplicate on retry."
    )
    assert isinstance(metadata["langgraph_step"], int), (
        f"langgraph_step must be int, got {type(metadata['langgraph_step']).__name__}"
    )


async def test_langgraph_metadata_unique_per_node() -> None:
    """Two nodes in the same graph must be distinguishable via langgraph_node.

    This guards against a regression where Pregel stops refreshing metadata between
    nodes — B4 would then attribute every LLM call to the first node.
    """
    fake_llm = _FakeChatModel()
    handler = _MetadataCapturingHandler()

    async def first_node(state: _State, config: RunnableConfig) -> _State:
        response = await fake_llm.ainvoke(state["messages"], config=config)
        return {"messages": state["messages"] + [response]}

    async def second_node(state: _State, config: RunnableConfig) -> _State:
        response = await fake_llm.ainvoke(state["messages"], config=config)
        return {"messages": state["messages"] + [response]}

    builder: StateGraph = StateGraph(_State)
    builder.add_node("first_node", first_node)
    builder.add_node("second_node", second_node)
    builder.add_edge(START, "first_node")
    builder.add_edge("first_node", "second_node")
    builder.add_edge("second_node", END)
    graph = builder.compile()

    await graph.ainvoke(
        {"messages": [HumanMessage(content="hi")]},
        config={"callbacks": [handler]},
    )

    assert len(handler.captured) == 2, (
        f"expected 2 chat_model_start events (one per node), got {len(handler.captured)}"
    )
    nodes_seen = [m.get("langgraph_node") for m in handler.captured]
    assert nodes_seen == ["first_node", "second_node"], (
        f"node attribution leaked across nodes: {nodes_seen}"
    )
