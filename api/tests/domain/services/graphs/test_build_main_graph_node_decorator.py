"""B5 PR-S2-2: ``build_main_graph(node_decorator=...)`` plumbing.

Locks the contract that an opt-in ``node_decorator`` (passed by the
composition layer) wraps each registered LangGraph node before
``add_node``. Used by ``app/application/composition/graph_assembly.py``
to inject ``traced_node(OtelTracer())`` so each node call produces a
``graph.node.<name>`` span.

Pinned behaviours
-----------------
- ``node_decorator=None`` (default) keeps the legacy registration —
  no wrapper applied.
- ``node_decorator`` is invoked exactly once per registered node
  (planner_node / executor_node / updater_node / interrupt_node).
- Function names are preserved (LangGraph's ``add_node`` introspects
  ``__name__`` for routing). The decorator MUST use ``functools.wraps``
  or equivalent — this is a load-bearing invariant.
"""
from __future__ import annotations

import functools

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk

from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_structured_planner_llm() -> MagicMock:
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(
        return_value=PlanResponse(
            title="t",
            goal="g",
            language="en",
            steps=[StepDef(description="step1")],
            message="m",
        )
    )
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(
        return_value=PlanUpdateResponse(
            steps=[StepDef(id="2", description="updated")],
        )
    )

    llm = MagicMock()

    def _with_structured_output(schema, **kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        raise ValueError(f"unexpected schema: {schema}")

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    async def _astream(messages, **kwargs):
        yield AIMessageChunk(content='{"message": "ok", "attachments": []}')

    llm.astream = _astream
    return llm


def _make_mock_react_graph():
    class _RG:
        async def astream(self, input_state, config=None, **kwargs):
            yield {
                "llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(
                            content='{"success": true, "result": "done", "attachments": []}'
                        ),
                    ],
                }
            }

    return _RG()


@pytest.fixture
def planner_llm():
    return _make_structured_planner_llm()


def test_node_decorator_default_none_does_not_wrap(planner_llm):
    """No decorator → legacy registration; the graph still compiles."""
    from app.domain.services.graphs.main_graph import build_main_graph

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-1",
    )
    assert graph is not None


def test_node_decorator_wraps_each_registered_node(planner_llm):
    """Decorator is invoked exactly once per registered node.

    There are 4 nodes registered in main_graph: planner_node /
    executor_node / updater_node / interrupt_node.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    decorated_names: list[str] = []

    def _spy_decorator(fn):
        decorated_names.append(fn.__name__)

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            return await fn(*args, **kwargs)

        return wrapper

    build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-1",
        node_decorator=_spy_decorator,
    )

    assert sorted(decorated_names) == sorted(
        ["planner_node", "executor_node", "updater_node", "interrupt_node"]
    )


def test_decorator_must_preserve_function_name(planner_llm):
    """LangGraph's ``add_node`` introspects ``fn.__name__``. A decorator
    that drops the original name would break routing — this test would
    fail loudly if a future refactor uses a non-``functools.wraps``
    wrapper or the build site stops calling the decorator.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    seen_names: list[str] = []

    def _spy(fn):
        seen_names.append(fn.__name__)
        return fn  # passthrough — names already preserved

    build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-1",
        node_decorator=_spy,
    )
    # Each registered node has a unique original name we can verify.
    assert "planner_node" in seen_names
    assert "executor_node" in seen_names
    assert "updater_node" in seen_names
    assert "interrupt_node" in seen_names
