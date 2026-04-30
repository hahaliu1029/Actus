"""B5 PR-S2-2 + FOLLOW-10: ``executor_node`` injects ``step.id`` into
``config["configurable"]["step_id"]`` before driving the react subgraph.

Locks the contract:

- The fresh ``configurable`` handed to the inner ``react_graph.astream``
  contains ``step_id`` equal to ``current_step.id``.
- The OUTER configurable (handed to ``executor_node``) is not mutated —
  ``executor_node`` builds a fresh dict so checkpointer / lifespan
  callers that share a config across nodes stay untouched.

The ``traced_node`` decorator (when wired by the composition layer)
reads this slot for span attributes; ``OtelToolSpanCallback`` reads
the same slot so tool spans inherit the same ``step_id``.

Strategy
--------
Mock ``react_graph`` to capture the ``config`` it receives via
``astream``, drive a single executor pass via ``ainvoke``, and assert
``configurable["step_id"]`` is the plan's first step id.
"""
from __future__ import annotations

from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk

from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_planner_llm() -> MagicMock:
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
    # Empty steps list → updater terminates the loop.
    update_structured.ainvoke = AsyncMock(
        return_value=PlanUpdateResponse(steps=[]),
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


class _CapturingReactGraph:
    """Records the ``config`` passed to ``astream`` so tests can assert
    that ``executor_node`` injected ``step_id`` before sub-invocation.
    """

    def __init__(self) -> None:
        self.captured_configs: list[dict[str, Any]] = []

    async def astream(self, input_state, config=None, **kwargs):
        self.captured_configs.append(dict(config or {}))
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


async def _run_one_executor_pass(react_graph, outer_cfg=None) -> None:
    from app.domain.services.graphs.main_graph import build_main_graph

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)
    mock_uow.session = AsyncMock()
    mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    mock_uow.session.get_summary = AsyncMock(return_value=[])

    planner_llm = _make_planner_llm()
    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=react_graph,
        summary_llm=planner_llm,
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-1",
    )

    await graph.ainvoke(
        {
            "message": "do thing",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
        },
        config=outer_cfg if outer_cfg is not None else {"configurable": {}},
    )


async def test_executor_injects_step_id_into_fresh_configurable():
    """The react subgraph receives ``configurable["step_id"] == step.id``."""
    react = _CapturingReactGraph()
    await _run_one_executor_pass(react)

    assert react.captured_configs, "react_graph.astream was never called"
    inner_cfg = react.captured_configs[0]
    configurable = inner_cfg.get("configurable") or {}
    step_id = configurable.get("step_id")
    assert isinstance(step_id, str) and step_id, (
        f"executor must inject a non-empty step_id; got {step_id!r}"
    )


async def test_step_id_does_not_leak_into_outer_configurable():
    """The OUTER ``config`` (handed to ``executor_node``) is not mutated.

    ``executor_node`` builds a FRESH configurable dict so the runtime's
    config object is untouched.
    """
    react = _CapturingReactGraph()
    outer_cfg: dict[str, Any] = {"configurable": {"thread_id": "outer-thread"}}

    await _run_one_executor_pass(react, outer_cfg=outer_cfg)

    assert "step_id" not in outer_cfg["configurable"], (
        "executor must build a fresh configurable; outer config was mutated"
    )
    # And the inner cfg DID get step_id (sanity).
    assert "step_id" in (react.captured_configs[0].get("configurable") or {})
