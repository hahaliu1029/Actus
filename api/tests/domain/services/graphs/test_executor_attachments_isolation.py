"""Negative regression: executor_node must NOT inherit attachments from
historical AI envelopes already in ``state["messages"]``.

Before the fix, ``executor_node`` extracted attachments by scanning
``react_final["messages"]`` in reverse. That list is
``dedup_messages(initial_messages + all_react_messages)`` — it contains
the pre-existing session history too. If the current step produced no
new ``AIMessage(content)`` (e.g. the react loop only emitted tool calls
or interrupted early), the reversed scan would land on the prior step's
AI envelope and copy its ``attachments`` onto the current step. The
``prior_step_outputs`` hint injected into later steps' EXECUTION_PROMPT
would then spread the stale path forward.

The contract is: only messages added during this react run
(``all_react_messages``) count toward the current step's summary and
attachments.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)

from app.domain.models.event import MessageEvent
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _planner_llm_one_step() -> MagicMock:
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=PlanResponse(
        title="Test", goal="Do thing", language="en",
        steps=[StepDef(description="Check something")],
        message="ok",
    ))
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(return_value=PlanUpdateResponse(
        steps=[StepDef(id="1", description="Check something")],
    ))

    llm = MagicMock()

    def _with_structured_output(schema, **_kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        raise ValueError(f"Unexpected schema: {schema}")

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
    return llm


def _summary_llm() -> MagicMock:
    llm = MagicMock()

    async def _astream(messages, **_kwargs):
        yield AIMessageChunk(content='{"message": "done", "attachments": []}')

    llm.astream = _astream
    return llm


def _react_graph_no_ai_content() -> object:
    """Mock react_graph whose step produces NO new ``AIMessage(content)``.

    Only yields a ToolMessage — a realistic shape when the last LLM turn
    issued a tool call that completed without a follow-up assistant text,
    or the step was interrupted between ``tool_node`` and ``llm_node``.
    """
    class _Graph:
        async def astream(self, _input_state, config=None, **_kwargs):
            yield {"tool_node": {
                "events": [],
                "messages": [
                    ToolMessage(
                        content="ok",
                        tool_call_id="t-1",
                        name="noop",
                    ),
                ],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }}

        async def ainvoke(self, _input_state, config=None):
            return {
                "events": [],
                "messages": [],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }

    return _Graph()


async def test_executor_does_not_inherit_historical_attachments() -> None:
    """With a stale AIMessage envelope already in ``state["messages"]`` and
    a react step that yields no new AI content, the completed step's
    attachments must remain empty — not copy the stale path.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=_planner_llm_one_step(),
        react_graph=_react_graph_no_ai_content(),
        summary_llm=_summary_llm(),
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-iso",
    )

    # Pre-seed ``messages`` with a historical AI envelope carrying a
    # stale attachment. If the executor scan walks this list instead of
    # just the new messages from this react run, it will copy
    # "/stale/prior.txt" onto the current step.
    stale_envelope = (
        '{"success": true, "result": "earlier run done", '
        '"attachments": ["/stale/prior.txt"]}'
    )

    result = await graph.ainvoke({
        "message": "check",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "plan": None,
        "current_step": None,
        "messages": [
            HumanMessage(content="earlier turn"),
            AIMessage(content=stale_envelope),
        ],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-iso",
        "should_interrupt": False,
        "resume_value": None,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": [],
    })

    plan = result.get("plan")
    assert plan is not None, "Plan should exist after executor runs"
    assert len(plan.steps) >= 1

    # The critical assertion: no stale leak. Previous implementation
    # would write ["/stale/prior.txt"] here because the reversed scan
    # on react_final["messages"] found the historical envelope.
    for step in plan.steps:
        assert "/stale/prior.txt" not in (step.attachments or []), (
            f"Stale prior-run attachment leaked into step {step!r}. "
            f"Executor_node must scan only the messages added during "
            f"this react run (all_react_messages), not the merged list."
        )
