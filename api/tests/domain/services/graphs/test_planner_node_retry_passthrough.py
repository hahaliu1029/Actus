"""planner_node must let ``ServerRequestsError`` propagate — regression.

Before this fix, ``planner_node`` wrapped ``structured_llm.ainvoke`` in a
bare ``except Exception`` and replaced any failure with a degraded
single-step fallback plan. That silently masked transient transport
errors from LangGraph's ``RetryPolicy`` (wired at build time with
``retry_on=ServerRequestsError``), so the retry never fired.

This test pins the new contract: ``ServerRequestsError`` from the LLM
call must surface out of ``planner_node`` so ``planner_retry`` — or, if
all attempts exhaust, the caller — sees the real error rather than a
papered-over fallback.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app.application.errors.exceptions import ServerRequestsError
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef


pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_react_graph_mock() -> MagicMock:
    class _ReactGraph:
        async def astream(self, *_a, **_kw):
            yield {"llm_node": {
                "events": [],
                "messages": [AIMessage(content='{"success": true, "result": "ok", "attachments": []}')],
            }}

        async def ainvoke(self, *_a, **_kw):
            return {
                "events": [],
                "messages": [AIMessage(content='{"success": true, "result": "ok", "attachments": []}')],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }
    return _ReactGraph()


def _make_summary_llm() -> MagicMock:
    llm = MagicMock()

    async def _astream(messages, **kwargs):
        yield AIMessageChunk(content='{"message": "done", "attachments": []}')

    llm.astream = _astream
    return llm


def _make_planner_llm_raising(exc: Exception) -> MagicMock:
    """planner_llm whose with_structured_output().ainvoke raises ``exc``."""
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(side_effect=exc)

    # Planner only ever invokes PlanResponse path before first executor run.
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(
        return_value=PlanUpdateResponse(steps=[StepDef(id="1", description="x")])
    )

    llm = MagicMock()

    def _with_structured_output(schema, **kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        raise ValueError(f"Unexpected schema: {schema}")

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)
    return llm


def _fast_retry_policy_patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink ``main_graph``'s ``RetryPolicy`` so tests don't spend real seconds
    on backoff. Keep ``retry_on=ServerRequestsError`` to exercise the real
    routing predicate.
    """
    import app.domain.services.graphs.main_graph as main_graph
    from langgraph.types import RetryPolicy

    original = main_graph.RetryPolicy

    def _fast(*args, **kwargs):
        # Force-override (not setdefault): production passes max_attempts=3,
        # and we want the test to exercise exactly one attempt so the raise
        # is observed without real backoff time.
        kwargs["max_attempts"] = 1
        kwargs["initial_interval"] = 0.0
        kwargs["backoff_factor"] = 1.0
        kwargs["max_interval"] = 0.0
        kwargs["jitter"] = False
        return original(*args, **kwargs)

    monkeypatch.setattr(main_graph, "RetryPolicy", _fast)


async def test_planner_node_propagates_server_requests_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.domain.services.graphs.main_graph import build_main_graph

    _fast_retry_policy_patch(monkeypatch)

    planner_llm = _make_planner_llm_raising(
        ServerRequestsError("LLM (m) returned empty response")
    )

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_react_graph_mock(),
        summary_llm=_make_summary_llm(),
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-x",
    )

    with pytest.raises(ServerRequestsError, match="empty response"):
        await graph.ainvoke({
            "message": "hello",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-x",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })


async def test_planner_node_still_falls_back_on_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-``ServerRequestsError`` failures still produce the degraded plan.

    Structured-output parse failures (pydantic ValidationError, LangChain
    output-parser errors) are terminal at the LLM layer — there's no
    retry path that would help, so ``planner_node`` keeps its existing
    ``except Exception`` safety net for them.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    _fast_retry_policy_patch(monkeypatch)

    planner_llm = _make_planner_llm_raising(ValueError("malformed JSON"))

    mock_uow = AsyncMock()
    mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
    mock_uow.__aexit__ = AsyncMock(return_value=False)

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_react_graph_mock(),
        summary_llm=_make_summary_llm(),
        uow_factory=MagicMock(return_value=mock_uow),
        session_id="sess-x",
    )

    # No raise — graph finishes using the fallback plan path.
    result = await graph.ainvoke({
        "message": "hello",
        "language": "en",
        "attachments": [],
        "image_content_blocks": [],
        "plan": None,
        "current_step": None,
        "messages": [],
        "execution_summary": "",
        "events": [],
        "flow_status": "idle",
        "session_id": "sess-x",
        "should_interrupt": False,
        "resume_value": None,
        "original_request": "",
        "skill_context": "",
        "conversation_summaries": [],
    })

    # Fallback plan title is "Task" and contains a single step mirroring
    # the user message — this confirms the parse-error branch still runs.
    plan = result.get("plan")
    assert plan is not None
    assert plan.title == "Task"
    assert len(plan.steps) == 1
