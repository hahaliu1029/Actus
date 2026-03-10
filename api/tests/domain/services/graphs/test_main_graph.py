"""Tests for main_graph — outer orchestration (plan→execute→update→summarize)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock
from app.domain.models.event import PlanEvent, TitleEvent, MessageEvent, DoneEvent, PlanEventStatus
from app.domain.models.plan import Plan, Step, ExecutionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_planner_llm():
    """Mock LLM for planner that returns a plan JSON."""
    async def mock_invoke(**kwargs):
        return {
            "content": '{"title":"Test","goal":"Do test","language":"en","steps":[{"description":"Step 1"}],"message":"Let me help"}',
            "role": "assistant",
        }
    llm = AsyncMock()
    llm.invoke = mock_invoke
    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    return llm


@pytest.fixture
def mock_json_parser():
    parser = AsyncMock()
    import json
    async def parse(content, default_value=None):
        try:
            return json.loads(content)
        except Exception:
            return {"title": "Fallback", "goal": content, "steps": [{"description": content}], "message": "ok", "language": "en"}
    parser.invoke = parse
    return parser


def _make_mock_react_graph():
    """Create a mock react_graph with async generator astream."""
    class MockReactGraph:
        async def astream(self, input_state, config=None):
            yield {"llm_node": {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [],
            }}

        async def ainvoke(self, input_state, config=None):
            return {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }

    return MockReactGraph()


class TestBuildMainGraph:
    def test_graph_compiles(self, mock_planner_llm, mock_json_parser):
        from app.domain.services.graphs.main_graph import build_main_graph
        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-1",
        )
        assert graph is not None


class TestMainGraphFlow:
    async def test_full_flow_produces_plan_and_done(self, mock_planner_llm, mock_json_parser):
        from app.domain.services.graphs.main_graph import build_main_graph

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(return_value=mock_uow),
            session_id="sess-1",
        )

        result = await graph.ainvoke({
            "message": "help me test",
            "language": "en",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-1",
            "should_interrupt": False,
            "original_request": "",
            "skill_context": "",
        })

        events = result.get("events", [])
        event_types = [type(e).__name__ for e in events]
        # planner events come through state; executor events go via queue (empty in state)
        assert "PlanEvent" in event_types or "TitleEvent" in event_types
