"""Tests for PlannerReActFlow — LangGraph-based implementation."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import DoneEvent, PlanEvent
from app.domain.models.message import Message
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm():
    """Mock LLM that returns plan JSON."""
    llm = AsyncMock()

    async def mock_invoke(**kwargs):
        return {
            "content": json.dumps({
                "title": "Test Plan",
                "goal": "Do test",
                "language": "en",
                "steps": [{"description": "Step 1"}],
                "message": "Let me help",
            }),
            "role": "assistant",
        }

    llm.invoke = mock_invoke
    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    type(llm).temperature = PropertyMock(return_value=0.7)
    type(llm).max_tokens = PropertyMock(return_value=4096)
    return llm


@pytest.fixture
def mock_json_parser():
    parser = AsyncMock()

    async def parse(content, default_value=None):
        try:
            return json.loads(content)
        except Exception:
            return default_value

    parser.invoke = parse
    return parser


@pytest.fixture
def mock_uow():
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=False)
    uow.session = AsyncMock()
    uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    return uow


def test_planner_react_flow_constructs_successfully(mock_llm, mock_json_parser, mock_uow):
    """Flow can be constructed with all required parameters."""
    flow = PlannerReActFlow(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        json_parser=mock_json_parser,
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )
    assert flow.done is True
    assert flow.plan is None


async def test_planner_react_flow_invoke_produces_events(mock_llm, mock_json_parser, mock_uow):
    """Flow.invoke() should yield events including DoneEvent."""
    flow = PlannerReActFlow(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        json_parser=mock_json_parser,
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )

    events = []
    async for event in flow.invoke(Message(message="help me test")):
        events.append(event)

    assert len(events) > 0
    assert any(isinstance(e, DoneEvent) for e in events)


async def test_planner_react_flow_produces_plan_event(mock_llm, mock_json_parser, mock_uow):
    """Flow should produce PlanEvent during execution."""
    flow = PlannerReActFlow(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        json_parser=mock_json_parser,
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )

    events = []
    async for event in flow.invoke(Message(message="help me test")):
        events.append(event)

    plan_events = [e for e in events if isinstance(e, PlanEvent)]
    assert len(plan_events) >= 1


def test_planner_react_flow_set_skill_context(mock_llm, mock_json_parser, mock_uow):
    """set_skill_context should store the context."""
    flow = PlannerReActFlow(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        json_parser=mock_json_parser,
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(),
        skill_tool=MagicMock(),
    )

    flow.set_skill_context("test context")
    assert flow._skill_context == "test context"
