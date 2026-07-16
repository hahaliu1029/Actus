"""Integration test: full flow from message -> events via LangGraph."""

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from app.application.services.sandbox_accessors import (
    EagerBrowserAccessor,
    EagerSandboxAccessor,
)
from app.domain.models.app_config import AgentConfig
from app.domain.models.event import PlanEvent, MessageEvent, TitleEvent
from app.domain.models.llm_responses import PlanResponse, StepDef
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm():
    """Mock BaseChatModel that returns plan JSON via with_structured_output and
    step results via ainvoke for react graph calls."""
    llm = MagicMock(spec=BaseChatModel)

    # with_structured_output returns a runnable whose ainvoke returns a parsed model
    mock_structured = AsyncMock()
    mock_structured.ainvoke = AsyncMock(return_value=PlanResponse(
        title="Integration Test Plan",
        goal="Test the full pipeline",
        language="en",
        steps=[StepDef(description="Execute test step")],
        message="I'll help you test.",
    ))
    llm.with_structured_output = MagicMock(return_value=mock_structured)

    # For react graph: ainvoke returns AIMessage (no tool calls = final answer)
    llm.ainvoke = AsyncMock(return_value=AIMessage(content="Step completed successfully"))

    # bind_tools returns a new mock that also has ainvoke
    bound_mock = MagicMock(spec=BaseChatModel)
    bound_mock.ainvoke = AsyncMock(return_value=AIMessage(content="Step completed successfully"))
    llm.bind_tools = MagicMock(return_value=bound_mock)

    type(llm).model_name = PropertyMock(return_value="gpt-4o")
    return llm


@pytest.fixture
def mock_uow():
    uow = AsyncMock()
    uow.__aenter__ = AsyncMock(return_value=uow)
    uow.__aexit__ = AsyncMock(return_value=False)
    uow.session = AsyncMock()
    uow.session.get_skill_graph_state = AsyncMock(return_value=None)
    uow.session.get_memory = AsyncMock(return_value=Memory())
    uow.session.get_summary = AsyncMock(return_value=[])
    uow.session.save_memory = AsyncMock()
    uow.session.save_summary = AsyncMock()
    return uow


class TestFullFlowIntegration:
    async def test_flow_produces_plan_and_done_events(
        self, mock_llm, mock_uow,
    ):
        """PlannerReActFlow.invoke() should yield Title + Plan + Message events and defer final state.

        Note: DoneEvent is emitted by agent_task_runner._do_postprocess, not by the flow itself.
        After a normal completion, the flow stores the final graph state in _deferred_final_state
        for the runner's postprocess path.
        """
        flow = PlannerReActFlow(
            _allow_default_prompt_assembler=True,
            uow_factory=MagicMock(return_value=mock_uow),
            llm=mock_llm,
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="integration-test",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            checkpointer=MemorySaver(),
        )

        events = []
        async for event in flow.invoke(Message(message="help me test the pipeline")):
            events.append(event)

        # Should have at least some events
        assert len(events) > 0

        # Final state must be deferred to the runner's postprocess path
        assert flow._deferred_final_state is not None

        # Should have a PlanEvent (plan was created)
        plan_events = [e for e in events if isinstance(e, PlanEvent)]
        assert len(plan_events) >= 1

        # Should have a TitleEvent
        title_events = [e for e in events if isinstance(e, TitleEvent)]
        assert len(title_events) >= 1

        # Should have a MessageEvent
        msg_events = [e for e in events if isinstance(e, MessageEvent)]
        assert len(msg_events) >= 1

    async def test_flow_updates_plan_after_completion(
        self, mock_llm, mock_uow,
    ):
        """After invoke() + _persist_after_graph, flow.plan should be set and flow.done should be True.

        The runner calls _persist_after_graph in _do_postprocess; this test mirrors that contract.
        """
        flow = PlannerReActFlow(
            _allow_default_prompt_assembler=True,
            uow_factory=MagicMock(return_value=mock_uow),
            llm=mock_llm,
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="integration-test-2",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            checkpointer=MemorySaver(),
        )

        assert flow.done is True
        assert flow.plan is None

        events = []
        async for event in flow.invoke(Message(message="do something")):
            events.append(event)

        # Runner's postprocess path assigns the plan via _persist_after_graph
        await flow._persist_after_graph(
            flow._deferred_final_state, flow._deferred_summaries,
        )

        assert flow.plan is not None
        assert flow.plan.title == "Integration Test Plan"
        assert flow.done is True

    async def test_skill_context_passed_through(
        self, mock_llm, mock_uow,
    ):
        """Skill context should be accessible in the flow and the flow should complete normally."""
        flow = PlannerReActFlow(
            _allow_default_prompt_assembler=True,
            uow_factory=MagicMock(return_value=mock_uow),
            llm=mock_llm,
            agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
            session_id="integration-test-3",
            browser_accessor=EagerBrowserAccessor(AsyncMock()),
            sandbox_accessor=EagerSandboxAccessor(AsyncMock()),
            search_engine=AsyncMock(),
            mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
            a2a_tool=MagicMock(manager=None),
            skill_tool=MagicMock(),
            checkpointer=MemorySaver(),
        )

        flow._skill_context_provider = lambda: "Use the calculator skill"
        assert flow._get_skill_context_seed() == "Use the calculator skill"

        events = []
        async for event in flow.invoke(Message(message="calculate 2+2")):
            events.append(event)

        # Flow should complete normally: final state deferred, no interrupt
        assert flow._deferred_final_state is not None
        assert not flow._deferred_final_state.get("should_interrupt")
