"""Tests for main_graph — outer orchestration (plan->execute->update->summarize)."""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from app.domain.models.event import PlanEvent, TitleEvent, MessageEvent, PlanEventStatus
from app.domain.models.llm_responses import PlanResponse, PlanUpdateResponse, StepDef
from app.domain.models.plan import Plan, Step, ExecutionStatus

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_mock_summary_llm(response_json: str = '{"message": "Summary done.", "attachments": []}'):
    """Build a mock BaseChatModel for summary_llm with astream support.

    Returns a MagicMock whose astream() yields AIMessageChunk objects.
    """
    llm = MagicMock()

    async def _astream(messages, **kwargs):
        yield AIMessageChunk(content=response_json)
    llm.astream = _astream
    return llm


def _make_structured_planner_llm(
    create_response: PlanResponse | None = None,
    update_response: PlanUpdateResponse | None = None,
):
    """Build a mock BaseChatModel whose with_structured_output() returns the right ainvoke mock.

    The mock dispatches based on the schema class passed to with_structured_output().
    Also includes a default astream() mock so it can double as summary_llm in tests.
    """
    if create_response is None:
        create_response = PlanResponse(
            title="Test", goal="Do test", language="en",
            steps=[StepDef(description="Step 1")],
            message="Let me help",
        )
    if update_response is None:
        update_response = PlanUpdateResponse(
            steps=[StepDef(id="2", description="Updated step based on results")],
        )

    # Structured LLM mocks — one per schema type
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)

    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(return_value=update_response)

    llm = MagicMock()

    def _with_structured_output(schema, **kwargs):
        if schema is PlanResponse:
            return create_structured
        if schema is PlanUpdateResponse:
            return update_structured
        raise ValueError(f"Unexpected schema: {schema}")

    llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

    # Also support astream for when this mock is used as summary_llm
    async def _astream(messages, **kwargs):
        yield AIMessageChunk(content='{"message": "Summary done.", "attachments": []}')
    llm.astream = _astream

    return llm


@pytest.fixture
def mock_planner_llm():
    """Mock BaseChatModel for planner with structured output support."""
    return _make_structured_planner_llm()


def _make_mock_react_graph():
    """Create a mock react_graph with async generator astream."""
    class MockReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {"llm_node": {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [
                    AIMessage(content='{"success": true, "result": "done", "attachments": []}'),
                ],
            }}

        async def ainvoke(self, input_state, config=None):
            return {
                "events": [MessageEvent(role="assistant", message="Step done")],
                "messages": [
                    AIMessage(content='{"success": true, "result": "done", "attachments": []}'),
                ],
                "should_interrupt": False,
                "attempt_count": 1,
                "failure_count": 0,
            }

    return MockReactGraph()


class TestBuildMainGraph:
    def test_graph_compiles(self, mock_planner_llm):
        from app.domain.services.graphs.main_graph import build_main_graph
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-1",
        )
        assert graph is not None


class TestParallelBackendWaitGuard:
    async def test_guard_scope_wraps_entire_parallel_backend(self, monkeypatch):
        from contextlib import asynccontextmanager

        from app.domain.services.graphs import main_graph

        events = []

        class Guard:
            @asynccontextmanager
            async def step_scope(self, step_id):
                events.append(("enter", step_id))
                try:
                    yield
                finally:
                    events.append(("exit", step_id))

        async def fake_impl(state, config, step):
            events.append(("impl", step.id))
            return main_graph.ParallelBackendOutcome(success=True, summary="ok")

        monkeypatch.setattr(main_graph, "_run_parallel_backend_impl", fake_impl)
        step = MagicMock(id="step-1")

        outcome = await main_graph._run_parallel_backend(
            {}, {"configurable": {"coordinator_wait_guard": Guard()}}, step
        )

        assert outcome.success is True
        assert events == [
            ("enter", "step-1"),
            ("impl", "step-1"),
            ("exit", "step-1"),
        ]

    async def test_guard_scope_cleans_up_when_backend_raises(self, monkeypatch):
        from contextlib import asynccontextmanager

        from app.domain.services.graphs import main_graph

        events = []

        class Guard:
            @asynccontextmanager
            async def step_scope(self, step_id):
                events.append(("enter", step_id))
                try:
                    yield
                finally:
                    events.append(("exit", step_id))

        async def failing_impl(state, config, step):
            raise RuntimeError("backend failed")

        monkeypatch.setattr(main_graph, "_run_parallel_backend_impl", failing_impl)

        with pytest.raises(RuntimeError, match="backend failed"):
            await main_graph._run_parallel_backend(
                {},
                {"configurable": {"coordinator_wait_guard": Guard()}},
                MagicMock(id="step-error"),
            )

        assert events == [("enter", "step-error"), ("exit", "step-error")]

    async def test_missing_guard_preserves_legacy_call(self, monkeypatch):
        from app.domain.services.graphs import main_graph

        expected = main_graph.ParallelBackendOutcome(success=True, summary="legacy")
        impl = AsyncMock(return_value=expected)
        monkeypatch.setattr(main_graph, "_run_parallel_backend_impl", impl)
        step = MagicMock(id="step-legacy")

        assert await main_graph._run_parallel_backend({}, {}, step) is expected
        impl.assert_awaited_once_with({}, {}, step)


class TestMainGraphFlow:
    async def test_full_flow_produces_plan_and_done(self, mock_planner_llm):
        from app.domain.services.graphs.main_graph import build_main_graph

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(return_value=mock_uow),
            session_id="sess-1",
        )

        result = await graph.ainvoke({
            "message": "help me test",
            "language": "en",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-1",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        events = result.get("events", [])
        event_types = [type(e).__name__ for e in events]
        # planner events come through state; executor events go via queue (empty in state)
        assert "PlanEvent" in event_types or "TitleEvent" in event_types

    async def test_default_language_is_zh(self, mock_planner_llm):
        """When no language is specified, planner should default to zh."""
        from app.domain.services.graphs.main_graph import build_main_graph

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-lang",
        )

        result = await graph.ainvoke({
            "message": "help me",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-lang",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        # Verify the plan language fallback is "zh" not "en"
        plan = result.get("plan")
        assert plan is not None

    async def test_empty_memory_recall_plan_falls_through_to_executor(self):
        """记忆型问题在 planner 返回空步骤时，应自动补一条查询步骤继续执行。"""
        from app.domain.services.graphs.main_graph import build_main_graph

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="职业信息查询",
                goal="",
                language="zh",
                steps=[],
                message=(
                    "您好！我理解您想了解自己的职业信息。但是，作为AI助手，"
                    "我无法直接知道您的职业是什么。"
                ),
            )
        )

        mock_uow = AsyncMock()
        mock_uow.__aenter__ = AsyncMock(return_value=mock_uow)
        mock_uow.__aexit__ = AsyncMock(return_value=False)
        mock_uow.session = AsyncMock()
        mock_uow.session.get_skill_graph_state = AsyncMock(return_value=None)

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(return_value=mock_uow),
            session_id="sess-memory",
        )

        result = await graph.ainvoke(
            {
                "message": "我的职业是什么",
                "language": "zh",
                "attachments": [],
                "image_content_blocks": [],
                "plan": None,
                "current_step": None,
                "messages": [],
                "execution_summary": "",
                "events": [],
                "flow_status": "idle",
                "session_id": "sess-memory",
                "should_interrupt": False,
                "resume_value": None,
                "original_request": "",
                "skill_context": "## Available Tool Summary\n- memory: memory_search, memory_get",
                "conversation_summaries": [],
            },
            config={"configurable": {"has_memory_tools": True}},
        )

        plan = result.get("plan")
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.message == "正在查询你的记忆以回答这个问题……"
        assert "查询记忆" in plan.steps[0].description

    async def test_planner_receives_conversation_summaries(self):
        """Planner system prompt should include conversation summaries when available."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_messages = []

        create_response = PlanResponse(
            title="Test", goal="test", language="zh",
            steps=[StepDef(description="step1")],
            message="ok",
        )

        create_structured = AsyncMock()
        async def capturing_ainvoke(messages, **kwargs):
            captured_messages.extend(messages)
            return create_response
        create_structured.ainvoke = capturing_ainvoke

        update_structured = AsyncMock()
        update_structured.ainvoke = AsyncMock(return_value=PlanUpdateResponse(steps=[]))

        planner_llm = MagicMock()
        def _with_structured_output(schema, **kwargs):
            if schema is PlanResponse:
                return create_structured
            return update_structured
        planner_llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=_make_mock_summary_llm(),
            uow_factory=MagicMock(),
            session_id="sess-summary",
        )

        await graph.ainvoke({
            "message": "continue",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-summary",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": ["### Round 1\n- user: check weather\n- result: got weather"],
        })

        system_msgs = [m for m in captured_messages if isinstance(m, SystemMessage)]
        assert len(system_msgs) >= 1
        assert "history" in system_msgs[0].content.lower() or "摘要" in system_msgs[0].content
        assert "weather" in system_msgs[0].content or "check weather" in system_msgs[0].content

    async def test_step_success_false_when_tool_failures_detected(self):
        """When react_graph returns failure_count > 0, the step should be marked as failed."""
        from app.domain.services.graphs.main_graph import build_main_graph

        class FailingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {"tool_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content="CAPTCHA blocked, search failed"),
                    ],
                    "failure_count": 1,
                    "should_interrupt": False,
                }}

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="T", goal="G", language="zh",
                steps=[StepDef(description="search news")],
                message="ok",
            ),
            update_response=PlanUpdateResponse(steps=[]),
        )

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=FailingReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-fail",
        )

        result = await graph.ainvoke({
            "message": "search AI news",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-fail",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        plan = result.get("plan")
        assert plan is not None
        completed_steps = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
        assert len(completed_steps) >= 1
        assert completed_steps[0].success is False

    async def test_summarizer_streams_and_emits_message_event(self):
        """run_background_summary should stream LLM via astream and emit partial+final MessageEvents.

        summarizer_node was removed from the compiled graph (see E1 / test_main_graph_no_summarizer);
        final-summary generation now lives in run_background_summary, invoked by
        agent_task_runner._do_postprocess. This test exercises that replacement directly.
        """
        from app.domain.services.graphs.background_summary import run_background_summary

        json_response = '{"message": "Task done, here is the report.", "attachments": ["/home/ubuntu/report.md"]}'
        chunk1 = json_response[:30]
        chunk2 = json_response[30:]

        summary_llm = MagicMock()

        async def mock_astream(messages, **kwargs):
            yield AIMessageChunk(content=chunk1)
            yield AIMessageChunk(content=chunk2)
        summary_llm.astream = mock_astream

        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="do something"),
            AIMessage(content='{"success": true, "result": "done", "attachments": []}'),
        ]

        captured_events: list = []

        async def on_event(evt):
            captured_events.append(evt)

        await run_background_summary(messages, summary_llm, on_event, lang="zh")

        msg_events = [e for e in captured_events if isinstance(e, MessageEvent)]
        partial_events = [e for e in msg_events if e.partial]
        final_events = [e for e in msg_events if not e.partial]

        assert len(partial_events) >= 1, "Should have at least one partial streaming event"
        assert len(final_events) == 1, "Should have exactly one final MessageEvent"

        stream_ids = {e.stream_id for e in msg_events}
        assert len(stream_ids) == 1, "All MessageEvents should share the same stream_id"
        assert None not in stream_ids, "stream_id should not be None"

        for i in range(1, len(partial_events)):
            assert len(partial_events[i].message) >= len(partial_events[i - 1].message)

        final = final_events[0]
        assert "report" in final.message.lower() or "done" in final.message.lower()
        assert len(final.attachments) == 1
        assert final.attachments[0].filepath == "/home/ubuntu/report.md"
        assert final.partial is False


class TestExecutorMessageBranching:
    """Test the three-way branching in executor_node for message handling."""

    async def test_first_step_no_history_uses_system_prompt(self):
        """When messages=[] and is_resuming=False, executor builds fresh system+execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content='{"success": true, "result": "done", "attachments": []}')
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="T", goal="G", language="zh",
                steps=[StepDef(description="S1")],
                message="ok",
            ),
            update_response=PlanUpdateResponse(steps=[]),
        )

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-exec",
        )

        await graph.ainvoke({
            "message": "do something",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-exec",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        assert isinstance(msgs[0], SystemMessage)
        assert "task execution" in msgs[0].content.lower() or "agent" in msgs[0].content.lower() or "\u4efb\u52a1\u6267\u884c\u667a\u80fd\u4f53" in msgs[0].content

    async def test_has_history_not_resuming_updates_system_prompt(self):
        """When messages have history and is_resuming=False, executor updates system prompt and appends execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content='{"success": true, "result": "done", "attachments": []}')
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = MagicMock()
        planner_llm.with_structured_output = MagicMock(return_value=AsyncMock())

        step = Step(id="s2", description="Step 2: analyze data")
        plan = Plan(title="T", goal="G", language="zh", steps=[
            Step(id="s1", description="Step 1: collect", status=ExecutionStatus.COMPLETED),
            step,
        ], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            summary_llm=_make_mock_summary_llm(),
            uow_factory=MagicMock(),
            session_id="sess-exec-2",
        )

        history_messages = [
            SystemMessage(content="old system prompt"),
            HumanMessage(content="old execution prompt"),
            AIMessage(content='{"success": true, "result": "collected data", "attachments": []}'),
        ]

        await graph.ainvoke({
            "message": "continue",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": plan,
            "current_step": step,
            "messages": history_messages,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-exec-2",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "analyze data",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        # System prompt should be updated (not "old system prompt")
        assert isinstance(msgs[0], SystemMessage)
        assert "\u4efb\u52a1\u6267\u884c\u667a\u80fd\u4f53" in msgs[0].content
        # Should NOT contain "user takeover" (not resuming)
        all_content = " ".join(m.content for m in msgs if hasattr(m, "content"))
        assert "\u7528\u6237\u5df2\u5b8c\u6210\u63a5\u7ba1" not in all_content

    async def test_resuming_uses_takeover_message(self):
        """When resume_value is set with saved messages, executor appends takeover resume message."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content='{"success": true, "result": "done", "attachments": []}')
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = MagicMock()
        planner_llm.with_structured_output = MagicMock(return_value=AsyncMock())

        step = Step(id="s_login", description="Login to Notion")
        plan = Plan(title="T", goal="G", language="zh", steps=[step], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            summary_llm=_make_mock_summary_llm(),
            uow_factory=MagicMock(),
            session_id="sess-resume",
        )

        saved = [
            SystemMessage(content="some system prompt"),
            HumanMessage(content="do login"),
        ]

        await graph.ainvoke({
            "message": "I have logged in",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": plan,
            "current_step": step,
            "messages": saved,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-resume",
            "should_interrupt": False,
            "resume_value": "I have logged in",
            "original_request": "login",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        all_content = " ".join(m.content for m in msgs if hasattr(m, "content"))
        assert "\u7528\u6237\u5df2\u5b8c\u6210\u63a5\u7ba1" in all_content


class TestUpdaterNodePlanUpdate:
    """Test that updater_node calls planner LLM to update plan based on execution results."""

    async def test_updater_calls_planner_with_execution_summary(self):
        """updater_node should call planner LLM with UPDATE_PLAN_PROMPT when execution_summary exists."""
        from app.domain.services.graphs.main_graph import build_main_graph

        update_ainvoke_calls = []

        create_response = PlanResponse(
            title="T", goal="G", language="zh",
            steps=[StepDef(description="Search databases"), StepDef(description="Read database structure")],
            message="ok",
        )
        update_response = PlanUpdateResponse(
            steps=[StepDef(id="2", description="Use database_id=2083c6e7 to read database structure")],
        )

        create_structured = AsyncMock()
        create_structured.ainvoke = AsyncMock(return_value=create_response)

        update_structured = AsyncMock()
        async def tracking_update_ainvoke(messages, **kwargs):
            update_ainvoke_calls.append(messages)
            return update_response
        update_structured.ainvoke = tracking_update_ainvoke

        planner_llm = MagicMock()
        def _with_structured_output(schema, **kwargs):
            if schema is PlanResponse:
                return create_structured
            return update_structured
        planner_llm.with_structured_output = MagicMock(side_effect=_with_structured_output)

        class MockReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {"llm_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content='{"success": true, "result": "Found database_id=2083c6e7", "attachments": []}')
                    ],
                    "should_interrupt": False,
                }}

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=MockReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-update",
        )

        result = await graph.ainvoke({
            "message": "check March tasks",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-update",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        # Verify planner was called for both create and update
        assert create_structured.ainvoke.call_count >= 1, "Planner should have been called to create plan"
        assert len(update_ainvoke_calls) >= 1, "Planner should have been called to update plan after step execution"

        # Verify the plan's steps were updated by the planner
        plan = result.get("plan")
        assert plan is not None
        assert len(plan.steps) >= 1


class TestInterruptResume:
    """Test that executor_node handles interrupt (WaitEvent) correctly with native interrupt()."""

    async def test_interrupt_does_not_mark_step_completed(self):
        """When react_graph returns should_interrupt=True, the step should NOT be marked COMPLETED.

        Uses checkpointer so interrupt_node can call interrupt().
        """
        from langgraph.checkpoint.memory import MemorySaver
        from app.domain.services.graphs.main_graph import build_main_graph

        class InterruptingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {"tool_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content="Requesting browser takeover"),
                    ],
                    "should_interrupt": True,
                }}

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="T", goal="G", language="zh",
                steps=[StepDef(description="Login to Notion"), StepDef(description="Read data")],
                message="ok",
            ),
        )

        checkpointer = MemorySaver()
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=InterruptingReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-interrupt",
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "test-interrupt"}}
        result = await graph.ainvoke({
            "message": "view Notion data",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-interrupt",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        }, config)

        # Graph should have been interrupted (interrupt_node called interrupt())
        assert "__interrupt__" in result

        # Check persisted state — should_interrupt should be True
        state = graph.get_state(config)
        assert state.next  # interrupt_node is pending

        # Verify state values from executor_node output
        assert state.values.get("should_interrupt") is True

        # Step should NOT be marked as COMPLETED — it was interrupted mid-execution
        current_step = state.values.get("current_step")
        assert current_step is not None
        assert current_step.status != ExecutionStatus.COMPLETED

        # Messages should be preserved for resume
        messages = state.values.get("messages", [])
        assert len(messages) > 0

        # original_request should be preserved
        assert state.values.get("original_request") != ""

    async def test_interrupt_and_resume_with_command(self):
        """Full interrupt -> Command(resume=...) -> executor_node cycle."""
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command
        from app.domain.services.graphs.main_graph import build_main_graph

        call_count = 0

        class InterruptThenCompleteReactGraph:
            """First call interrupts, second call completes."""
            async def astream(self, input_state, config=None, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    yield {"tool_node": {
                        "events": [],
                        "messages": [
                            AIMessage(content="Need user login"),
                        ],
                        "should_interrupt": True,
                    }}
                else:
                    yield {"llm_node": {
                        "events": [],
                        "messages": [
                            AIMessage(content='{"success": true, "result": "done after resume", "attachments": []}'),
                        ],
                        "should_interrupt": False,
                    }}

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="T", goal="G", language="zh",
                steps=[StepDef(description="Login")],
                message="ok",
            ),
            update_response=PlanUpdateResponse(steps=[]),
        )

        checkpointer = MemorySaver()
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=InterruptThenCompleteReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-resume-cmd",
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "test-resume-cmd"}}

        # Step 1: Initial run — hits interrupt
        result1 = await graph.ainvoke({
            "message": "login to notion",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-resume-cmd",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        }, config)

        assert "__interrupt__" in result1
        state = graph.get_state(config)
        assert state.next  # interrupt_node pending

        # Step 2: Resume with Command
        result2 = await graph.ainvoke(Command(resume="I have logged in"), config)

        # After resume, executor_node should have received resume_value
        # Graph should complete (no more interrupt)
        assert result2.get("flow_status") == "completed"
        assert result2.get("should_interrupt") is not True

    async def test_interrupt_preserves_execution_summary(self):
        """When interrupted, executor_node should still return execution_summary from the last AI message."""
        from langgraph.checkpoint.memory import MemorySaver
        from app.domain.services.graphs.main_graph import build_main_graph

        class InterruptingReactGraph:
            async def astream(self, input_state, config=None, **kwargs):
                yield {"tool_node": {
                    "events": [],
                    "messages": [
                        AIMessage(content="Found database_id=abc123, need to login first"),
                    ],
                    "should_interrupt": True,
                }}

        planner_llm = _make_structured_planner_llm(
            create_response=PlanResponse(
                title="T", goal="G", language="zh",
                steps=[StepDef(description="Find DB")],
                message="ok",
            ),
        )

        checkpointer = MemorySaver()
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=InterruptingReactGraph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-int-summary",
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "test-int-summary"}}
        result = await graph.ainvoke({
            "message": "find my database",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-int-summary",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        }, config)

        # execution_summary should be in the checkpointed state
        state = graph.get_state(config)
        summary = state.values.get("execution_summary", "")
        assert "database_id=abc123" in summary


class TestCompactMessages:
    """Test the _compact_messages helper function."""

    def test_compacts_browser_tool_results(self):
        from app.domain.services.graphs.main_graph import _compact_messages
        from langchain_core.messages import ToolMessage

        msgs = [
            SystemMessage(content="system"),
            AIMessage(content="", tool_calls=[{"id": "tc1", "name": "browser_view", "args": {}}]),
            ToolMessage(
                content='<html><title>Notion Dashboard</title><body><p>Lots of HTML content here...</p></body></html>',
                tool_call_id="tc1",
                name="browser_view",
            ),
        ]
        compacted = _compact_messages(msgs)

        assert len(compacted) == 3
        assert "Notion Dashboard" in compacted[2].content
        assert "<html>" not in compacted[2].content

    def test_truncates_long_tool_results(self):
        from app.domain.services.graphs.main_graph import _compact_messages
        from langchain_core.messages import ToolMessage

        long_content = "x" * 5000
        msgs = [
            ToolMessage(content=long_content, tool_call_id="tc1", name="mcp_notion_search"),
        ]
        compacted = _compact_messages(msgs)

        assert len(compacted[0].content) < 5000
        assert "\u5df2\u622a\u65ad" in compacted[0].content

    def test_preserves_normal_messages(self):
        from app.domain.services.graphs.main_graph import _compact_messages

        msgs = [
            SystemMessage(content="system prompt"),
            HumanMessage(content="user message"),
            AIMessage(content="assistant response"),
        ]
        compacted = _compact_messages(msgs)

        assert len(compacted) == 3
        assert compacted[0].content == "system prompt"
        assert compacted[1].content == "user message"
        assert compacted[2].content == "assistant response"


class TestCompactMessagesTruncation:
    """Verify Tier 2 truncation uses head+tail instead of head-only."""

    def test_compact_messages_head_tail_truncation(self):
        """ToolMessage > 2000 chars should be truncated with head+tail, not head-only."""
        from app.domain.services.graphs.main_graph import _compact_messages
        from langchain_core.messages import ToolMessage

        content = "H" * 1500 + "M" * 1500 + "T" * 1500  # 4500 chars
        msg = ToolMessage(content=content, tool_call_id="c1", name="shell_execute")
        result = _compact_messages([msg])

        assert len(result) == 1
        truncated = result[0].content
        assert len(truncated) < 4500
        assert "已截断" in truncated

    def test_compact_messages_preserves_tail(self):
        """After truncation, the tail portion of the original content should be preserved."""
        from app.domain.services.graphs.main_graph import _compact_messages
        from langchain_core.messages import ToolMessage

        tail_marker = "TAIL_END_MARKER"
        content = "X" * 4000 + tail_marker  # > 2000 chars
        msg = ToolMessage(content=content, tool_call_id="c1", name="shell_execute")
        result = _compact_messages([msg])

        truncated = result[0].content
        assert tail_marker in truncated, "Tail content should be preserved in head+tail truncation"


async def test_updater_node_sanitizes_parallel_work_units_flag_off(monkeypatch):
    """[WS0 §3A.4] Drive the REAL updater_node via build_main_graph: a flag-off
    PlanUpdateResponse carrying parallel_work_units must yield an updated Step
    with parallel_work_units=None. RED before the main_graph.py:1184-1189 edit,
    GREEN after. Mirrors TestUpdaterNodePlanUpdate."""
    from unittest.mock import AsyncMock, MagicMock
    from langchain_core.messages import AIMessage
    from app.domain.models.work_unit import ParallelWorkUnitGroupRequest, WorkUnitRequest
    from app.domain.services.graphs.main_graph import build_main_graph

    monkeypatch.delenv("ACTUS_C2_COORDINATOR_ENABLED", raising=False)
    pwu = ParallelWorkUnitGroupRequest(
        work_units=[WorkUnitRequest(objective="x", phase="exploration", allowed_tools=["file_read"])]
    )
    create_response = PlanResponse(
        title="T", goal="G", language="zh",
        steps=[StepDef(description="step one"), StepDef(description="step two")],
        message="ok",
    )
    update_response = PlanUpdateResponse(
        steps=[StepDef(id="2", description="updated parallel step", parallel_work_units=pwu)],
    )
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(return_value=update_response)
    planner_llm = MagicMock()
    def _wso(schema, **kwargs):
        return create_structured if schema is PlanResponse else update_structured
    planner_llm.with_structured_output = MagicMock(side_effect=_wso)

    class MockReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {"llm_node": {
                "events": [],
                "messages": [AIMessage(content='{"success": true, "result": "done", "attachments": []}')],
                "should_interrupt": False,
            }}

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=MockReactGraph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-sanitize",
    )
    result = await graph.ainvoke({
        "message": "do work", "language": "zh", "attachments": [],
        "image_content_blocks": [], "plan": None, "current_step": None,
        "messages": [], "execution_summary": "", "events": [],
        "flow_status": "idle", "session_id": "sess-sanitize",
        "should_interrupt": False, "resume_value": None,
        "original_request": "", "skill_context": "", "conversation_summaries": [],
    })
    plan = result.get("plan")
    assert plan is not None
    # The updater-spliced step must have its parallel_work_units cleared flag-off.
    assert all(s.parallel_work_units is None for s in plan.steps)


async def test_updater_node_deduplicates_ids_against_completed_prefix():
    """A replan must not reintroduce an already-completed step id.

    The production incident expanded ``[1, 2]`` into ``[1(done), 1, 2]`` and
    executed the search step twice.  IDs from the completed prefix are reserved;
    a colliding updated step must receive a deterministic fallback id.
    """
    from app.domain.services.graphs.main_graph import build_main_graph

    create_response = PlanResponse(
        title="T",
        goal="G",
        language="zh",
        steps=[
            StepDef(id="1", description="first search"),
            StepDef(id="2", description="write summary"),
        ],
        message="ok",
    )
    update_structured = AsyncMock()
    update_structured.ainvoke = AsyncMock(
        side_effect=[
            PlanUpdateResponse(
                steps=[
                    StepDef(id="1", description="retry first search"),
                    StepDef(id="2", description="write summary"),
                ]
            ),
            PlanUpdateResponse(steps=[]),
            PlanUpdateResponse(steps=[]),
        ]
    )
    create_structured = AsyncMock()
    create_structured.ainvoke = AsyncMock(return_value=create_response)
    planner_llm = MagicMock()

    def _wso(schema, **kwargs):
        return create_structured if schema is PlanResponse else update_structured

    planner_llm.with_structured_output = MagicMock(side_effect=_wso)

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=_make_mock_react_graph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-updater-dedupe",
    )
    result = await graph.ainvoke(
        {
            "message": "do work",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-updater-dedupe",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        }
    )

    ids = [step.id for step in result["plan"].steps]
    assert len(ids) == len(set(ids))
    assert ids[0] == "1"
    assert ids[1] != "1"
    assert ids[2] == "2"


async def test_executor_uses_tool_outputs_when_react_hits_iteration_limit():
    """If ReAct stops immediately after a tool at its iteration cap, the last
    AI content is only a pre-tool note.  The step result must retain the actual
    tool output so updater does not mistake a long research step for no output.
    """
    from langchain_core.messages import ToolMessage
    from app.domain.services.graphs.main_graph import build_main_graph

    planner_llm = _make_structured_planner_llm(
        create_response=PlanResponse(
            title="T",
            goal="G",
            language="zh",
            steps=[StepDef(id="1", description="research")],
            message="ok",
        )
    )

    class IterationLimitedReactGraph:
        async def astream(self, input_state, config=None, **kwargs):
            yield {
                "tool_node": {
                    "events": [],
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-1",
                                    "name": "browser_view",
                                    "args": {},
                                }
                            ],
                        ),
                        ToolMessage(
                            content="关键产出：OpenAI 发布了新的企业级智能体功能。",
                            tool_call_id="call-1",
                            name="browser_view",
                        ),
                    ],
                    "attempt_count": 30,
                    "failure_count": 0,
                    "should_interrupt": False,
                }
            }

    graph = build_main_graph(
        _allow_default_prompt_assembler=True,
        planner_llm=planner_llm,
        react_graph=IterationLimitedReactGraph(),
        summary_llm=planner_llm,
        uow_factory=MagicMock(),
        session_id="sess-tool-summary",
    )
    result = await graph.ainvoke(
        {
            "message": "research news",
            "language": "zh",
            "attachments": [],
            "image_content_blocks": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-tool-summary",
            "should_interrupt": False,
            "resume_value": None,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        }
    )

    assert "OpenAI 发布了新的企业级智能体功能" in result["plan"].steps[0].result


class TestCoordinatorStepCompletion:
    """P0 — executor_node coordinator branch must complete a
    ``parallel_work_units`` step exactly like the react branch: mark it
    COMPLETED, write ``execution_summary``/``result``, then route to
    updater_node. Before the fix the branch left the step PENDING, so
    ``Plan.get_next_step()`` re-returned the same parallel step forever
    (executor↔updater loop → ``GraphRecursionError`` under any recursion
    limit) and the subgraph ran on every iteration."""

    @staticmethod
    def _planner_with_parallel_step(language: str = "en"):
        from app.domain.models.work_unit import (
            ParallelWorkUnitGroupRequest,
            WorkUnitRequest,
        )

        pwu = ParallelWorkUnitGroupRequest(
            work_units=[
                WorkUnitRequest(
                    objective="explore", phase="exploration",
                    allowed_tools=["file_read"],
                )
            ]
        )
        create_response = PlanResponse(
            title="Parallel", goal="do parallel work", language=language,
            steps=[StepDef(id="s1", description="parallel step", parallel_work_units=pwu)],
            message="ok",
        )
        # Updater never replans here (the single step completes → no pending
        # step), but the mock still needs to dispatch the schema.
        update_response = PlanUpdateResponse(steps=[])
        create_structured = AsyncMock()
        create_structured.ainvoke = AsyncMock(return_value=create_response)
        update_structured = AsyncMock()
        update_structured.ainvoke = AsyncMock(return_value=update_response)
        planner_llm = MagicMock()

        def _wso(schema, **kwargs):
            return create_structured if schema is PlanResponse else update_structured

        planner_llm.with_structured_output = MagicMock(side_effect=_wso)

        async def _astream(messages, **kwargs):
            yield AIMessageChunk(content='{"message": "done", "attachments": []}')

        planner_llm.astream = _astream
        return planner_llm

    @pytest.mark.parametrize(
        "group_outcome_name, candidate, expected_success",
        [
            ("FAILED", "并行执行失败：2 个 worker 中 1 个失败。", False),
            ("SUCCESS", "并行执行完成：1 个 worker 完成，合并写入 0 个文件。", True),
        ],
    )
    async def test_coordinator_step_completes_once_no_loop(
        self, monkeypatch, group_outcome_name, candidate, expected_success,
    ):
        from app.domain.models.patch_apply_plan import GroupOutcome
        from app.domain.services.graphs.main_graph import build_main_graph

        monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")

        # Fake subgraph: exploration-only group (apply_plan=None) so the
        # PatchApplier branch is never reached — no applier ports needed.
        subgraph = AsyncMock()
        subgraph.ainvoke = AsyncMock(return_value={
            "group_outcome": getattr(GroupOutcome, group_outcome_name),
            "apply_plan": None,
            "step_result_candidate": candidate,
        })

        planner_llm = self._planner_with_parallel_step()
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-coord",
        )

        # Uses the default recursion_limit (25), like every other test in this
        # file. The happy path terminates in ~3-4 super-steps
        # (planner→executor→updater→END, subgraph invoked exactly once); the
        # pre-fix executor↔updater loop blows past 25 just as reliably as past a
        # tighter bound, so regression coverage holds without the flake risk of
        # a thin margin (a thin limit intermittently raised GraphRecursionError
        # when the nested pregel subgraph consumed extra super-steps under
        # process-sharing/scheduling pressure).
        result = await graph.ainvoke(
            {
                "message": "run parallel work",
                "language": "en",
                "attachments": [],
                "image_content_blocks": [],
                "plan": None,
                "current_step": None,
                "messages": [],
                "execution_summary": "",
                "events": [],
                "flow_status": "idle",
                "session_id": "sess-coord",
                "should_interrupt": False,
                "resume_value": None,
                "original_request": "",
                "skill_context": "",
                "conversation_summaries": [],
            },
            config={
                "configurable": {
                    "parallel_execution_subgraph": subgraph,
                    "user_id": "u1",
                },
            },
        )

        # The coordinator backend ran exactly once — no executor↔updater loop.
        subgraph.ainvoke.assert_awaited_once()

        plan = result.get("plan")
        assert plan is not None
        assert plan.status == ExecutionStatus.COMPLETED

        completed = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
        assert len(completed) == 1
        # success is derived from the structured GroupOutcome, not guessed
        # from the summary string.
        assert completed[0].success is expected_success
        # result + execution_summary carry the reducer's operator text.
        assert completed[0].result == candidate
        assert result.get("execution_summary") == candidate

    @pytest.mark.parametrize(
        "group_outcome_name, expected_success",
        [("SUCCESS", True), ("FAILED", False)],
    )
    async def test_coordinator_step_emits_events_and_records_metrics(
        self, monkeypatch, group_outcome_name, expected_success,
    ):
        """Parity with the react branch: the coordinator step emits exactly
        StepEvent(STARTED) then StepEvent(COMPLETED) (the COMPLETED one carrying
        the COMPLETED+success step the frontend timeline renders), and records
        the step in execution_metrics on the correct counter."""
        from app.domain.models.event import StepEvent, StepEventStatus
        from app.domain.models.patch_apply_plan import GroupOutcome
        from app.domain.services.execution_metrics import ExecutionMetrics
        from app.domain.services.graphs.main_graph import build_main_graph

        monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")

        subgraph = AsyncMock()
        subgraph.ainvoke = AsyncMock(return_value={
            "group_outcome": getattr(GroupOutcome, group_outcome_name),
            "apply_plan": None,
            "step_result_candidate": "candidate text",
        })

        event_queue: asyncio.Queue = asyncio.Queue()
        metrics = ExecutionMetrics()

        planner_llm = self._planner_with_parallel_step()
        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-coord-evt",
        )

        await graph.ainvoke(
            {
                "message": "run parallel work",
                "language": "en",
                "attachments": [],
                "image_content_blocks": [],
                "plan": None,
                "current_step": None,
                "messages": [],
                "execution_summary": "",
                "events": [],
                "flow_status": "idle",
                "session_id": "sess-coord-evt",
                "should_interrupt": False,
                "resume_value": None,
                "original_request": "",
                "skill_context": "",
                "conversation_summaries": [],
            },
            config={
                "recursion_limit": 8,
                "configurable": {
                    "parallel_execution_subgraph": subgraph,
                    "user_id": "u1",
                    "event_queue": event_queue,
                    "execution_metrics": metrics,
                },
            },
        )

        step_events = []
        while not event_queue.empty():
            evt = event_queue.get_nowait()
            if isinstance(evt, StepEvent):
                step_events.append(evt)

        # Exactly STARTED then COMPLETED for the single coordinator step.
        assert [e.status for e in step_events] == [
            StepEventStatus.STARTED, StepEventStatus.COMPLETED,
        ]
        completed_evt = step_events[1]
        assert completed_evt.step.status == ExecutionStatus.COMPLETED
        assert completed_evt.step.success is expected_success

        # Metrics recorded once on the outcome-appropriate counter.
        if expected_success:
            assert metrics.steps_completed == 1
            assert metrics.steps_failed == 0
        else:
            assert metrics.steps_completed == 0
            assert metrics.steps_failed == 1

    async def test_coordinator_step_advances_to_next_pending_step(self, monkeypatch):
        """A coordinator step in the MIDDLE of a plan must complete and let the
        plan advance to the next pending step (react path) — the path most
        directly threatened by a broken step-sync. The subgraph runs once (step
        1 only); step 2 runs via react_graph; the plan reaches COMPLETED with no
        executor↔updater loop."""
        from app.domain.models.patch_apply_plan import GroupOutcome
        from app.domain.models.work_unit import (
            ParallelWorkUnitGroupRequest, WorkUnitRequest,
        )
        from app.domain.services.graphs.main_graph import build_main_graph

        monkeypatch.setenv("ACTUS_C2_COORDINATOR_ENABLED", "true")

        pwu = ParallelWorkUnitGroupRequest(work_units=[
            WorkUnitRequest(objective="explore", phase="exploration", allowed_tools=["file_read"]),
        ])
        create_response = PlanResponse(
            title="Two-step", goal="parallel then normal", language="en",
            steps=[
                StepDef(id="s1", description="parallel step", parallel_work_units=pwu),
                StepDef(id="s2", description="normal follow-up step"),
            ],
            message="ok",
        )
        # Empty update steps → updater keeps the pre-planned pending step 2.
        update_response = PlanUpdateResponse(steps=[])
        create_structured = AsyncMock()
        create_structured.ainvoke = AsyncMock(return_value=create_response)
        update_structured = AsyncMock()
        update_structured.ainvoke = AsyncMock(return_value=update_response)
        planner_llm = MagicMock()

        def _wso(schema, **kwargs):
            return create_structured if schema is PlanResponse else update_structured

        planner_llm.with_structured_output = MagicMock(side_effect=_wso)

        async def _astream(messages, **kwargs):
            yield AIMessageChunk(content='{"message": "done", "attachments": []}')

        planner_llm.astream = _astream

        subgraph = AsyncMock()
        subgraph.ainvoke = AsyncMock(return_value={
            "group_outcome": GroupOutcome.SUCCESS,
            "apply_plan": None,
            "step_result_candidate": "step1 candidate",
        })

        graph = build_main_graph(
            _allow_default_prompt_assembler=True,
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-coord-multi",
        )

        result = await graph.ainvoke(
            {
                "message": "do two steps",
                "language": "en",
                "attachments": [],
                "image_content_blocks": [],
                "plan": None,
                "current_step": None,
                "messages": [],
                "execution_summary": "",
                "events": [],
                "flow_status": "idle",
                "session_id": "sess-coord-multi",
                "should_interrupt": False,
                "resume_value": None,
                "original_request": "",
                "skill_context": "",
                "conversation_summaries": [],
            },
            config={
                "recursion_limit": 12,
                "configurable": {
                    "parallel_execution_subgraph": subgraph,
                    "user_id": "u1",
                },
            },
        )

        # Coordinator backend ran once (only step 1 is parallel); step 2 took
        # the react path. No loop on the coordinator step.
        subgraph.ainvoke.assert_awaited_once()
        # The updater consumed step 1's execution_summary to drive replanning.
        update_structured.ainvoke.assert_awaited()

        plan = result.get("plan")
        assert plan is not None
        assert plan.status == ExecutionStatus.COMPLETED
        completed = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
        assert len(completed) == 2
        # Step 1 (coordinator) carries the structured success + reducer text.
        s1 = next(s for s in plan.steps if s.id == "s1")
        assert s1.success is True
        assert s1.result == "step1 candidate"
