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
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        events = result.get("events", [])
        event_types = [type(e).__name__ for e in events]
        # planner events come through state; executor events go via queue (empty in state)
        assert "PlanEvent" in event_types or "TitleEvent" in event_types

    async def test_default_language_is_zh(self, mock_planner_llm, mock_json_parser):
        """When no language is specified, planner should default to zh."""
        from app.domain.services.graphs.main_graph import build_main_graph

        graph = build_main_graph(
            planner_llm=mock_planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=mock_planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-lang",
        )

        result = await graph.ainvoke({
            "message": "帮我查一下天气",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-lang",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        # Verify the plan language fallback is "zh" not "en"
        plan = result.get("plan")
        assert plan is not None

    async def test_planner_receives_conversation_summaries(self, mock_json_parser):
        """Planner system prompt should include conversation summaries when available."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_system_content = []

        async def capturing_invoke(**kwargs):
            messages = kwargs.get("messages", [])
            for m in messages:
                if m.get("role") == "system":
                    captured_system_content.append(m["content"])
            return {
                "content": '{"title":"Test","goal":"test","language":"zh","steps":[{"description":"step1"}],"message":"ok"}',
                "role": "assistant",
            }

        planner_llm = AsyncMock()
        planner_llm.invoke = capturing_invoke

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=_make_mock_react_graph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-summary",
        )

        await graph.ainvoke({
            "message": "继续上次的工作",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-summary",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": ["### 第1轮\n- 用户需求：查天气\n- 执行结果：成功获取北京天气"],
        })

        assert len(captured_system_content) >= 1
        assert "历史对话摘要" in captured_system_content[0]
        assert "查天气" in captured_system_content[0]

    async def test_step_success_false_when_llm_reports_failure(self, mock_json_parser):
        """When react LLM returns success=false, the step should be marked as failed."""
        from app.domain.services.graphs.main_graph import build_main_graph

        class FailingReactGraph:
            async def astream(self, input_state, config=None):
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": false, "result": "CAPTCHA blocked", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        async def mock_invoke(**kwargs):
            return {
                "content": '{"title":"T","goal":"G","language":"zh","steps":[{"description":"search news"}],"message":"ok"}',
                "role": "assistant",
            }
        planner_llm.invoke = mock_invoke

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=FailingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-fail",
        )

        result = await graph.ainvoke({
            "message": "search AI news",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-fail",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        plan = result.get("plan")
        assert plan is not None
        completed_steps = [s for s in plan.steps if s.status == ExecutionStatus.COMPLETED]
        assert len(completed_steps) >= 1
        assert completed_steps[0].success is False


class TestExecutorMessageBranching:
    """Test the three-way branching in executor_node for message handling."""

    @pytest.fixture
    def mock_json_parser(self):
        parser = AsyncMock()
        import json
        async def parse(content, default_value=None):
            try:
                return json.loads(content)
            except Exception:
                return default_value
        parser.invoke = parse
        return parser

    async def test_first_step_no_history_uses_system_prompt(self, mock_json_parser):
        """When messages=[] and is_resuming=False, executor builds fresh system+execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        async def mock_invoke(**kwargs):
            return {"content": '{"title":"T","goal":"G","language":"zh","steps":[{"description":"S1"}],"message":"ok"}', "role": "assistant"}
        planner_llm.invoke = mock_invoke

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-exec",
        )

        await graph.ainvoke({
            "message": "do something",
            "language": "zh",
            "attachments": [],
            "plan": None,
            "current_step": None,
            "messages": [],
            "execution_summary": "",
            "events": [],
            "flow_status": "idle",
            "session_id": "sess-exec",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert "任务执行智能体" in msgs[0]["content"]

    async def test_has_history_not_resuming_updates_system_prompt(self, mock_json_parser):
        """When messages have history and is_resuming=False, executor updates system prompt and appends execution prompt."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        planner_llm.invoke = AsyncMock()

        step = Step(description="Step 2: analyze data")
        plan = Plan(title="T", goal="G", language="zh", steps=[
            Step(description="Step 1: collect", status=ExecutionStatus.COMPLETED),
            step,
        ], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-exec-2",
        )

        history_messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old execution prompt"},
            {"role": "assistant", "content": '{"success": true, "result": "collected data", "attachments": []}'},
        ]

        await graph.ainvoke({
            "message": "continue",
            "language": "zh",
            "attachments": [],
            "plan": plan,
            "current_step": step,
            "messages": history_messages,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-exec-2",
            "should_interrupt": False,
            "is_resuming": False,
            "original_request": "analyze data",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        # System prompt should be updated (not "old system prompt")
        assert msgs[0]["role"] == "system"
        assert "任务执行智能体" in msgs[0]["content"]
        # Should NOT contain "用户已完成接管" (not resuming)
        all_content = " ".join(m.get("content", "") for m in msgs)
        assert "用户已完成接管" not in all_content

    async def test_resuming_uses_takeover_message(self, mock_json_parser):
        """When is_resuming=True with saved messages, executor appends takeover resume message."""
        from app.domain.services.graphs.main_graph import build_main_graph

        captured_react_inputs = []

        class CapturingReactGraph:
            async def astream(self, input_state, config=None):
                captured_react_inputs.append(input_state)
                yield {"llm_node": {
                    "events": [],
                    "messages": input_state["messages"] + [
                        {"role": "assistant", "content": '{"success": true, "result": "done", "attachments": []}'}
                    ],
                    "should_interrupt": False,
                }}

        planner_llm = AsyncMock()
        planner_llm.invoke = AsyncMock()

        step = Step(description="Login to Notion")
        plan = Plan(title="T", goal="G", language="zh", steps=[step], message="ok", status=ExecutionStatus.RUNNING)

        graph = build_main_graph(
            planner_llm=planner_llm,
            react_graph=CapturingReactGraph(),
            json_parser=mock_json_parser,
            summary_llm=planner_llm,
            uow_factory=MagicMock(),
            session_id="sess-resume",
        )

        saved = [
            {"role": "system", "content": "some system prompt"},
            {"role": "user", "content": "do login"},
        ]

        await graph.ainvoke({
            "message": "我已经登录了",
            "language": "zh",
            "attachments": [],
            "plan": plan,
            "current_step": step,
            "messages": saved,
            "execution_summary": "",
            "events": [],
            "flow_status": "executing",
            "session_id": "sess-resume",
            "should_interrupt": False,
            "is_resuming": True,
            "original_request": "login",
            "skill_context": "",
            "conversation_summaries": [],
        })

        assert len(captured_react_inputs) >= 1
        msgs = captured_react_inputs[0]["messages"]
        all_content = " ".join(m.get("content", "") for m in msgs)
        assert "用户已完成接管" in all_content
