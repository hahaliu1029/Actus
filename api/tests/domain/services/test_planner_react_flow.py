"""Tests for PlannerReActFlow — LangGraph-based implementation."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langgraph.checkpoint.memory import MemorySaver

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import DoneEvent, PlanEvent, WaitEvent
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_llm():
    """Mock BaseChatModel LLM."""
    llm = MagicMock(spec=BaseChatModel)
    # with_structured_output returns a runnable whose ainvoke returns a parsed model
    mock_structured = AsyncMock()
    llm.with_structured_output = MagicMock(return_value=mock_structured)
    # Default: return None (callers override per-test as needed)
    mock_structured.ainvoke = AsyncMock(return_value=None)
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


def _make_flow(mock_llm, mock_uow, **overrides):
    """Helper to create a PlannerReActFlow with standard test params."""
    kwargs = dict(
        uow_factory=MagicMock(return_value=mock_uow),
        llm=mock_llm,
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="test-session",
        browser=AsyncMock(),
        sandbox=AsyncMock(),
        search_engine=AsyncMock(),
        mcp_tool=MagicMock(get_tools=MagicMock(return_value=[])),
        a2a_tool=MagicMock(manager=None),
        skill_tool=MagicMock(),
        checkpointer=MemorySaver(),
    )
    kwargs.update(overrides)
    return PlannerReActFlow(**kwargs)


def test_planner_react_flow_constructs_successfully(mock_llm, mock_uow):
    """Flow can be constructed with all required parameters (graphs not built yet)."""
    flow = _make_flow(mock_llm, mock_uow)
    assert flow.done is True
    assert flow.plan is None
    # 延迟绑定：构造时不构建图
    assert flow._graphs_built is False


async def test_planner_react_flow_invoke_produces_events(mock_llm, mock_uow):
    """Flow.invoke() should yield events including DoneEvent."""
    flow = _make_flow(mock_llm, mock_uow)

    # Mock the graph to produce a DoneEvent without real LLM calls
    mock_bridge = MagicMock()
    mock_bridge.final_state = {
        "plan": Plan(title="Test", goal="test", language="en",
                     steps=[Step(description="s1")], message="ok"),
        "messages": [SystemMessage(content="sys"), HumanMessage(content="hi"),
                     AIMessage(content="done")],
        "original_request": "test",
        "should_interrupt": False,
        "flow_status": "completed",
    }

    async def mock_run(graph, input_state, config=None):
        from app.domain.models.event import TitleEvent, MessageEvent, PlanEvent, PlanEventStatus
        yield TitleEvent(title="Test")
        yield MessageEvent(role="assistant", message="ok")
        yield PlanEvent(
            plan=mock_bridge.final_state["plan"],
            status=PlanEventStatus.CREATED,
        )
        yield DoneEvent()

    mock_bridge.run = mock_run

    events = []
    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        async for event in flow.invoke(Message(message="help me test")):
            events.append(event)

    assert len(events) > 0
    assert any(isinstance(e, DoneEvent) for e in events)


async def test_planner_react_flow_produces_plan_event(mock_llm, mock_uow):
    """Flow should produce PlanEvent during execution."""
    flow = _make_flow(mock_llm, mock_uow)

    mock_bridge = MagicMock()
    plan = Plan(title="Test", goal="test", language="en",
                steps=[Step(description="s1")], message="ok")
    mock_bridge.final_state = {
        "plan": plan,
        "messages": [SystemMessage(content="sys"), AIMessage(content="done")],
        "original_request": "test",
        "should_interrupt": False,
        "flow_status": "completed",
    }

    async def mock_run(graph, input_state, config=None):
        from app.domain.models.event import PlanEvent, PlanEventStatus
        yield PlanEvent(plan=plan, status=PlanEventStatus.CREATED)
        yield DoneEvent()

    mock_bridge.run = mock_run

    events = []
    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        async for event in flow.invoke(Message(message="help me test")):
            events.append(event)

    plan_events = [e for e in events if isinstance(e, PlanEvent)]
    assert len(plan_events) >= 1


def test_planner_react_flow_skill_context_provider(mock_llm, mock_uow):
    """_skill_context_provider callback drives _get_skill_context_seed.

    Replaces the legacy set_skill_context test (TODO #30 retired the
    self._skill_context instance field; the provider callback is the
    new clock 2 replacement).
    """
    flow = _make_flow(mock_llm, mock_uow)

    # Default: no provider wired -> seed returns ""
    assert flow._skill_context_provider is None
    assert flow._get_skill_context_seed() == ""

    # Wire a provider -> seed returns its value
    flow._skill_context_provider = lambda: "## test context"
    assert flow._get_skill_context_seed() == "## test context"


async def test_persist_after_graph_saves_memory_on_interrupt(
    mock_llm, mock_uow,
):
    """_persist_after_graph should save Memory when should_interrupt=True.
    Checkpointer handles state persistence — no InterruptState needed."""
    flow = _make_flow(mock_llm, mock_uow)

    plan = Plan(title="t", goal="g", language="zh", steps=[Step(description="s1")], message="m")
    step = plan.steps[0]
    msgs = [SystemMessage(content="sys"), HumanMessage(content="u")]
    final_state = {
        "plan": plan,
        "current_step": step,
        "messages": msgs,
        "original_request": "g",
        "should_interrupt": True,
        "flow_status": "executing",
    }

    await flow._persist_after_graph(final_state, summaries=[])

    # Memory should be saved
    mock_uow.session.save_memory.assert_called_once()
    # Flow status should be EXECUTING for resume
    from app.domain.services.flows.base import FlowStatus
    assert flow.status == FlowStatus.EXECUTING


async def test_ensure_graphs_builds_tools_lazily(mock_llm, mock_uow):
    """_ensure_graphs should build tools from MCP/A2A at call time, not __init__ time."""
    mock_mcp = MagicMock()
    # MCP returns tools only after initialization
    mock_mcp.get_tools = MagicMock(return_value=[
        {"function": {"name": "notion_search", "description": "Search Notion"}},
    ])
    mock_a2a = MagicMock()
    mock_a2a.manager = None  # A2A not initialized

    flow = _make_flow(mock_llm, mock_uow,
                      mcp_tool=mock_mcp, a2a_tool=mock_a2a)
    flow._allow_default_prompt_assembler = True

    # Configure always_bind so MCP tools are queried during _ensure_graphs
    flow._mcp_always_bind_names = {"notion_search"}

    # Before _ensure_graphs: no graphs
    assert flow._graphs_built is False
    assert flow._react_graph is None

    # Call _ensure_graphs: should pick up MCP tools (filtered by always_bind)
    await flow._ensure_graphs()

    assert flow._graphs_built is True
    assert flow._react_graph is not None
    assert flow._main_graph is not None
    # Verify MCP tools were queried (for always_bind filtering)
    mock_mcp.get_tools.assert_called()


async def test_ensure_graphs_passes_tool_result_max_chars(mock_llm, mock_uow):
    """_ensure_graphs should forward overflow_config.tool_result_max_chars to build_react_graph."""
    from app.domain.models.context_overflow_config import ContextOverflowConfig

    overflow = ContextOverflowConfig(tool_result_max_chars=4000)
    flow = _make_flow(mock_llm, mock_uow, overflow_config=overflow)
    flow._allow_default_prompt_assembler = True

    with patch(
        "app.domain.services.flows.planner_react.build_react_graph",
        return_value=MagicMock(),
    ) as mock_build:
        await flow._ensure_graphs()

    mock_build.assert_called_once()
    _, kwargs = mock_build.call_args
    assert kwargs["tool_result_max_chars"] == 4000


async def test_generator_early_close_still_persists(
    mock_llm, mock_uow,
):
    """Simulates WaitEvent early-return: closing the generator triggers finally persistence."""
    flow = _make_flow(mock_llm, mock_uow)

    plan = Plan(title="t", goal="g", language="zh", steps=[Step(description="s1")], message="m")
    wait_event = WaitEvent()

    # Patch GraphEventBridge to yield a WaitEvent and set should_interrupt in final_state
    mock_bridge_instance = MagicMock()
    mock_bridge_instance.final_state = {
        "plan": plan,
        "current_step": plan.steps[0],
        "messages": [SystemMessage(content="sys")],
        "original_request": "g",
        "should_interrupt": True,
        "flow_status": "executing",
    }

    async def mock_run(graph, input_state, config=None):
        yield wait_event

    mock_bridge_instance.run = mock_run

    with patch(
        "app.domain.services.flows.planner_react.GraphEventBridge",
        return_value=mock_bridge_instance,
    ), patch.object(flow, "_ensure_graphs", new_callable=AsyncMock):
        gen = flow.invoke(Message(message="test"))
        # Consume only the first event (WaitEvent), then close — simulating early return
        first_event = await gen.__anext__()
        assert isinstance(first_event, WaitEvent)
        await gen.aclose()  # Simulates the consumer abandoning the generator

    # Memory should be saved via finally block, even on early close
    mock_uow.session.save_memory.assert_called()


class TestRunPlannerForDetectionLanguageDispatch:
    """#29 — verify ``_run_planner_for_detection`` honors ``message.language``.

    These tests directly target the 6 ``getattr(message, "language", "zh")``
    call sites replaced in Task 2: lines ~714, 715, 795, 809, 999, 1022.

    All assertions are on function-externally-observable outputs (the
    returned ``plan`` or the ``input_for_graph`` dict passed to
    ``build_main_graph``), never on internal local variables.

    TDD note: tests 11, 12, 14, 15 pass immediately on first run because
    Task 1 added ``Message.language`` and ``getattr`` finds the real field.
    They serve as post-replacement regression guards. Test 13 has a
    slightly stronger RED state because it exercises the ``or`` fallback.
    """

    async def test_run_planner_for_detection_en_loads_en_prompt_bundle(
        self, mock_llm, mock_uow, monkeypatch
    ) -> None:
        """#29 test 11: Message(language="en") selects EN PromptBundle.

        Covers planner_react.py lines ~714 and ~715
        (get_prompt_bundle + get_prompt_section_bundle both use message.language).

        NOTE: The functions are imported *locally* inside
        ``_run_planner_for_detection`` (not at module level), so they are
        resolved from ``app.domain.services.prompts`` at call time.
        We patch the source module attributes; the local ``from ... import``
        then picks up the spy wrappers.
        """
        from app.domain.models.message import Message
        import app.domain.services.prompts as prompts_mod

        flow = _make_flow(mock_llm, mock_uow)
        flow._supports_vision = False
        flow._prompt_assembler = None
        flow._allow_default_prompt_assembler = True

        calls: list[tuple[str, str]] = []
        real_get_bundle = prompts_mod.get_prompt_bundle
        real_get_section = prompts_mod.get_prompt_section_bundle

        def _spy_bundle(lang: str):
            calls.append(("bundle", lang))
            return real_get_bundle(lang)

        def _spy_section(lang: str):
            calls.append(("section", lang))
            return real_get_section(lang)

        monkeypatch.setattr(prompts_mod, "get_prompt_bundle", _spy_bundle)
        monkeypatch.setattr(prompts_mod, "get_prompt_section_bundle", _spy_section)

        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=Exception("force fallback so we don't need a real plan")
        )

        msg = Message(message="hello", language="en")
        await flow._run_planner_for_detection(msg, [])

        bundle_langs = [lang for kind, lang in calls if kind == "bundle"]
        section_langs = [lang for kind, lang in calls if kind == "section"]
        assert "en" in bundle_langs, f"expected 'en' in {bundle_langs}"
        assert "en" in section_langs, f"expected 'en' in {section_langs}"

    async def test_run_planner_for_detection_fallback_plan_inherits_message_language(
        self, mock_llm, mock_uow
    ) -> None:
        """#29 test 12: structured output raises → fallback PlanResponse
        is built from message.language, then wrapped into plan.language.

        Covers planner_react.py line ~795 (fallback PlanResponse.language).
        Assertion is on the observable returned ``plan.language``, NOT
        on the internal local ``PlanResponse`` variable.
        """
        from app.domain.models.message import Message

        flow = _make_flow(mock_llm, mock_uow)
        flow._supports_vision = False
        flow._prompt_assembler = None
        flow._allow_default_prompt_assembler = True

        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=RuntimeError("simulated structured output failure")
        )

        msg = Message(message="please help", language="en")
        plan, _ = await flow._run_planner_for_detection(msg, [])

        assert plan.language == "en", (
            "fallback path should inherit message.language; "
            f"got plan.language={plan.language!r}"
        )

    async def test_run_planner_for_detection_parsed_plan_inherits_message_language_when_empty(
        self, mock_llm, mock_uow
    ) -> None:
        """#29 test 13: structured output returns PlanResponse(language="")
        → plan.language falls back to message.language via the `or` chain.

        Covers planner_react.py line ~809
        (``parsed.language or message.language``).
        """
        from app.domain.models.message import Message
        from app.domain.models.llm_responses import PlanResponse, StepDef

        flow = _make_flow(mock_llm, mock_uow)
        flow._supports_vision = False
        flow._prompt_assembler = None
        flow._allow_default_prompt_assembler = True

        # Return a valid PlanResponse but with empty language
        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlanResponse(
                title="Task",
                goal="help",
                language="",  # empty → should fall back to message.language
                steps=[StepDef(description="do something")],
                message="on it",
            )
        )

        msg = Message(message="please help", language="en")
        plan, _ = await flow._run_planner_for_detection(msg, [])

        assert plan.language == "en", (
            "empty parsed.language should fall back to message.language via "
            f"`or`; got plan.language={plan.language!r}"
        )

    async def test_run_planner_for_detection_en_when_skill_tools_available_but_not_used(
        self, mock_llm, mock_uow, monkeypatch
    ) -> None:
        """#29 test 14: skill creation tools configured AND skill graph
        canary active, but planner's plan does NOT reference skill creation
        → reaches the branch that constructs ``input_for_graph`` via
        pre-computed plan.

        Covers planner_react.py line ~999.

        NOTE 1: this is NOT the real "skill creation path" — that returns
        early at the subgraph dispatch. This path triggers when
        ``_skill_tools_available`` is True but ``_plan_uses_skill_creation``
        is False.

        NOTE 2: line ~977's ``_skill_tools_available`` gate has TWO
        conditions — non-None tools AND ``_is_skill_graph_active()``
        returning True. The canary gate depends on
        ``skill_graph_canary_percent``, which defaults to 0 in
        ``_make_flow``'s kwargs → gate defaults to False. We must force
        the gate open via instance-level monkeypatch; otherwise the test
        silently falls through to the no-skill-tools branch (wrong branch).
        """
        from app.domain.models.message import Message
        from app.domain.models.llm_responses import PlanResponse, StepDef

        # Give the flow non-None skill creation tools
        flow = _make_flow(
            mock_llm,
            mock_uow,
            create_skill_tool=MagicMock(),
            brainstorm_skill_tool=MagicMock(),
        )
        flow._supports_vision = False
        flow._prompt_assembler = None
        flow._allow_default_prompt_assembler = True

        # CRITICAL: force the canary gate open. Without this,
        # _skill_tools_available is False regardless of the tools above.
        monkeypatch.setattr(
            flow,
            "_is_skill_graph_active",
            lambda: True,
        )

        mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlanResponse(
                title="Do work",
                goal="do work",
                language="en",
                steps=[StepDef(description="just a normal step")],
                message="sure",
            )
        )

        # Force _plan_uses_skill_creation → False so we don't take the
        # subgraph early-return branch. Instance-level patch avoids
        # staticmethod descriptor binding issues.
        monkeypatch.setattr(
            flow,
            "_plan_uses_skill_creation",
            lambda _plan: False,
        )

        # Intercept build_main_graph → capture input_for_graph
        captured_inputs: list[dict] = []

        class _FakeGraph:
            async def astream(self, input_dict, *args, **kwargs):
                del args, kwargs  # mock ignores the astream kwargs
                captured_inputs.append(input_dict)
                # empty async generator: iterate over a runtime-assigned
                # empty list so Pylance doesn't flag the yield as
                # statically unreachable
                _empty: list = []
                for item in _empty:
                    yield item

        def _fake_build_main_graph(*args, **kwargs):
            del args, kwargs
            return _FakeGraph()

        monkeypatch.setattr(
            "app.domain.services.flows.planner_react.build_main_graph",
            _fake_build_main_graph,
        )

        msg = Message(message="please help with english work", language="en")
        async for _ in flow.invoke(msg):
            pass

        assert captured_inputs, "expected at least one astream call"
        assert captured_inputs[0].get("language") == "en", (
            "skill-tools-available-no-skill-intent path should pass en to "
            f"input_for_graph; got {captured_inputs[0].get('language')!r}"
        )

    async def test_run_planner_for_detection_en_without_skill_creation_tools(
        self, mock_llm, mock_uow, monkeypatch
    ) -> None:
        """#29 test 15: no skill creation tools configured
        (``_skill_tools_available`` is False) → reaches the else branch
        that constructs ``input_for_graph`` without pre-computed planning.

        Covers planner_react.py line ~1022.

        NOTE: contrasts with test 14 — that test forces the canary gate
        OPEN; this test leaves the default-False canary gate alone AND
        sets tools to None, so either condition alone would land us in
        this else branch.
        """
        from app.domain.models.message import Message

        flow = _make_flow(
            mock_llm,
            mock_uow,
            create_skill_tool=None,
            brainstorm_skill_tool=None,
        )
        flow._supports_vision = False
        flow._prompt_assembler = None
        flow._allow_default_prompt_assembler = True

        captured_inputs: list[dict] = []

        class _FakeGraph:
            async def astream(self, input_dict, *args, **kwargs):
                del args, kwargs  # mock ignores the astream kwargs
                captured_inputs.append(input_dict)
                _empty: list = []
                for item in _empty:
                    yield item

        def _fake_build_main_graph(*args, **kwargs):
            del args, kwargs
            return _FakeGraph()

        monkeypatch.setattr(
            "app.domain.services.flows.planner_react.build_main_graph",
            _fake_build_main_graph,
        )

        msg = Message(message="please help", language="en")
        async for _ in flow.invoke(msg):
            pass

        assert captured_inputs, "expected at least one astream call"
        assert captured_inputs[0].get("language") == "en", (
            "no-skill-tools path should pass en to input_for_graph; "
            f"got {captured_inputs[0].get('language')!r}"
        )
