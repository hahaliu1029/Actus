import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.conversation_summary import ConversationSummary
from app.domain.models.context_overflow_config import ContextOverflowConfig
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.models.session import Session, SessionStatus
from app.domain.models.event import (
    ControlAction,
    ControlEvent,
    ControlScope,
    ControlSource,
    DoneEvent,
    PlanEvent,
    PlanEventStatus,
    WaitEvent,
)
from app.domain.services.flows.planner_react import PlannerReActFlow

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _DummyUoW:
    def __init__(self, session=None) -> None:
        self.session = session

    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def _uow_factory() -> _DummyUoW:
    return _DummyUoW()


class _CapturedAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def set_runtime_system_context(self, _context: str) -> None:
        return None


def test_planner_react_flow_passes_overflow_config_to_agents(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        _CapturedAgent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        _CapturedAgent,
    )

    overflow_config = ContextOverflowConfig(
        context_window=131072,
        context_overflow_guard_enabled=True,
    )

    flow = PlannerReActFlow(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        session_id="session-ctx-overflow",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
        overflow_config=overflow_config,
    )

    assert flow.planner.kwargs["overflow_config"] is overflow_config
    assert flow.react.kwargs["overflow_config"] is overflow_config


class _FlowSessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session
        self.updated_statuses: list[SessionStatus] = []
        self.summaries: list[ConversationSummary] = []

    async def get_by_id(self, _session_id: str) -> Session:
        return self._session

    async def update_status(self, _session_id: str, status: SessionStatus) -> None:
        self.updated_statuses.append(status)
        self._session.status = status

    async def get_summary(self, _session_id: str) -> list[ConversationSummary]:
        return list(self.summaries)

    async def save_summary(
        self, _session_id: str, summaries: list[ConversationSummary]
    ) -> None:
        self.summaries = list(summaries)


class _FlowAgent:
    def __init__(self, name: str = "agent", **kwargs) -> None:
        self.name = name
        self.kwargs = kwargs
        self.inject_calls: list[dict] = []
        self.update_plan_calls: list[dict] = []
        self.compact_calls: list[bool] = []
        self.summary_sets: list[list[ConversationSummary]] = []
        self.create_plan_calls: list[Message] = []

    def set_runtime_system_context(self, _context: str) -> None:
        return None

    def set_conversation_summaries(
        self, summaries: list[ConversationSummary]
    ) -> None:
        self.summary_sets.append(list(summaries))

    async def roll_back(self, _message: Message) -> None:
        return None

    def inject_context_anchor(
        self,
        session_status: str,
        user_message: str,
        original_request: str = "",
        completed_steps: list[str] | None = None,
    ) -> None:
        self.inject_calls.append(
            {
                "session_status": session_status,
                "user_message": user_message,
                "original_request": original_request,
                "completed_steps": completed_steps or [],
            }
        )

    def get_latest_assistant_content(self, max_chars: int = 500) -> str:
        return "来自react的执行摘要"[:max_chars]

    async def execute_step(self, plan: Plan, step: Step, _message: Message):
        step.status = ExecutionStatus.COMPLETED
        step.success = True
        step.result = "步骤执行完成"
        if False:
            yield None

    async def compact_memory(self, keep_summary: bool = False) -> None:
        self.compact_calls.append(keep_summary)

    async def summarize(self):
        if False:
            yield None

    async def create_plan(self, _message: Message):
        self.create_plan_calls.append(_message)
        if False:
            yield None

    async def update_plan(
        self, plan: Plan, step: Step, execution_summary: str = ""
    ):
        self.update_plan_calls.append(
            {
                "plan": plan,
                "step": step,
                "execution_summary": execution_summary,
            }
        )
        yield PlanEvent(plan=plan, status=PlanEventStatus.UPDATED)


class _WaitingFlowAgent(_FlowAgent):
    async def execute_step(self, plan: Plan, step: Step, _message: Message):
        step.status = ExecutionStatus.RUNNING
        yield WaitEvent()


class _TakeoverFlowAgent(_FlowAgent):
    async def execute_step(self, plan: Plan, step: Step, _message: Message):
        step.status = ExecutionStatus.RUNNING
        yield ControlEvent(
            action=ControlAction.REQUESTED,
            scope=ControlScope.SHELL,
            source=ControlSource.AGENT,
        )


async def test_planner_react_flow_injects_anchor_and_passes_execution_summary(
    monkeypatch,
) -> None:
    completed_step = Step(
        description="已完成步骤",
        status=ExecutionStatus.COMPLETED,
        success=True,
        result="ok",
    )
    pending_step = Step(description="待执行步骤")
    plan = Plan(goal="完成数据分析", steps=[completed_step, pending_step])
    session = Session(
        id="session-anchor",
        status=SessionStatus.RUNNING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)

    planner_agent = _FlowAgent(name="planner")
    react_agent = _FlowAgent(name="react")

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-anchor",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    events = [event async for event in flow.invoke(Message(message="继续执行"))]

    assert planner_agent.inject_calls[0]["session_status"] == "running"
    assert planner_agent.inject_calls[0]["user_message"] == "继续执行"
    assert planner_agent.inject_calls[0]["original_request"] == "完成数据分析"
    assert planner_agent.inject_calls[0]["completed_steps"] == ["已完成步骤"]
    assert react_agent.inject_calls[0]["session_status"] == "running"
    assert planner_agent.update_plan_calls[0]["execution_summary"] == "来自react的执行摘要"
    assert react_agent.compact_calls == [True]
    assert session_repo.updated_statuses[0] == SessionStatus.RUNNING
    assert isinstance(events[-1], DoneEvent)


class _SummaryJsonParser:
    async def invoke(self, payload):
        import json

        if isinstance(payload, str):
            return json.loads(payload)
        return payload


class _SummaryLLM:
    async def invoke(self, **kwargs):
        return {
            "role": "assistant",
            "content": (
                '{"user_intent":"继续分析数据","plan_summary":"完成剩余步骤",'
                '"execution_results":["步骤执行完成"],'
                '"decisions":["保留现有方案"],"unresolved":[]}'
            ),
        }


async def test_planner_react_flow_loads_and_generates_summaries(monkeypatch) -> None:
    completed_step = Step(
        description="已完成步骤",
        status=ExecutionStatus.COMPLETED,
        success=True,
        result="ok",
    )
    pending_step = Step(description="待执行步骤")
    plan = Plan(goal="完成数据分析", steps=[completed_step, pending_step])
    session = Session(
        id="session-summary",
        status=SessionStatus.RUNNING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)
    session_repo.summaries = [
        ConversationSummary(
            round_number=1,
            user_intent="分析数据",
            plan_summary="读取 CSV",
            execution_results=["成功读取文件"],
            decisions=["按月聚合"],
            unresolved=[],
        )
    ]

    planner_agent = _FlowAgent(name="planner")
    react_agent = _FlowAgent(name="react")

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        summary_llm=_SummaryLLM(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-summary",
        json_parser=_SummaryJsonParser(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    _ = [event async for event in flow.invoke(Message(message="继续执行"))]

    assert planner_agent.summary_sets[0][0].round_number == 1
    assert react_agent.summary_sets[0][0].round_number == 1
    assert len(session_repo.summaries) == 2
    assert session_repo.summaries[-1].round_number == 2
    assert session_repo.summaries[-1].user_intent == "继续分析数据"


async def test_planner_react_flow_short_circuits_on_wait_event(monkeypatch) -> None:
    plan = Plan(goal="创建 skill", steps=[Step(description="等待用户确认蓝图")])
    session = Session(
        id="session-wait-short-circuit",
        status=SessionStatus.WAITING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)

    planner_agent = _FlowAgent(name="planner")
    react_agent = _WaitingFlowAgent(name="react")

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-wait-short-circuit",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    events = [event async for event in flow.invoke(Message(message="可以，继续生成"))]

    assert any(isinstance(event, WaitEvent) for event in events)
    assert not planner_agent.update_plan_calls
    assert not react_agent.compact_calls
    assert not any(isinstance(event, DoneEvent) for event in events)


async def test_planner_react_flow_short_circuits_on_control_requested(monkeypatch) -> None:
    plan = Plan(goal="创建 skill", steps=[Step(description="等待用户接管 shell")])
    session = Session(
        id="session-control-short-circuit",
        status=SessionStatus.WAITING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)

    planner_agent = _FlowAgent(name="planner")
    react_agent = _TakeoverFlowAgent(name="react")

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-control-short-circuit",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    events = [event async for event in flow.invoke(Message(message="请继续"))]

    assert any(
        isinstance(event, ControlEvent) and event.action == ControlAction.REQUESTED
        for event in events
    )
    assert not planner_agent.update_plan_calls
    assert not react_agent.compact_calls
    assert not any(isinstance(event, DoneEvent) for event in events)


async def test_planner_react_flow_skips_replanning_during_skill_confirmation_resume(
    monkeypatch,
) -> None:
    plan = Plan(goal="创建 skill", steps=[Step(description="继续生成 skill")])
    session = Session(
        id="session-skill-resume-running",
        status=SessionStatus.RUNNING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)

    planner_agent = _FlowAgent(name="planner")
    react_agent = _WaitingFlowAgent(name="react")
    react_agent._skill_creation_approved_actions = {"generate"}

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-skill-resume-running",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    events = [event async for event in flow.invoke(Message(message="好的，根据这个蓝图开始生产"))]

    assert any(isinstance(event, WaitEvent) for event in events)
    assert not planner_agent.create_plan_calls
    assert not planner_agent.update_plan_calls


async def test_planner_react_flow_skips_replanning_when_structured_generate_confirmation_arrives_without_runtime_token(
    monkeypatch,
) -> None:
    plan = Plan(goal="创建 skill", steps=[Step(description="继续生成 skill")])
    session = Session(
        id="session-skill-resume-structured-action",
        status=SessionStatus.RUNNING,
        events=[PlanEvent(plan=plan, status=PlanEventStatus.CREATED)],
    )
    session_repo = _FlowSessionRepo(session)

    planner_agent = _FlowAgent(name="planner")
    react_agent = _WaitingFlowAgent(name="react")

    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.PlannerAgent",
        lambda **kwargs: planner_agent,
    )
    monkeypatch.setattr(
        "app.domain.services.flows.planner_react.ReActAgent",
        lambda **kwargs: react_agent,
    )

    flow = PlannerReActFlow(
        uow_factory=lambda: _DummyUoW(session_repo),
        llm=object(),
        agent_config=AgentConfig(
            max_iterations=100,
            max_retries=3,
            max_search_results=10,
        ),
        session_id="session-skill-resume-structured-action",
        json_parser=object(),
        browser=object(),
        sandbox=object(),
        search_engine=object(),
        mcp_tool=object(),
        a2a_tool=object(),
        skill_tool=object(),
    )

    events = [
        event
        async for event in flow.invoke(
            Message(
                message="确认蓝图并开始生成",
                skill_confirmation_action="generate",
            )
        )
    ]

    assert any(isinstance(event, WaitEvent) for event in events)
    assert not planner_agent.create_plan_calls
    assert not planner_agent.update_plan_calls
