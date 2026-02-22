import pytest
from typing import Any

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import MessageEvent, PlanEvent, PlanEventStatus
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import ExecutionStatus, Plan, Step
from app.domain.services.agents.planner import PlannerAgent

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _DummySessionRepo:
    def __init__(self) -> None:
        self._memory = Memory()

    async def get_memory(self, _session_id: str, _agent_name: str) -> Memory:
        return self._memory

    async def save_memory(
        self, _session_id: str, _agent_name: str, memory: Memory
    ) -> None:
        self._memory = memory


class _DummyUoW:
    def __init__(self) -> None:
        self.session = _DummySessionRepo()

    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyJsonParser:
    async def invoke(self, _payload: Any) -> Any:
        # 模拟异常场景：解析器返回空字符串，而非结构化字典
        return ""


class _DummyLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "unused"}


async def test_create_plan_degrades_when_parser_returns_non_dict() -> None:
    agent = PlannerAgent(
        uow_factory=_DummyUoW,
        session_id="s-planner",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield MessageEvent(
            role="assistant",
            message="我会先读取附件并输出可执行工作计划。",
        )

    # 仅测试 create_plan 的结构化降级逻辑
    agent.invoke = fake_invoke  # type: ignore[method-assign]

    message = Message(message="读取我上传的文件，输出一个可执行的工作计划")
    events = [event async for event in agent.create_plan(message)]

    plan_events = [event for event in events if isinstance(event, PlanEvent)]
    assert len(plan_events) == 1
    assert plan_events[0].status == PlanEventStatus.CREATED
    assert len(plan_events[0].plan.steps) >= 1
    assert plan_events[0].plan.steps[0].description != ""


async def test_update_plan_degrades_when_parser_returns_non_dict() -> None:
    agent = PlannerAgent(
        uow_factory=_DummyUoW,
        session_id="s-planner-update",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield MessageEvent(
            role="assistant",
            message="计划更新完成。",
        )

    # 仅测试 update_plan 的结构化降级逻辑
    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="读取附件并整理要点")])
    current_step = plan.steps[0]
    current_step.status = ExecutionStatus.COMPLETED
    current_step.success = True
    current_step.result = "已提取关键信息"

    events = [event async for event in agent.update_plan(plan, current_step)]

    plan_events = [event for event in events if isinstance(event, PlanEvent)]
    assert len(plan_events) == 1
    assert plan_events[0].status == PlanEventStatus.UPDATED
    assert plan_events[0].plan == plan
    assert plan_events[0].plan.steps[0].description == "读取附件并整理要点"
