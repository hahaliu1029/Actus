import pytest
from typing import Any

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import MessageEvent, StepEvent
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.services.agents.react import ReActAgent

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


async def test_execute_step_degrades_when_parser_returns_non_dict() -> None:
    agent = ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )

    async def fake_invoke(_query: str):
        yield MessageEvent(role="assistant", message="这是一段普通文本结果")

    # 仅测试 execute_step 的结构化降级逻辑
    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="测试步骤")])
    step = plan.steps[0]
    assert step is not None
    message = Message(message="请执行测试步骤")

    events = [event async for event in agent.execute_step(plan, step, message)]

    step_events = [event for event in events if isinstance(event, StepEvent)]
    message_events = [event for event in events if isinstance(event, MessageEvent)]

    assert len(step_events) == 2
    assert step_events[-1].step.status.value == "completed"
    assert step_events[-1].step.result == "这是一段普通文本结果"
    assert len(message_events) == 1
    assert message_events[0].message == "这是一段普通文本结果"
