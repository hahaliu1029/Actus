from typing import Any

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import ErrorEvent, MessageEvent, ToolEvent
from app.domain.models.memory import Memory
from app.domain.models.tool_result import ToolResult
from app.domain.services.agents.base import BaseAgent
from app.domain.services.tools.base import BaseTool, tool


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


class _FailingLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("mock llm failure")


class _SequenceLLM:
    def __init__(self) -> None:
        self._index = 0

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        self._index += 1
        if self._index == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool-1",
                        "function": {
                            "name": "noop_tool",
                            "arguments": "null",
                        },
                    }
                ],
            }

        return {
            "role": "assistant",
            "content": "done",
            "tool_calls": [],
        }


class _DummyJsonParser:
    async def invoke(self, payload: Any) -> Any:
        if payload == "null":
            return None
        return {}


class _NoopTool(BaseTool):
    name = "noop"

    @tool(
        name="noop_tool",
        description="noop",
        parameters={},
        required=[],
    )
    async def noop_tool(self) -> ToolResult:
        return ToolResult(success=True, message="ok")


class _DummyAgent(BaseAgent):
    name = "dummy"
    _system_prompt = "test system prompt"


def _build_agent(llm: Any) -> _DummyAgent:
    return _DummyAgent(
        uow_factory=_DummyUoW,
        session_id="s-test",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=llm,
        json_parser=_DummyJsonParser(),
        tools=[_NoopTool()],
    )


async def test_invoke_returns_error_event_when_llm_fails_exhausting_retries() -> None:
    agent = _build_agent(_FailingLLM())

    events = [event async for event in agent.invoke("hello")]

    assert len(events) == 1
    assert isinstance(events[0], ErrorEvent)
    assert "调用语言模型失败" in events[0].error


async def test_invoke_handles_non_dict_tool_args_without_crashing() -> None:
    agent = _build_agent(_SequenceLLM())

    events = [event async for event in agent.invoke("hello")]

    assert any(isinstance(event, ToolEvent) for event in events)
    assert isinstance(events[-1], MessageEvent)
    assert events[-1].message == "done"
