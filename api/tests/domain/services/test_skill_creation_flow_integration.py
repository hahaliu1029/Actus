from __future__ import annotations

import json
from typing import Any

import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.event import MessageEvent, StepEvent, WaitEvent
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.tool_result import ToolResult
from app.domain.services.agents.react import ReActAgent
from app.domain.services.tools.base import BaseTool, tool

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _SharedSessionRepo:
    def __init__(self) -> None:
        self._memory = Memory()
        self._skill_creation_state: SkillCreationState | None = None

    async def get_memory(self, _session_id: str, _agent_name: str) -> Memory:
        return self._memory

    async def save_memory(
        self, _session_id: str, _agent_name: str, memory: Memory
    ) -> None:
        self._memory = memory

    async def get_skill_creation_state(
        self, _session_id: str
    ) -> SkillCreationState | None:
        return self._skill_creation_state

    async def save_skill_creation_state(
        self, _session_id: str, state: SkillCreationState
    ) -> None:
        self._skill_creation_state = state

    async def clear_skill_creation_state(self, _session_id: str) -> None:
        self._skill_creation_state = None


class _DummyUoW:
    def __init__(self, session: _SharedSessionRepo) -> None:
        self.session = session

    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _JsonParser:
    async def invoke(self, payload: Any) -> Any:
        if isinstance(payload, str):
            return json.loads(payload)
        return payload


class _ScriptedLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)

    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        if not self._responses:
            raise AssertionError("LLM 响应脚本已耗尽")
        return self._responses.pop(0)


class _FakeBrainstormTool(BaseTool):
    name = "skill_brainstormer"

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    @tool(
        name="brainstorm_skill",
        description="生成 skill 蓝图",
        parameters={"description": {"type": "string"}},
        required=["description"],
    )
    async def brainstorm_skill(self, description: str) -> ToolResult:
        self._calls.append("brainstorm_skill")
        payload = {"skill_name": "meeting-audio-analyzer"}
        return ToolResult(
            success=True,
            message="Skill 蓝图预览",
            data={
                **payload,
                "blueprint": payload,
                "blueprint_json": json.dumps(payload, ensure_ascii=False),
            },
        )


class _FakeCreateSkillTool(BaseTool):
    name = "skill_creator"

    def __init__(self, calls: list[str]) -> None:
        super().__init__()
        self._calls = calls

    @tool(
        name="generate_skill",
        description="生成 skill",
        parameters={"description": {"type": "string"}},
        required=["description"],
    )
    async def generate_skill(self, description: str) -> ToolResult:
        self._calls.append("generate_skill")
        return ToolResult(
            success=True,
            message="Skill 代码生成并验证通过",
            data={
                "skill_data": json.dumps(
                    {
                        "skill_md": "# Skill",
                        "manifest": {"tools": []},
                        "scripts": [],
                        "dependencies": [],
                    }
                )
            },
        )

    @tool(
        name="install_skill",
        description="安装 skill",
        parameters={"skill_data": {"type": "string"}},
        required=["skill_data"],
    )
    async def install_skill(self, skill_data: str) -> ToolResult:
        self._calls.append("install_skill")
        return ToolResult(success=True, message="Skill 安装完成")


def _tool_call_response(call_id: str, function_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }


def _final_step_response(result: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": json.dumps(
            {
                "success": True,
                "result": result,
                "attachments": [],
            },
            ensure_ascii=False,
        ),
        "tool_calls": [],
    }


def _build_agent(
    repo: _SharedSessionRepo,
    llm: _ScriptedLLM,
    tools: list[BaseTool],
) -> ReActAgent:
    return ReActAgent(
        uow_factory=lambda: _DummyUoW(repo),
        session_id="s-skill-flow",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=llm,
        json_parser=_JsonParser(),
        tools=tools,
    )


def _build_plan() -> tuple[Plan, Step]:
    plan = Plan(language="zh", steps=[Step(description="创建一个新的 skill")])
    return plan, plan.steps[0]


async def test_waiting_session_recovers_skill_creation_state_across_turns() -> None:
    repo = _SharedSessionRepo()
    calls: list[str] = []
    tools: list[BaseTool] = [
        _FakeBrainstormTool(calls),
        _FakeCreateSkillTool(calls),
    ]

    first_agent = _build_agent(
        repo,
        _ScriptedLLM(
            [
                _tool_call_response(
                    "brainstorm-1",
                    "brainstorm_skill",
                    {"description": "创建一个会议音频分析 skill"},
                )
            ]
        ),
        tools,
    )
    plan, step = _build_plan()
    first_events = [
        event
        async for event in first_agent.execute_step(
            plan, step, Message(message="先给我看蓝图")
        )
    ]

    assert any(isinstance(event, WaitEvent) for event in first_events)
    assert repo._skill_creation_state is not None
    assert repo._skill_creation_state.pending_action == "generate"
    assert calls == ["brainstorm_skill"]

    second_agent = _build_agent(
        repo,
        _ScriptedLLM(
            [
                _tool_call_response(
                    "generate-1",
                    "generate_skill",
                    {"description": "创建一个会议音频分析 skill"},
                )
            ]
        ),
        tools,
    )
    await second_agent.roll_back(Message(message="好的"))
    plan, step = _build_plan()
    second_events = [
        event
        async for event in second_agent.execute_step(plan, step, Message(message="好的"))
    ]

    assert any(isinstance(event, WaitEvent) for event in second_events)
    assert repo._skill_creation_state is not None
    assert repo._skill_creation_state.pending_action == "install"
    assert calls == ["brainstorm_skill", "generate_skill"]


async def test_skill_creation_requires_real_confirmation_end_to_end() -> None:
    repo = _SharedSessionRepo()
    calls: list[str] = []
    tools: list[BaseTool] = [
        _FakeBrainstormTool(calls),
        _FakeCreateSkillTool(calls),
    ]

    first_agent = _build_agent(
        repo,
        _ScriptedLLM(
            [
                _tool_call_response(
                    "brainstorm-1",
                    "brainstorm_skill",
                    {"description": "创建一个会议音频分析 skill"},
                )
            ]
        ),
        tools,
    )
    plan, step = _build_plan()
    first_events = [
        event
        async for event in first_agent.execute_step(
            plan, step, Message(message="先给我看蓝图")
        )
    ]
    assert any(isinstance(event, WaitEvent) for event in first_events)

    second_agent = _build_agent(
        repo,
        _ScriptedLLM(
            [
                _tool_call_response(
                    "generate-1",
                    "generate_skill",
                    {"description": "创建一个会议音频分析 skill"},
                )
            ]
        ),
        tools,
    )
    await second_agent.roll_back(Message(message="好的，继续生成吧"))
    plan, step = _build_plan()
    second_events = [
        event
        async for event in second_agent.execute_step(
            plan, step, Message(message="好的，继续生成吧")
        )
    ]
    assert any(isinstance(event, WaitEvent) for event in second_events)
    assert repo._skill_creation_state is not None
    assert repo._skill_creation_state.pending_action == "install"

    skill_data = repo._skill_creation_state.skill_data
    third_agent = _build_agent(
        repo,
        _ScriptedLLM(
            [
                _tool_call_response(
                    "install-1",
                    "install_skill",
                    {"skill_data": skill_data},
                ),
                _final_step_response("Skill 已安装"),
            ]
        ),
        tools,
    )
    await third_agent.roll_back(Message(message="可以，安装"))
    plan, step = _build_plan()
    third_events = [
        event
        async for event in third_agent.execute_step(
            plan, step, Message(message="可以，安装")
        )
    ]

    assert not any(isinstance(event, WaitEvent) for event in third_events)
    assert calls == ["brainstorm_skill", "generate_skill", "install_skill"]
    assert repo._skill_creation_state is None
    assert any(
        isinstance(event, StepEvent) for event in third_events
    )
    assert any(
        isinstance(event, MessageEvent) and event.message == "Skill 已安装"
        for event in third_events
    )
