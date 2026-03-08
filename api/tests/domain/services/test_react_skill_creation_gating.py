from __future__ import annotations

from typing import Any

import pytest
from app.domain.models.app_config import AgentConfig
from app.domain.models.event import (
    MessageEvent,
    StepEvent,
    ToolEvent,
    ToolEventStatus,
    WaitEvent,
)
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.plan import Plan, Step
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.tool_result import ToolResult
from app.domain.services.agents.react import ReActAgent
from app.domain.services.tools.base import BaseTool, tool

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _DummySessionRepo:
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
    def __init__(self) -> None:
        self.session = _DummySessionRepo()

    async def __aenter__(self) -> "_DummyUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyJsonParser:
    async def invoke(self, _payload: Any) -> Any:
        return {}


class _DummyLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "unused"}


class _DummyMessageTool(BaseTool):
    name = "message"

    @tool(
        name="message_notify_user",
        description="通知用户",
        parameters={"text": {"type": "string"}},
        required=["text"],
    )
    async def message_notify_user(self, text: str) -> ToolResult:
        return ToolResult(success=True, message=text)

    @tool(
        name="message_ask_user",
        description="询问用户",
        parameters={"text": {"type": "string"}},
        required=["text"],
    )
    async def message_ask_user(self, text: str) -> ToolResult:
        return ToolResult(success=True, message=text)


class _DummySkillCreatorTool(BaseTool):
    name = "skill_creator"

    @tool(
        name="brainstorm_skill",
        description="蓝图预览",
        parameters={"description": {"type": "string"}},
        required=["description"],
    )
    async def brainstorm_skill(self, description: str) -> ToolResult:
        return ToolResult(success=True, message=description)

    @tool(
        name="generate_skill",
        description="生成 skill",
        parameters={"description": {"type": "string"}},
        required=["description"],
    )
    async def generate_skill(self, description: str) -> ToolResult:
        return ToolResult(success=True, message=description)

    @tool(
        name="install_skill",
        description="安装 skill",
        parameters={"skill_data": {"type": "string"}},
        required=["skill_data"],
    )
    async def install_skill(self, skill_data: str) -> ToolResult:
        return ToolResult(success=True, message=skill_data)


@pytest.fixture
def agent() -> ReActAgent:
    return ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-skill-gating",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )


@pytest.fixture
def agent_with_creator_tools() -> ReActAgent:
    return ReActAgent(
        uow_factory=_DummyUoW,
        session_id="s-react-skill-gating-tools",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[_DummyMessageTool(), _DummySkillCreatorTool()],
    )


def test_intercept_generate_when_blueprint_confirmation_pending(
    agent: ReActAgent,
) -> None:
    agent._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
    )

    result = agent._intercept_tool_call("generate_skill", {"description": "x"})

    assert result is not None
    assert result.success is False
    assert result.data["code"] == "SKILL_CONFIRMATION_REQUIRED"


def test_intercept_install_when_install_confirmation_pending(agent: ReActAgent) -> None:
    agent._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
    )

    result = agent._intercept_tool_call("install_skill", {"skill_data": "{}"})

    assert result is not None
    assert result.data["pending_action"] == "install"


def test_intercept_consumes_generate_approval_token(agent: ReActAgent) -> None:
    """放行令牌在 intercept 阶段仅做放行判定，不消费；
    令牌在 execute_step 中工具成功后才会被 discard。"""
    agent._skill_creation_approved_actions = {"generate"}

    result = agent._intercept_tool_call("generate_skill", {"description": "x"})

    assert result is None
    # 令牌应保留，以便工具失败重试时 _filter_tools 继续生效
    assert "generate" in agent._skill_creation_approved_actions


def test_intercept_blocks_install_without_approval_token(agent: ReActAgent) -> None:
    agent._skill_creation_state = None
    agent._skill_creation_approved_actions = set()

    result = agent._intercept_tool_call("install_skill", {"skill_data": "{}"})

    assert result is not None
    assert result.data["code"] == "SKILL_CONFIRMATION_REQUIRED"


def test_intercept_ignores_unrelated_tools(agent: ReActAgent) -> None:
    agent._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
    )

    result = agent._intercept_tool_call("web_search", {"query": "hello"})

    assert result is None or result.data.get("code") != "SKILL_CONFIRMATION_REQUIRED"


async def test_wait_after_brainstorm_when_not_skipped(agent: ReActAgent) -> None:
    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="brainstorm-1",
            tool_name="skill_brainstormer",
            function_name="brainstorm_skill",
            function_args={"description": "创建一个会议音频分析 skill"},
            function_result=ToolResult(
                success=True,
                message="Skill 蓝图预览",
                data={
                    "skill_name": "meeting-audio-analyzer",
                    "blueprint": {"skill_name": "meeting-audio-analyzer"},
                    "blueprint_json": '{"skill_name":"meeting-audio-analyzer"}',
                },
            ),
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="创建 skill")])
    step = plan.steps[0]
    message = Message(message="先给我看蓝图")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, StepEvent) for event in events)
    assert any(isinstance(event, MessageEvent) for event in events)
    assert any(isinstance(event, WaitEvent) for event in events)


async def test_wait_after_generate_before_install(agent: ReActAgent) -> None:
    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="generate-1",
            tool_name="skill_creator",
            function_name="generate_skill",
            function_args={"description": "创建一个会议音频分析 skill"},
            function_result=ToolResult(
                success=True,
                message="Skill 生成并验证通过",
                data={
                    "skill_data": '{"skill_md":"x","manifest":{"tools":[]},"scripts":[],"dependencies":[]}'
                },
            ),
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="创建 skill")])
    step = plan.steps[0]
    message = Message(message="开始生成")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, MessageEvent) for event in events)
    assert any(isinstance(event, WaitEvent) for event in events)


async def test_skill_confirmation_required_persists_state(agent: ReActAgent) -> None:
    """SKILL_CONFIRMATION_REQUIRED 拦截应持久化等待状态，供 roll_back 使用。"""

    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="install-1",
            tool_name="skill_creator",
            function_name="install_skill",
            function_args={
                "skill_data": '{"skill_md":"x","manifest":{"tools":[]},"scripts":[],"dependencies":[]}'
            },
            function_result=ToolResult(
                success=False,
                message="SKILL_CONFIRMATION_REQUIRED",
                data={
                    "code": "SKILL_CONFIRMATION_REQUIRED",
                    "pending_action": "install",
                    "tool_name": "install_skill",
                },
            ),
            status=ToolEventStatus.CALLED,
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="安装 skill")])
    step = plan.steps[0]
    message = Message(message="安装这个 skill")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, WaitEvent) for event in events)
    # 验证等待状态已持久化
    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "install"
    assert state.last_tool_name == "install_skill"
    assert state.last_tool_call_id == "install-1"


async def test_brainstorm_does_not_wait_when_user_explicitly_says_start(
    agent: ReActAgent,
) -> None:
    async def fake_invoke(_query: str):
        yield ToolEvent(
            tool_call_id="brainstorm-1",
            tool_name="skill_brainstormer",
            function_name="brainstorm_skill",
            function_args={"description": "创建一个会议音频分析 skill"},
            function_result=ToolResult(
                success=True,
                message="Skill 蓝图预览",
                data={
                    "skill_name": "meeting-audio-analyzer",
                    "blueprint": {"skill_name": "meeting-audio-analyzer"},
                    "blueprint_json": '{"skill_name":"meeting-audio-analyzer"}',
                },
            ),
            status=ToolEventStatus.CALLED,
        )
        yield MessageEvent(role="assistant", message="继续执行")

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="创建 skill")])
    step = plan.steps[0]
    message = Message(message="需求已明确，开始吧，直接创建")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert not any(isinstance(event, WaitEvent) for event in events)


async def test_execute_step_augments_query_after_blueprint_confirmation(
    agent: ReActAgent,
) -> None:
    captured_queries: list[str] = []
    agent._skill_creation_approved_actions = {"generate"}

    async def fake_invoke(query: str):
        captured_queries.append(query)
        yield MessageEvent(
            role="assistant",
            message='{"success": true, "result": "ok", "attachments": []}',
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="创建 skill")])
    step = plan.steps[0]
    message = Message(message="可以")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, StepEvent) for event in events)
    assert captured_queries
    assert "不要再次调用 brainstorm_skill" in captured_queries[0]
    assert "优先调用 generate_skill" in captured_queries[0]


async def test_execute_step_augments_query_after_install_confirmation(
    agent: ReActAgent,
) -> None:
    captured_queries: list[str] = []
    agent._skill_creation_approved_actions = {"install"}

    async def fake_invoke(query: str):
        captured_queries.append(query)
        yield MessageEvent(
            role="assistant",
            message='{"success": true, "result": "ok", "attachments": []}',
        )

    agent.invoke = fake_invoke  # type: ignore[method-assign]

    plan = Plan(language="zh", steps=[Step(description="安装 skill")])
    step = plan.steps[0]
    message = Message(message="可以")

    events = [event async for event in agent.execute_step(plan, step, message)]

    assert any(isinstance(event, StepEvent) for event in events)
    assert captured_queries
    assert "直接调用 install_skill" in captured_queries[0]
    assert "不要重新调用 brainstorm_skill 或 generate_skill" in captured_queries[0]


def test_available_tools_are_gated_after_blueprint_confirmation(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"generate"}

    tool_names = {
        item["function"]["name"]
        for item in agent_with_creator_tools._get_available_tools()
    }

    assert "brainstorm_skill" not in tool_names
    assert "install_skill" not in tool_names
    assert "generate_skill" in tool_names
    assert "message_notify_user" in tool_names
    assert "message_ask_user" in tool_names


def test_available_tools_are_gated_after_install_confirmation(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"install"}

    tool_names = {
        item["function"]["name"]
        for item in agent_with_creator_tools._get_available_tools()
    }

    assert "brainstorm_skill" not in tool_names
    assert "generate_skill" not in tool_names
    assert "install_skill" in tool_names
    assert "message_notify_user" in tool_names
    assert "message_ask_user" in tool_names


def test_system_prompt_contains_generate_gate_notice(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"generate"}

    prompt = agent_with_creator_tools._build_effective_system_prompt()

    assert "Skill Creation Tool Gate" in prompt
    assert "蓝图确认后的恢复阶段" in prompt
    assert "generate_skill" in prompt
    assert "其他 Skill Creator 工具当前不可调用" in prompt


def test_system_prompt_contains_install_gate_notice(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"install"}

    prompt = agent_with_creator_tools._build_effective_system_prompt()

    assert "Skill Creation Tool Gate" in prompt
    assert "安装确认后的恢复阶段" in prompt
    assert "install_skill" in prompt
    assert "其他 Skill Creator 工具当前不可调用" in prompt


def test_tool_choice_is_required_during_generate_resume(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"generate"}

    assert agent_with_creator_tools._get_tool_choice() == "required"


def test_tool_choice_is_required_during_install_resume(
    agent_with_creator_tools: ReActAgent,
) -> None:
    agent_with_creator_tools._skill_creation_approved_actions = {"install"}

    assert agent_with_creator_tools._get_tool_choice() == "required"
