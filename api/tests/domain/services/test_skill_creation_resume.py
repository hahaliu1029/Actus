from __future__ import annotations

from typing import Any

import pytest

from app.domain.models.app_config import AgentConfig
from app.domain.models.memory import Memory
from app.domain.models.message import Message
from app.domain.models.skill_creation_state import SkillCreationState
from app.domain.models.tool_result import ToolResult
from app.domain.services.agents.base import BaseAgent
from app.domain.services.tools.base import BaseTool

pytestmark = pytest.mark.anyio


@pytest.fixture()
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


class _LazySessionUoW:
    def __init__(self, session: _DummySessionRepo) -> None:
        self._session_repo = session

    async def __aenter__(self) -> "_LazySessionUoW":
        self.session = self._session_repo
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _DummyLLM:
    async def invoke(self, **kwargs: Any) -> dict[str, Any]:
        return {"role": "assistant", "content": "unused", "tool_calls": []}


class _DummyJsonParser:
    async def invoke(self, payload: Any) -> Any:
        return payload


class _DummyAgent(BaseAgent):
    name = "dummy"
    _system_prompt = "test system prompt"


def _build_agent() -> _DummyAgent:
    return _DummyAgent(
        uow_factory=_DummyUoW,
        session_id="s-skill-resume",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )


def _waiting_tool_call(function_name: str = "brainstorm_skill") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "function": {
                    "name": function_name,
                    "arguments": "{}",
                },
            }
        ],
    }


@pytest.fixture()
def agent() -> _DummyAgent:
    return _build_agent()


def test_classify_skill_creation_reply_affirmative(agent: _DummyAgent) -> None:
    for text in [
        "好的",
        "安装吧",
        "嗯",
        "行",
        "ok",
        "yes",
        "对",
        "是的",
        "确定",
        "确认",
        "没问题",
        "可以",
        "继续",
        "同意",
        "通过",
        "好",
        "好的，继续生成吧",
        "可以，安装",
        "嗯 开始吧",
        # Pattern-based matches
        "确认蓝图并开始生成",
        "确认蓝图",
        "确认安装",
        "开始生成",
        "可以生成",
        "可以安装",
        "开始安装",
        "继续安装",
        "确认生成",
        "同意蓝图",
        "蓝图没问题",
        "就这样吧",
        "符合预期",
        "好的，根据这个蓝图开始生产",
    ]:
        assert (
            agent._classify_skill_creation_reply(text) == "affirmative"
        ), f"{text!r} should be affirmative"


def test_classify_skill_creation_reply_negative(agent: _DummyAgent) -> None:
    for text in [
        "不安装",
        "取消",
        "不",
        "no",
        "cancel",
        "否",
        "拒绝",
        "不要",
        "不用",
        "先别安装",
        "不要继续",
    ]:
        assert (
            agent._classify_skill_creation_reply(text) == "negative"
        ), f"{text!r} should be negative"


def test_classify_skill_creation_reply_revise(agent: _DummyAgent) -> None:
    assert agent._classify_skill_creation_reply("把 output_dir 改成必填") == "revise"
    assert agent._classify_skill_creation_reply("取消后重新生成") == "revise"


async def test_roll_back_injects_saved_tool_result_on_affirmative(
    agent: _DummyAgent,
) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call())
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
        last_tool_name="brainstorm_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 蓝图预览",
            data={"blueprint": {"skill_name": "meeting-audio-analyzer"}},
        ).model_dump_json(),
    )

    await agent.roll_back(Message(message="好的"))

    last_message = agent._uow.session._memory.get_last_message()
    assert last_message is not None
    assert last_message["role"] == "tool"
    assert last_message["tool_call_id"] == "call_1"
    assert last_message["function_name"] == "brainstorm_skill"
    assert "generate" in agent._skill_creation_approved_actions
    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "generate"
    assert state.approval_status == "approved"


async def test_roll_back_accepts_structured_generate_confirmation(
    agent: _DummyAgent,
) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call())
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
        last_tool_name="brainstorm_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 蓝图预览",
            data={"blueprint": {"skill_name": "meeting-audio-analyzer"}},
        ).model_dump_json(),
    )

    await agent.roll_back(
        Message(message="根据这个蓝图开始生产", skill_confirmation_action="generate")
    )

    last_message = agent._uow.session._memory.get_last_message()
    assert last_message is not None
    assert last_message["role"] == "tool"
    assert last_message["function_name"] == "brainstorm_skill"
    assert "generate" in agent._skill_creation_approved_actions
    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "generate"
    assert state.approval_status == "approved"


async def test_roll_back_keeps_install_state_with_skill_data_on_affirmative(
    agent: _DummyAgent,
) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call("generate_skill"))
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
        last_tool_name="generate_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 代码生成并验证通过",
            data={"skill_data": "{\"skill_md\":\"# demo\"}"},
        ).model_dump_json(),
        skill_data="{\"skill_md\":\"# demo\"}",
    )

    await agent.roll_back(Message(message="确认安装"))

    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "install"
    assert state.approval_status == "approved"
    assert state.skill_data == "{\"skill_md\":\"# demo\"}"
    assert "install" in agent._skill_creation_approved_actions


async def test_roll_back_clears_state_and_rolls_back_on_negative(
    agent: _DummyAgent,
) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call("generate_skill"))
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
        last_tool_name="generate_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 代码生成并验证通过",
            data={"skill_data": "{}"},
        ).model_dump_json(),
    )

    await agent.roll_back(Message(message="先别安装"))

    assert agent._uow.session._memory.get_last_message() is None
    assert agent._uow.session._skill_creation_state is None
    assert "install" not in agent._skill_creation_approved_actions


async def test_roll_back_keeps_gate_when_structured_action_mismatches_pending_action(
    agent: _DummyAgent,
) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call("generate_skill"))
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
        last_tool_name="generate_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 代码生成并验证通过",
            data={"skill_data": "{}"},
        ).model_dump_json(),
    )

    await agent.roll_back(
        Message(message="开始生成", skill_confirmation_action="generate")
    )

    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "install"
    assert "install" not in agent._skill_creation_approved_actions


async def test_roll_back_keeps_gate_for_revise(agent: _DummyAgent) -> None:
    agent._uow.session._memory.add_message(_waiting_tool_call())
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
        last_tool_name="brainstorm_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=ToolResult(
            success=True,
            message="Skill 蓝图预览",
            data={"blueprint": {"skill_name": "meeting-audio-analyzer"}},
        ).model_dump_json(),
    )

    await agent.roll_back(Message(message="把输出语言默认改成 auto"))

    state = agent._uow.session._skill_creation_state
    assert state is not None
    assert state.pending_action == "generate"
    assert state.approval_status == "pending"
    last_message = agent._uow.session._memory.get_last_message()
    assert last_message is not None
    assert last_message["role"] == "tool"
    assert last_message["function_name"] == "brainstorm_skill"
    assert "generate" not in agent._skill_creation_approved_actions


async def test_roll_back_skips_duplicate_tool_result_when_already_in_memory(
    agent: _DummyAgent,
) -> None:
    """当 execute_step 已将工具结果写入记忆时，roll_back 不应重复添加。"""
    tool_result_json = ToolResult(
        success=True,
        message="Skill 代码生成并验证通过",
        data={"skill_data": "{\"skill_md\":\"# demo\"}"},
    ).model_dump_json()
    # 模拟 execute_step 已写入的记忆：assistant(tool_call) + tool(result)
    agent._uow.session._memory.add_message(_waiting_tool_call("generate_skill"))
    agent._uow.session._memory.add_message(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "function_name": "generate_skill",
            "content": tool_result_json,
        }
    )
    agent._uow.session._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
        last_tool_name="generate_skill",
        last_tool_call_id="call_1",
        saved_tool_result_json=tool_result_json,
        skill_data="{\"skill_md\":\"# demo\"}",
    )

    await agent.roll_back(Message(message="确认安装"))

    # 验证记忆中只有 1 条 tool result（不应重复）
    tool_results = [
        m
        for m in agent._uow.session._memory.messages
        if m.get("role") == "tool" and m.get("function_name") == "generate_skill"
    ]
    assert len(tool_results) == 1
    assert "install" in agent._skill_creation_approved_actions


async def test_skill_creation_state_helpers_support_lazy_uow_session() -> None:
    repo = _DummySessionRepo()
    agent = _DummyAgent(
        uow_factory=lambda: _LazySessionUoW(repo),
        session_id="s-lazy-uow",
        agent_config=AgentConfig(max_iterations=3, max_retries=2, max_search_results=5),
        llm=_DummyLLM(),
        json_parser=_DummyJsonParser(),
        tools=[],
    )
    repo._skill_creation_state = SkillCreationState(
        pending_action="generate",
        approval_status="pending",
    )

    await agent._ensure_skill_creation_state()
    assert agent._skill_creation_state is not None
    assert agent._skill_creation_state.pending_action == "generate"

    agent._skill_creation_state = SkillCreationState(
        pending_action="install",
        approval_status="pending",
    )
    await agent._persist_skill_creation_state()
    assert repo._skill_creation_state is not None
    assert repo._skill_creation_state.pending_action == "install"

    await agent._clear_skill_creation_state()
    assert repo._skill_creation_state is None
