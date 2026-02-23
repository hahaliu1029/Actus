from __future__ import annotations

import pytest
from app.domain.models.app_config import (
    A2AConfig,
    A2AServerConfig,
    AgentConfig,
    MCPConfig,
    MCPServerConfig,
    MCPTransport,
)
from app.domain.models.skill import Skill, SkillRuntimeType, SkillSourceType
from app.domain.models.user_tool_preference import ToolType
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _NoopSessionRepository:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, object]] = []

    async def update_status(self, session_id: str, status) -> None:
        self.status_updates.append((session_id, status))

    async def add_event(self, session_id: str, event) -> None:
        return None

    async def update_title(self, session_id: str, title: str) -> None:
        return None

    async def update_latest_message(self, session_id: str, message: str, timestamp) -> None:
        return None

    async def increment_unread_message_count(self, session_id: str) -> None:
        return None


class _NoopUoW:
    def __init__(self) -> None:
        self.session = _NoopSessionRepository()

    async def __aenter__(self) -> "_NoopUoW":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


def _uow_factory() -> _NoopUoW:
    return _NoopUoW()


class _DummyFlow:
    def __init__(self, **kwargs) -> None:
        self.skill_contexts: list[str] = []

    def set_skill_context(self, skill_context: str) -> None:
        self.skill_contexts.append(skill_context)

    async def invoke(self, message):
        if False:
            yield message


class _EmptyInputStream:
    async def is_empty(self) -> bool:
        return True


class _DummyTask:
    def __init__(self) -> None:
        self.input_stream = _EmptyInputStream()
        self.output_stream = None


class _FakeSandbox:
    async def ensure_sandbox(self) -> None:
        return None


class _FakeMCPTool:
    def __init__(self) -> None:
        self.initialized_with: MCPConfig | None = None

    async def initialize(self, mcp_config: MCPConfig) -> None:
        self.initialized_with = mcp_config

    async def cleanup(self) -> None:
        return None


class _FakeA2ATool:
    def __init__(self) -> None:
        self.initialized_with: A2AConfig | None = None
        self.manager = None

    async def initialize(self, a2a_config: A2AConfig) -> None:
        self.initialized_with = a2a_config

    async def cleanup(self) -> None:
        return None


class _FakeSkillTool:
    def __init__(self) -> None:
        self.initialized_with: list[Skill] | None = None

    async def initialize(self, skills: list[Skill]) -> None:
        self.initialized_with = skills

    async def cleanup(self) -> None:
        return None


async def test_invoke_applies_user_preferences_before_tool_initialization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.domain.services.agent_task_runner.PlannerReActFlow",
        _DummyFlow,
    )

    runner = AgentTaskRunner(
        uow_factory=_uow_factory,
        llm=object(),
        agent_config=AgentConfig(max_iterations=100, max_retries=3, max_search_results=10),
        mcp_config=MCPConfig(
            mcpServers={
                "mcp-enabled": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://enabled.example.com/mcp",
                ),
                "mcp-disabled-by-user": MCPServerConfig(
                    transport=MCPTransport.STREAMABLE_HTTP,
                    enabled=True,
                    url="https://disabled.example.com/mcp",
                ),
            }
        ),
        a2a_config=A2AConfig(
            a2a_servers=[
                A2AServerConfig(id="a2a-enabled", base_url="https://a2a-enabled.example.com", enabled=True),
                A2AServerConfig(id="a2a-disabled-by-user", base_url="https://a2a-disabled.example.com", enabled=True),
            ]
        ),
        session_id="session-1",
        user_id="user-1",
        file_storage=object(),
        json_parser=object(),
        browser=object(),
        search_engine=object(),
        sandbox=_FakeSandbox(),
    )

    fake_mcp_tool = _FakeMCPTool()
    fake_a2a_tool = _FakeA2ATool()
    fake_skill_tool = _FakeSkillTool()
    runner._mcp_tool = fake_mcp_tool
    runner._a2a_tool = fake_a2a_tool
    runner._skill_tool = fake_skill_tool

    preferences_by_tool_type = {
        ToolType.MCP: {"mcp-disabled-by-user": False},
        ToolType.A2A: {"a2a-disabled-by-user": False},
        ToolType.SKILL: {"skill-disabled-by-user": False},
    }

    async def fake_load_user_preferences_map(tool_type: ToolType) -> dict[str, bool]:
        return preferences_by_tool_type.get(tool_type, {})

    async def fake_load_enabled_skills() -> list[Skill]:
        return [
            Skill(
                id="skill-enabled",
                slug="skill-enabled",
                name="Skill Enabled",
                description="enabled skill",
                source_type=SkillSourceType.GITHUB,
                source_ref="github:owner/enabled",
                runtime_type=SkillRuntimeType.NATIVE,
                manifest={"tools": []},
                enabled=True,
            ),
            Skill(
                id="skill-disabled-by-user",
                slug="skill-disabled",
                name="Skill Disabled",
                description="disabled by preference",
                source_type=SkillSourceType.GITHUB,
                source_ref="github:owner/disabled",
                runtime_type=SkillRuntimeType.NATIVE,
                manifest={"tools": []},
                enabled=True,
            ),
        ]

    monkeypatch.setattr(runner, "_load_user_preferences_map", fake_load_user_preferences_map)
    monkeypatch.setattr(runner, "_load_enabled_skills", fake_load_enabled_skills)

    await runner.invoke(_DummyTask())

    assert fake_mcp_tool.initialized_with is not None
    assert fake_mcp_tool.initialized_with.mcpServers["mcp-enabled"].enabled is True
    assert (
        fake_mcp_tool.initialized_with.mcpServers["mcp-disabled-by-user"].enabled
        is False
    )

    assert fake_a2a_tool.initialized_with is not None
    a2a_enabled_map = {
        server.id: server.enabled for server in fake_a2a_tool.initialized_with.a2a_servers
    }
    assert a2a_enabled_map["a2a-enabled"] is True
    assert a2a_enabled_map["a2a-disabled-by-user"] is False

    assert fake_skill_tool.initialized_with is not None
    assert [skill.id for skill in fake_skill_tool.initialized_with] == ["skill-enabled"]
    assert runner._flow.skill_contexts
    assert "Skill Enabled" in runner._flow.skill_contexts[-1]
